"""A bounded pair adapter over the public protocol-v4 compiler and engine.

No loader, training objective, vocabulary filter, or task selection lives here.
A caller supplies prepared weights and exact pre-tokenized inputs. Static
validation never imports tensor libraries or touches the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causalab.io.env import file_env
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.registry import (
    component_width,
    get_model_info,
    model_info_from_hf_config,
)
from causalab.protocol.rules.errors import ProtocolError

SCHEMA = "causalab.paired-intervention/v1"
RESULT_SCHEMA = "causalab.paired-intervention-result/v1"


class PairRefusal(ValueError):
    """A named input or capability refusal, before a model forward."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _integer(value: Any, label: str, limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < limit:
        raise PairRefusal(
            "index_out_of_bounds", f"{label} must be an integer in [0,{limit})"
        )
    return value


def build_pair_protocol(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build one explicit donor-to-recipient replacement or zero ablation.

    Positions are zero-based token indices in each *untruncated* plain-text
    input, including tokenizer-added specials. They need not have equal IDs:
    the caller establishes correspondence; this adapter verifies the exact
    IDs addressed before execution. No alignment is guessed from text length.
    """
    if not isinstance(spec, Mapping):
        raise PairRefusal("missing_metadata", "spec must be an object")
    if spec.get("schema") != SCHEMA:
        raise PairRefusal("schema_mismatch", f"expected {SCHEMA}")
    try:
        model = spec["model"]
        key, revision, dtype = model["key"], model["revision"], model["dtype"]
        inputs = spec["inputs"]
        edit = spec["intervention"]
    except (KeyError, TypeError) as error:
        raise PairRefusal(
            "missing_metadata", "model, inputs and intervention are required"
        ) from error
    if (
        not isinstance(model, Mapping)
        or not isinstance(inputs, Mapping)
        or not isinstance(edit, Mapping)
    ):
        raise PairRefusal(
            "missing_metadata", "model, inputs and intervention must be objects"
        )
    if not isinstance(key, str) or not key:
        raise PairRefusal("missing_metadata", "model.key must be a nonempty string")
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise PairRefusal(
            "unpinned_revision", "model.revision must be a full commit hash"
        )
    if not isinstance(dtype, str) or dtype not in {"fp32", "bf16", "fp16"}:
        raise PairRefusal("unsupported_dtype", "dtype must be fp32, bf16 or fp16")
    tokenizer = spec.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or not isinstance(tokenizer.get("key"), str)
        or not tokenizer["key"]
    ):
        raise PairRefusal(
            "missing_tokenizer_identity",
            "tokenizer key and pinned revision are required",
        )
    if not isinstance(tokenizer.get("revision"), str) or not re.fullmatch(
        "[0-9a-f]{40}", tokenizer["revision"]
    ):
        raise PairRefusal(
            "unpinned_tokenizer_revision",
            "tokenizer.revision must be a full commit hash",
        )
    tokenization = spec.get("tokenization")
    if (
        not isinstance(tokenization, Mapping)
        or tokenization
        != {"add_special_tokens": True, "chat_template": False, "truncation": False}
        or any(not isinstance(value, bool) for value in tokenization.values())
    ):
        raise PairRefusal(
            "unsupported_tokenization",
            "only explicit plain, untruncated tokenizer-specials mode is supported",
        )
    info = get_model_info(key)
    if info.family != "qwen3":
        raise PairRefusal(
            "unsupported_output_semantics",
            "this adapter verifies lm_head output semantics only for the qwen3 "
            "family; other families need actual final-output capture and parity",
        )
    try:
        component, kind, layer = edit["component"], edit["kind"], edit["layer"]
        source_pos, recipient_pos = edit["source_position"], edit["recipient_position"]
        channels = edit["channels"]
        alignment = edit["alignment"]
    except (KeyError, TypeError) as error:
        raise PairRefusal(
            "missing_metadata", "all intervention fields must be explicit"
        ) from error
    if not isinstance(component, str) or component not in {
        "block_output",
        "mlp_neuron_output",
    }:
        raise PairRefusal(
            "unsupported_component",
            "only block_output and mlp_neuron_output are supported",
        )
    if not isinstance(kind, str) or kind not in {"replacement", "zero_ablation"}:
        raise PairRefusal(
            "unsupported_intervention",
            "replacement and zero_ablation are distinct interventions",
        )
    _integer(layer, "layer", info.num_layers)
    if not isinstance(alignment, str) or not alignment.strip():
        raise PairRefusal(
            "missing_alignment", "state how source and recipient positions correspond"
        )
    for role, position in (("source", source_pos), ("recipient", recipient_pos)):
        value = inputs.get(role)
        if (
            not isinstance(value, Mapping)
            or not isinstance(value.get("text"), str)
            or not value["text"]
        ):
            raise PairRefusal("missing_input", f"{role} needs nonempty text")
        ids = value.get("token_ids")
        if not isinstance(ids, list) or not ids:
            raise PairRefusal("missing_token_ids", f"{role} needs exact token IDs")
        for token_id in ids:
            _integer(token_id, f"{role}.token_id", info.vocab_size)
        _integer(position, f"{role}_position", len(ids))
    dims = {}
    if channels is not None:
        if not isinstance(channels, list) or not channels:
            raise PairRefusal(
                "invalid_channels",
                "channels is null for the full component, or a nonempty list",
            )
        width = component_width(info, component)
        for channel in channels:
            _integer(channel, "channel", width)
        if len(set(channels)) != len(channels):
            raise PairRefusal("invalid_channels", "channels must be unique")
        dims["dims"] = list(channels)
    return {
        "header": {
            "protocol_version": "4",
            "description": "Bounded observed pair intervention; no fitting or answer filtering.",
        },
        "model": {"key": key, "revision": revision, "dtype": dtype},
        "data": {
            "base": {"inputs": [inputs["recipient"]["text"]]},
            "counterfactual": {"inputs": [inputs["source"]["text"]]},
        },
        "method": {
            "intervened_models": {
                "source": {"input": "counterfactual", "reads": ["donor", "logits"]},
                "recipient": {"input": "base", "reads": ["baseline", "logits"]},
                "patched": {
                    "input": "base",
                    "reads": ["edited", "logits"],
                    "writes": ["patch"],
                },
            },
            "sites": {
                "target": {"component": component, "layers": [layer]},
                "head": {"component": "lm_head"},
            },
            "reads": {
                "donor": {"site": "target", "pos": source_pos, **dims},
                "baseline": {"site": "target", "pos": recipient_pos, **dims},
                "edited": {"site": "target", "pos": recipient_pos, **dims},
                "logits": {"site": "head", "pos": "all"},
            },
            "writes": {
                "patch": {
                    "site": "target",
                    "pos": recipient_pos,
                    **dims,
                    "do": {
                        "swap": {"read": "donor", "model": "source"}
                        if kind == "replacement"
                        else 0
                    },
                }
            },
            "save": [
                *(
                    {
                        "read": "logits",
                        "model": name,
                        "file_path": f"{name}_logits.safetensors",
                    }
                    for name in ("source", "recipient", "patched")
                ),
                {
                    "read": "donor",
                    "model": "source",
                    "file_path": "source_activation.safetensors",
                },
                {
                    "read": "baseline",
                    "model": "recipient",
                    "file_path": "recipient_activation.safetensors",
                },
                {
                    "read": "edited",
                    "model": "patched",
                    "file_path": "patched_activation.safetensors",
                },
            ],
        },
    }


def validate_pair(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Offline dry-run: schema, static model bounds, and actual compiler rules."""
    try:
        document = build_pair_protocol(spec)
        compiled = compile_protocol(
            document,
            env=file_env(Path("."), Path(".")),
            engine="pytorch_hooks",
            data=True,
        )
    except (PairRefusal, ProtocolError) as error:
        return {
            "schema": RESULT_SCHEMA,
            "status": "unsupported",
            "code": error.code,
            "message": str(error),
        }
    return {
        "schema": RESULT_SCHEMA,
        "status": "validated_not_executed",
        "protocol_digest": compiled.campaign_digest,
        "spec_sha256": _digest(spec),
        "model": dict(spec["model"]),
        "tokenizer": dict(spec["tokenizer"]),
        "model_support": "static_registry_only; released_weight_parity_unverified",
        "unsupported_captures": ["dense_raw_gate", "dense_up_value"],
    }


def run_pair(
    spec: Mapping[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    output_directory: Path,
    environment: str,
) -> dict[str, Any]:
    """Run only caller-owned prepared weights through Causalab's public engine.

    Model identity is checked against config metadata, not cryptographically
    proven from parameter bytes. Missing runtime dependencies return a named
    unsupported state. Model failures propagate; Causalab removes its hooks.
    Output directories must be new, preserving prior evidence on any failure.
    """
    validation = validate_pair(spec)
    if validation["status"] != "validated_not_executed":
        return validation
    if not isinstance(environment, str) or not environment.strip():
        raise PairRefusal("missing_environment", "record the execution environment")
    from causalab.protocol.positions.encoding import encode

    wanted = spec["model"]
    config = getattr(model, "config", None)
    if getattr(config, "_name_or_path", None) != wanted["key"]:
        raise PairRefusal(
            "model_mismatch", "loaded config model identity differs or is unknown"
        )
    if getattr(config, "_commit_hash", None) != wanted["revision"]:
        raise PairRefusal(
            "model_revision_mismatch", "loaded config revision differs or is unknown"
        )
    tokenizer_meta = spec["tokenizer"]
    if getattr(tokenizer, "name_or_path", None) != tokenizer_meta["key"]:
        raise PairRefusal(
            "tokenizer_mismatch", "loaded tokenizer identity differs or is unknown"
        )
    tokenizer_init = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(tokenizer_init, Mapping):
        raise PairRefusal(
            "tokenizer_revision_mismatch",
            "loaded tokenizer revision metadata is malformed",
        )
    tokenizer_revision = tokenizer_init.get("_commit_hash") or tokenizer_init.get(
        "revision"
    )
    if tokenizer_revision != tokenizer_meta["revision"]:
        raise PairRefusal(
            "tokenizer_revision_mismatch",
            "loaded tokenizer revision differs or is unknown",
        )
    for role in ("source", "recipient"):
        supplied = spec["inputs"][role]
        actual = list(encode(tokenizer, [supplied["text"]]).real_ids(0))
        if actual != supplied["token_ids"]:
            raise PairRefusal(
                "token_ids_mismatch",
                f"{role} token IDs differ from the declared sequence",
            )
    expected_info = get_model_info(wanted["key"])
    try:
        actual_info = model_info_from_hf_config(wanted["key"], config)
    except (AttributeError, TypeError, ValueError) as error:
        raise PairRefusal(
            "model_geometry_unknown", "loaded model config lacks verifiable geometry"
        ) from error
    geometry_fields = (
        "hidden_size",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "family",
    )
    if any(
        getattr(actual_info, name) != getattr(expected_info, name)
        for name in geometry_fields
    ):
        raise PairRefusal(
            "model_geometry_mismatch",
            "loaded config differs from registered model geometry",
        )
    try:
        from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
        from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
        from causalab.protocol import run_protocol
    except ImportError as error:
        return {
            "schema": RESULT_SCHEMA,
            "status": "unsupported",
            "code": "missing_runtime_dependency",
            "message": str(error),
        }
    bundle = ModelBundle.from_model(model, tokenizer, **wanted)
    # from_model registers the loaded config. Re-check bounds against that tree.
    document = build_pair_protocol(spec)
    output_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = run_protocol(
        document,
        file_env(output_directory, output_directory),
        PytorchHooksEngine(
            bundle=bundle, device=bundle.devices.requested, batch_rows=1
        ),
        output_directory,
        record=True,
    )
    return {
        **validation,
        "status": "executed",
        "environment": environment,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_bytes": None,
        "memory_status": "not_measured_by_this_adapter",
        "model_support": "caller_owned_weights; config_identity_checked; weight_bytes_not_verified",
        "input_provenance": {
            role: {
                "prompt_sha256": hashlib.sha256(
                    spec["inputs"][role]["text"].encode()
                ).hexdigest(),
                "token_ids_sha256": _digest(spec["inputs"][role]["token_ids"]),
            }
            for role in ("source", "recipient")
        },
        "files": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for name, path in result.files.items()
        },
        "intervention": dict(spec["intervention"]),
        "logits_semantics": "actual lm_head output, every input position, unfiltered full vocabulary",
    }


def main(argv: list[str] | None = None) -> int:
    """Validate a pair spec and optionally emit its executable protocol JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--protocol-out", type=Path)
    args = parser.parse_args(argv)
    try:
        spec = json.loads(args.spec.read_text())
        result = validate_pair(spec)
        if result["status"] == "validated_not_executed" and args.protocol_out:
            with args.protocol_out.open("x") as target:
                json.dump(build_pair_protocol(spec), target, indent=2)
                target.write("\n")
    except json.JSONDecodeError as error:
        result = {
            "schema": RESULT_SCHEMA,
            "status": "error",
            "code": "invalid_json",
            "message": str(error),
        }
    except FileExistsError as error:
        result = {
            "schema": RESULT_SCHEMA,
            "status": "error",
            "code": "output_exists",
            "message": str(error),
        }
    except OSError as error:
        result = {
            "schema": RESULT_SCHEMA,
            "status": "error",
            "code": "io_error",
            "message": str(error),
        }
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "validated_not_executed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
