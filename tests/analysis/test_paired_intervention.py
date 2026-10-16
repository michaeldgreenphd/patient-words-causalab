"""Pure compiler and static shape tests; every prompt is synthetic."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from causalab.analysis.paired_intervention import build_pair_protocol, validate_pair
from causalab.protocol.registry import get_model_info, model_info_from_hf_config

pytestmark = pytest.mark.unit


def sample_spec() -> dict[str, Any]:
    return {
        "schema": "causalab.paired-intervention/v1",
        "model": {
            "key": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "dtype": "bf16",
        },
        "tokenizer": {
            "key": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        },
        "inputs": {
            "source": {"text": "alpha beta", "token_ids": [1, 2]},
            "recipient": {"text": "gamma delta epsilon", "token_ids": [3, 4, 5]},
        },
        "tokenization": {
            "add_special_tokens": True,
            "chat_template": False,
            "truncation": False,
        },
        "intervention": {
            "component": "mlp_neuron_output",
            "kind": "replacement",
            "layer": 1,
            "source_position": 1,
            "recipient_position": 2,
            "channels": [0, 2],
            "alignment": "prespecified last input position",
        },
    }


def test_registered_exact_model_matches_byte_pinned_config() -> None:
    path = Path(__file__).parents[1] / "protocol/fixtures/qwen3_4b_config.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"
    )
    raw = json.loads(path.read_text())
    raw["dtype"] = raw["torch_dtype"]
    actual = model_info_from_hf_config("Qwen/Qwen3-4B", SimpleNamespace(**raw))
    registered = get_model_info("Qwen/Qwen3-4B")
    for field in (
        "hidden_size",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "family",
        "native_dtype",
    ):
        assert getattr(actual, field) == getattr(registered, field)
    assert raw["architectures"] == ["Qwen3ForCausalLM"]
    assert raw["tie_word_embeddings"] is True


@pytest.mark.parametrize("component", ["block_output", "mlp_neuron_output"])
@pytest.mark.parametrize("kind", ["replacement", "zero_ablation"])
def test_protocol_uses_compiler_and_distinct_original_outputs(
    component: str, kind: str
) -> None:
    spec = sample_spec()
    spec["intervention"].update(component=component, kind=kind)
    before = copy.deepcopy(spec)
    result = validate_pair(spec)
    assert result["status"] == "validated_not_executed", result
    assert len(result["protocol_digest"]) == 64
    assert spec == before
    doc = build_pair_protocol(spec)
    assert doc["method"]["reads"]["logits"]["pos"] == "all"
    assert {s["model"] for s in doc["method"]["save"] if s["read"] == "logits"} == {
        "source",
        "recipient",
        "patched",
    }
    assert doc["method"]["writes"]["patch"]["do"] == {
        "swap": {"read": "donor", "model": "source"} if kind == "replacement" else 0
    }
    assert "train" not in doc["method"]
    assert result["unsupported_captures"] == ["dense_raw_gate", "dense_up_value"]


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("component", "mlp_activation", "unsupported_component"),
        ("kind", "rescue", "unsupported_intervention"),
        ("layer", 36, "index_out_of_bounds"),
        ("layer", True, "index_out_of_bounds"),
        ("source_position", 2, "index_out_of_bounds"),
        ("recipient_position", -1, "index_out_of_bounds"),
        ("channels", [], "invalid_channels"),
        ("channels", [0, 0], "invalid_channels"),
        ("channels", [9728], "index_out_of_bounds"),
        ("alignment", "", "missing_alignment"),
    ],
)
def test_named_refusals(field: str, value: Any, code: str) -> None:
    spec = sample_spec()
    spec["intervention"][field] = value
    result = validate_pair(spec)
    assert result["status"] == "unsupported"
    assert result["code"] == code



def test_registered_post_head_model_refuses_before_execution(tmp_path: Path) -> None:
    from causalab.analysis.paired_intervention import run_pair

    spec = sample_spec()
    spec["model"]["key"] = "google/gemma-2-2b"
    spec["tokenizer"]["key"] = spec["model"]["key"]
    result = validate_pair(spec)
    assert result["status"] == "unsupported"
    assert result["code"] == "unsupported_output_semantics"
    assert run_pair(
        spec, model=None, tokenizer=None, output_directory=tmp_path / "run",
        environment="synthetic_test",
    ) == result
    assert not (tmp_path / "run").exists()


def test_output_semantics_uses_registry_family_not_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from causalab.analysis import paired_intervention

    spec = sample_spec()
    monkeypatch.setattr(
        paired_intervention, "get_model_info",
        lambda _key: SimpleNamespace(family="gemma2"),
    )
    result = validate_pair(spec)
    assert result["status"] == "unsupported"
    assert result["code"] == "unsupported_output_semantics"


def test_unregistered_variant_is_not_aliased() -> None:
    spec = sample_spec()
    spec["model"]["key"] = "Qwen/Qwen3-4B-Base"
    result = validate_pair(spec)
    assert result["status"] == "unsupported"
    assert result["code"] == "V4"


def test_unpinned_and_missing_metadata_refuse() -> None:
    spec = sample_spec()
    spec["model"]["revision"] = "main"
    assert validate_pair(spec)["code"] == "unpinned_revision"
    del spec["model"]
    assert validate_pair(spec)["code"] == "missing_metadata"


def test_tokenization_mode_must_be_explicit() -> None:
    spec = sample_spec()
    spec["tokenization"]["truncation"] = True
    assert validate_pair(spec)["code"] == "unsupported_tokenization"


def test_full_component_and_changed_alignment_change_identity() -> None:
    spec = sample_spec()
    before = validate_pair(spec)
    spec["intervention"]["channels"] = None
    assert validate_pair(spec)["protocol_digest"] != before["protocol_digest"]
    spec["intervention"]["alignment"] = "separately prespecified boundary"
    assert validate_pair(spec)["spec_sha256"] != before["spec_sha256"]


def test_loaded_model_mismatch_refuses_without_backend_import(tmp_path: Path) -> None:
    from causalab.analysis.paired_intervention import PairRefusal, run_pair

    spec = sample_spec()
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="Qwen/Qwen3-4B-Base"))
    with pytest.raises(PairRefusal) as caught:
        run_pair(
            spec,
            model=model,
            tokenizer=None,
            output_directory=tmp_path / "new",
            environment="synthetic_test",
        )
    assert caught.value.code == "model_mismatch"
    assert not (tmp_path / "new").exists()


def test_loaded_revision_mismatch_refuses(tmp_path: Path) -> None:
    from causalab.analysis.paired_intervention import PairRefusal, run_pair

    spec = sample_spec()
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path=spec["model"]["key"], _commit_hash="0" * 40
        )
    )
    with pytest.raises(PairRefusal) as caught:
        run_pair(
            spec,
            model=model,
            tokenizer=None,
            output_directory=tmp_path / "new",
            environment="synthetic_test",
        )
    assert caught.value.code == "model_revision_mismatch"


def test_exact_input_ids_are_checked_before_backend_import(tmp_path: Path) -> None:
    from causalab.analysis.paired_intervention import PairRefusal, run_pair

    spec = sample_spec()
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path=spec["model"]["key"], _commit_hash=spec["model"]["revision"]
        )
    )

    class Tokenizer:
        name_or_path = "Qwen/Qwen3-4B"
        init_kwargs: ClassVar[dict[str, str]] = {
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c"
        }
        bos_token_id = None
        padding_side = "left"
        pad_token_id = 0

        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            return {
                "input_ids": [[7, 8]],
                "attention_mask": [[1, 1]],
                "offset_mapping": [[(0, 5), (6, 10)]],
            }

    with pytest.raises(PairRefusal) as caught:
        run_pair(
            spec,
            model=model,
            tokenizer=Tokenizer(),
            output_directory=tmp_path / "new",
            environment="synthetic_test",
        )
    assert caught.value.code == "token_ids_mismatch"
    assert not (tmp_path / "new").exists()


def test_cli_emits_protocol_without_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from causalab.analysis.paired_intervention import main

    spec_path, out_path = tmp_path / "spec.json", tmp_path / "protocol.json"
    spec_path.write_text(json.dumps(sample_spec()))
    assert main([str(spec_path), "--protocol-out", str(out_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated_not_executed"
    assert json.loads(out_path.read_text())["header"]["protocol_version"] == "4"
    before = out_path.read_bytes()
    assert main([str(spec_path), "--protocol-out", str(out_path)]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "output_exists"
    assert out_path.read_bytes() == before


def test_tokenizer_revision_is_required_and_stamped() -> None:
    spec = sample_spec()
    result = validate_pair(spec)
    assert result["tokenizer"] == spec["tokenizer"]
    assert result["model"] == spec["model"]
    spec["tokenizer"]["revision"] = "main"
    assert validate_pair(spec)["code"] == "unpinned_tokenizer_revision"
    del spec["tokenizer"]
    assert validate_pair(spec)["code"] == "missing_tokenizer_identity"


@pytest.mark.parametrize("revision", [None, "0" * 40])
def test_loaded_tokenizer_revision_must_match(
    tmp_path: Path, revision: str | None
) -> None:
    from causalab.analysis.paired_intervention import PairRefusal, run_pair

    spec = sample_spec()
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path=spec["model"]["key"], _commit_hash=spec["model"]["revision"]
        )
    )
    tokenizer = SimpleNamespace(
        name_or_path=spec["tokenizer"]["key"], init_kwargs={"revision": revision}
    )
    with pytest.raises(PairRefusal) as caught:
        run_pair(
            spec,
            model=model,
            tokenizer=tokenizer,
            output_directory=tmp_path / "new",
            environment="synthetic_test",
        )
    assert caught.value.code == "tokenizer_revision_mismatch"


@pytest.mark.parametrize("field", ["dtype", "component", "kind"])
def test_unhashable_enums_produce_structured_refusals(field: str) -> None:
    spec = sample_spec()
    spec["model" if field == "dtype" else "intervention"][field] = []
    assert validate_pair(spec)["status"] == "unsupported"


def test_resized_model_cannot_replace_registered_geometry(tmp_path: Path) -> None:
    from causalab.analysis.paired_intervention import PairRefusal, run_pair

    spec = sample_spec()
    raw = json.loads(
        (
            Path(__file__).parents[1] / "protocol/fixtures/qwen3_4b_config.json"
        ).read_text()
    )
    raw.update(
        _name_or_path=spec["model"]["key"], _commit_hash=spec["model"]["revision"]
    )
    raw["vocab_size"] += 1
    model = SimpleNamespace(config=SimpleNamespace(**raw))

    class Tokenizer:
        name_or_path = "Qwen/Qwen3-4B"
        init_kwargs: ClassVar[dict[str, str]] = {
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c"
        }
        bos_token_id = None
        padding_side = "left"
        pad_token_id = 0

        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            ids = [1, 2] if texts == ["alpha beta"] else [3, 4, 5]
            return {
                "input_ids": [ids],
                "attention_mask": [[1] * len(ids)],
                "offset_mapping": [[(0, 1)] * len(ids)],
            }

    with pytest.raises(PairRefusal) as caught:
        run_pair(
            spec,
            model=model,
            tokenizer=Tokenizer(),
            output_directory=tmp_path / "new",
            environment="synthetic_test",
        )
    assert caught.value.code == "model_geometry_mismatch"
    assert get_model_info("Qwen/Qwen3-4B").vocab_size == 151936
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    "value", [None, [], "scalar", 3, {"schema": "causalab.paired-intervention/v1"}]
)
def test_malformed_root_returns_named_refusal(value: Any) -> None:
    result = validate_pair(value)
    assert result["status"] == "unsupported"
    assert isinstance(result["code"], str)


@pytest.mark.parametrize(
    "field", ["model", "inputs", "intervention", "tokenizer", "tokenization"]
)
@pytest.mark.parametrize("value", [None, [], 3, "scalar"])
def test_malformed_nested_objects_return_named_refusals(field: str, value: Any) -> None:
    spec = sample_spec()
    spec[field] = value
    result = validate_pair(spec)
    assert result["status"] == "unsupported"
    assert isinstance(result["code"], str)


def test_cli_bad_json_and_missing_file_are_structured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from causalab.analysis.paired_intervention import main

    source = tmp_path / "spec.json"
    assert main([str(source)]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "io_error"
    source.write_text('{"truncated":')
    assert main([str(source)]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "invalid_json"


def test_tokenization_flags_reject_numeric_coercion() -> None:
    spec = sample_spec()
    spec["tokenization"]["add_special_tokens"] = 1
    assert validate_pair(spec)["code"] == "unsupported_tokenization"
