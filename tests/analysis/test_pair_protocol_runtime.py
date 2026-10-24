"""Public protocol execution checks, requiring the built Causalab runtime.

These use random tiny Qwen3 weights and a local tokenizer, never Hub files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

if importlib.util.find_spec("causalab.io.fastersafetensors._core") is None:
    pytest.skip(
        "Causalab Rust fastersafetensors._core is not built", allow_module_level=True
    )

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from causalab.analysis.paired_intervention import run_pair
from causalab.protocol.registry import model_info_from_hf_config, register_model
from tests.analysis.test_paired_intervention import sample_spec

pytestmark = pytest.mark.unit


def _objects() -> tuple[Any, Any, dict[str, Any]]:
    key, revision = "synthetic/tiny-qwen3-pair", "0" * 40
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    config._name_or_path, config._commit_hash, config._attn_implementation = (
        key,
        revision,
        "eager",
    )
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    raw = Tokenizer(
        WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]")
    )
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", padding_side="left"
    )
    tokenizer.name_or_path = key
    tokenizer.init_kwargs["revision"] = revision
    register_model(model_info_from_hf_config(key, config))
    spec = sample_spec()
    spec["model"] = {"key": key, "revision": revision, "dtype": "fp32"}
    spec["tokenizer"] = {"key": key, "revision": revision}
    spec["inputs"] = {
        role: {"text": "alpha beta", "token_ids": [2, 3]}
        for role in ("source", "recipient")
    }
    spec["intervention"].update(source_position=1, recipient_position=1)
    return model, tokenizer, spec


@pytest.mark.parametrize("component", ["block_output", "mlp_neuron_output"])
def test_public_protocol_self_patch_and_original_logits(
    tmp_path: Path, component: str
) -> None:
    model, tokenizer, spec = _objects()
    spec["intervention"]["component"] = component
    result = run_pair(
        spec,
        model=model,
        tokenizer=tokenizer,
        output_directory=tmp_path / "run",
        environment="synthetic_test",
    )
    assert result["status"] == "executed"
    original = load_file(tmp_path / "run/recipient_logits.safetensors")
    patched = load_file(tmp_path / "run/patched_logits.safetensors")
    assert len(original) == len(patched) == 1
    assert torch.equal(next(iter(original.values())), next(iter(patched.values())))
    direct = model(torch.tensor([[2, 3]])).logits
    assert torch.equal(next(iter(original.values())), direct)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_public_engine_cleanup_after_forward_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, tokenizer, spec = _objects()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(model.lm_head, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        run_pair(
            spec,
            model=model,
            tokenizer=tokenizer,
            output_directory=tmp_path / "run",
            environment="synthetic_test",
        )
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
