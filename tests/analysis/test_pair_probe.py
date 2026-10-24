"""Random tiny Qwen3 architecture only; no pretrained weights or task data."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from causalab.neural.shared.model_tree import stream_at
from causalab.neural.shared.pair_probe import (
    capture_qwen_dense_product,
    temporary_pair_patch,
)
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.registry import model_info_from_hf_config
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import SiteSpec

pytestmark = pytest.mark.unit


@pytest.fixture
def bundle() -> Any:
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
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    return SimpleNamespace(
        model=model,
        info=model_info_from_hf_config("synthetic/tiny-qwen3", config),
        key="synthetic/tiny-qwen3",
        mixer_at=lambda layer: model.model.layers[layer].self_attn,
        stream_at=lambda layer: stream_at(
            model.model.layers, layer, key="synthetic/tiny-qwen3"
        ),
    )


def _hooks(bundle: Any) -> int:
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks)
        for m in bundle.model.modules()
    )


def _capture(bundle: Any, component: str, ids: torch.Tensor) -> torch.Tensor:
    site = resolve_site(bundle, SiteSpec(component=component, layers=(0,)))
    seen = []
    handle = (
        site.module.register_forward_hook(
            lambda _m, _a, out: seen.append(out.detach().clone())
        )
        if site.kind == "out"
        else site.module.register_forward_pre_hook(
            lambda _m, args: seen.append(args[0].detach().clone())
        )
    )
    try:
        bundle.model(ids)
    finally:
        handle.remove()
    return seen[0]


@pytest.mark.parametrize("component", ["block_output", "mlp_neuron_output"])
def test_self_patch_identity_and_cleanup(bundle: Any, component: str) -> None:
    ids = torch.tensor([[1, 2, 3]])
    expected = bundle.model(ids).logits
    source = _capture(bundle, component, ids)
    with temporary_pair_patch(
        bundle,
        component=component,
        layer=0,
        source=source,
        source_position=2,
        recipient_position=2,
        channels=[0, 2],
        kind="replacement",
    ) as report:
        actual = bundle.model(ids).logits
    assert torch.equal(actual, expected)
    assert report["edit_l2"] == 0
    assert _hooks(bundle) == 0


def test_product_preserves_signed_gate_and_value(bundle: Any) -> None:
    ids = torch.tensor([[1, 2, 3, 4]])
    with capture_qwen_dense_product(bundle, 0) as captured:
        bundle.model(ids)
    assert torch.equal(
        captured["gate_activation"], torch.nn.functional.silu(captured["raw_gate"])
    )
    assert torch.equal(
        captured["product"], captured["gate_activation"] * captured["value"]
    )
    assert (captured["product"] > 0).any() and (captured["product"] < 0).any()
    assert not torch.equal(captured["product"], captured["gate_activation"])
    assert _hooks(bundle) == 0


@pytest.mark.parametrize("kind", ["replacement", "zero_ablation"])
def test_patch_touches_only_selected_position_and_channels(
    bundle: Any, kind: str
) -> None:
    ids = torch.tensor([[1, 2, 3]])
    source = _capture(bundle, "mlp_neuron_output", torch.tensor([[4, 5, 6]]))
    baseline = _capture(bundle, "mlp_neuron_output", ids)
    with temporary_pair_patch(
        bundle,
        component="mlp_neuron_output",
        layer=0,
        source=source,
        source_position=1,
        recipient_position=2,
        channels=[0, 2],
        kind=kind,
    ) as report:
        actual = _capture(bundle, "mlp_neuron_output", ids)
    expected = baseline.clone()
    expected[0, 2, [0, 2]] = source[0, 1, [0, 2]] if kind == "replacement" else 0
    assert torch.equal(actual, expected)
    assert report["edit_l2"] > 0
    assert _hooks(bundle) == 0


def test_cleanup_when_forward_raises(bundle: Any) -> None:
    ids = torch.tensor([[1, 2]])
    source = _capture(bundle, "block_output", ids)
    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        temporary_pair_patch(
            bundle,
            component="block_output",
            layer=0,
            source=source,
            source_position=1,
            recipient_position=1,
            channels=None,
            kind="replacement",
        ),
        capture_qwen_dense_product(bundle, 0),
    ):
        bundle.model(ids)
        raise RuntimeError("synthetic failure")
    assert _hooks(bundle) == 0


@pytest.mark.parametrize("channels", [[], [0, 0], [-1], [12], [True]])
def test_invalid_channels_refuse_before_hook(bundle: Any, channels: list[int]) -> None:
    source = _capture(bundle, "mlp_neuron_output", torch.tensor([[1, 2]]))
    with (
        pytest.raises(ProtocolError, match="channels"),
        temporary_pair_patch(
            bundle,
            component="mlp_neuron_output",
            layer=0,
            source=source,
            source_position=1,
            recipient_position=1,
            channels=channels,
            kind="replacement",
        ),
    ):
        pass
    assert _hooks(bundle) == 0


def test_wrong_family_is_not_qwen_by_width(bundle: Any) -> None:
    bundle.info = SimpleNamespace(family="gemma2")
    with (
        pytest.raises(ProtocolError, match="Qwen3"),
        capture_qwen_dense_product(bundle, 0),
    ):
        pass
    assert _hooks(bundle) == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_source_refuses_and_leaves_no_hook(bundle: Any, bad: float) -> None:
    source = _capture(bundle, "block_output", torch.tensor([[1, 2]]))
    source[0, 1, 0] = bad
    with (
        pytest.raises(ProtocolError, match="finite"),
        temporary_pair_patch(
            bundle,
            component="block_output",
            layer=0,
            source=source,
            source_position=1,
            recipient_position=1,
            channels=[0],
            kind="replacement",
        ),
    ):
        pass
    assert _hooks(bundle) == 0


def test_nonfinite_recipient_refuses_and_removes_hook(bundle: Any) -> None:
    ids = torch.tensor([[1, 2]])
    source = _capture(bundle, "block_output", ids)
    block = bundle.model.model.layers[0]
    handle = block.register_forward_hook(lambda _m, _a, value: value * float("nan"))
    try:
        with (
            pytest.raises(ProtocolError, match="finite"),
            temporary_pair_patch(
                bundle,
                component="block_output",
                layer=0,
                source=source,
                source_position=1,
                recipient_position=1,
                channels=[0],
                kind="replacement",
            ),
        ):
            bundle.model(ids)
    finally:
        handle.remove()
    assert _hooks(bundle) == 0
