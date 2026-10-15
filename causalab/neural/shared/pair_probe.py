"""Small component probes using Causalab's registered site semantics.

This is a single-forward diagnostic seam, not a protocol runner, loader or
measurement writer. Use protocol-v4 for recorded experiments. It deliberately
supports only unsharded block outputs and dense down-projection inputs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from causalab.neural.shared.sites import resolve_site
from causalab.protocol.parallel import ONE
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import SiteSpec


def _refuse(message: str) -> None:
    raise ProtocolError("P4", message)


def _site(bundle: Any, component: str, layer: int) -> Any:
    if not isinstance(component, str) or component not in {
        "block_output",
        "mlp_neuron_output",
    }:
        _refuse("pair probe supports block_output or mlp_neuron_output only")
    if getattr(bundle, "geometry", ONE).world != 1:
        _refuse("pair probe requires an unsharded model")
    if (
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or not 0 <= layer < bundle.info.num_layers
    ):
        _refuse("pair probe layer is out of bounds")
    site = resolve_site(bundle, SiteSpec(component=component, layers=(layer,)))
    if site.kind != ("out" if component == "block_output" else "in"):
        _refuse("registered component side differs from the probe contract")
    return site


@contextmanager
def temporary_pair_patch(
    bundle: Any,
    *,
    component: str,
    layer: int,
    source: torch.Tensor,
    source_position: int,
    recipient_position: int,
    channels: Sequence[int] | None,
    kind: str,
) -> Iterator[dict[str, float]]:
    """Temporarily replace selected coordinates, or explicitly set them to zero.

    Caller supplies the observed source tensor and pre-aligned token positions.
    The returned edit norm is over edited coordinates of one batch row. A probe
    requires exactly one forward and removes its hook even when that forward fails.
    """
    site = _site(bundle, component, layer)
    if not isinstance(kind, str) or kind not in {"replacement", "zero_ablation"}:
        _refuse("pair probe kind must be replacement or zero_ablation")
    if source.ndim != 3 or source.shape[0] != 1 or source.shape[2] != site.shape.width:
        _refuse("source must have shape (1, positions, registered component width)")
    if not torch.is_floating_point(source) or not torch.isfinite(source).all():
        _refuse("source must contain finite floating point activations")
    for position in (source_position, recipient_position):
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            _refuse("pair probe positions must be nonnegative token indices")
    if source_position >= source.shape[1]:
        _refuse("source position is out of bounds")
    dims = list(range(source.shape[2])) if channels is None else list(channels)
    if (
        not dims
        or len(set(dims)) != len(dims)
        or any(
            isinstance(d, bool)
            or not isinstance(d, int)
            or not 0 <= d < source.shape[2]
            for d in dims
        )
    ):
        _refuse("pair probe channels must be unique valid component indices")
    source = source.detach().clone()
    report: dict[str, float] = {}
    calls = 0

    def edit(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls != 1:
            _refuse("pair probe requires exactly one forward")
        if (
            value.ndim != 3
            or value.shape[0] != 1
            or value.shape[2] != source.shape[2]
            or recipient_position >= value.shape[1]
        ):
            _refuse("recipient shape or position differs from the probe contract")
        if value.device != source.device or value.dtype != source.dtype:
            _refuse("source and recipient precision/device must match")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            _refuse("recipient must contain finite floating point activations")
        result = value.clone()
        before = value[0, recipient_position, dims]
        after = (
            source[0, source_position, dims]
            if kind == "replacement"
            else torch.zeros_like(before)
        )
        result[0, recipient_position, dims] = after
        report["edit_l2"] = float(
            torch.linalg.vector_norm((after - before).float()).item()
        )
        if not torch.isfinite(torch.tensor(report["edit_l2"], dtype=torch.float64)):
            _refuse("edit magnitude is not finite")
        return result

    def out_hook(_module: Any, _args: Any, out: Any) -> Any:
        if isinstance(out, tuple):
            return (edit(out[0]), *out[1:])
        return edit(out)

    def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return (edit(args[0]), *args[1:])

    handle = (
        site.module.register_forward_hook(out_hook)
        if site.kind == "out"
        else site.module.register_forward_pre_hook(pre_hook)
    )
    try:
        yield report
        if calls != 1:
            _refuse("pair probe site was never called")
    finally:
        handle.remove()


@contextmanager
def capture_qwen_dense_product(
    bundle: Any, layer: int
) -> Iterator[dict[str, torch.Tensor]]:
    """Capture raw gate, activated gate, value and actual down-projection input.

    Qwen3 only; model code supplies its activation function. Captures are signed,
    detached copies. No ReLU, absolute value or inferred neuron labels are applied.
    """
    if bundle.info.family != "qwen3":
        _refuse("dense product diagnostic is verified only on the Qwen3 tree")
    site = _site(bundle, "mlp_neuron_output", layer)
    mlp = bundle.model.model.layers[layer].mlp
    if site.module is not getattr(mlp, "down_proj", None) or any(
        not isinstance(getattr(mlp, name, None), torch.nn.Module)
        for name in ("gate_proj", "up_proj", "act_fn")
    ):
        _refuse("Qwen3 dense MLP tree does not expose gate/value/product modules")
    seen: dict[str, torch.Tensor] = {}
    handles = []

    def capture(name: str, value: torch.Tensor) -> None:
        if name in seen:
            _refuse("dense product diagnostic requires exactly one forward")
        seen[name] = value.detach().clone()

    try:
        for name, module in (
            ("raw_gate", mlp.gate_proj),
            ("gate_activation", mlp.act_fn),
            ("value", mlp.up_proj),
        ):
            handles.append(
                module.register_forward_hook(
                    lambda _m, _a, out, label=name: capture(label, out)
                )
            )
        handles.append(
            site.module.register_forward_pre_hook(
                lambda _m, args: capture("product", args[0])
            )
        )
        yield seen
        if set(seen) != {"raw_gate", "gate_activation", "value", "product"}:
            _refuse("dense product diagnostic did not observe every required tensor")
        if not torch.equal(seen["product"], seen["gate_activation"] * seen["value"]):
            _refuse("down-projection input differs from activation(gate) * value")
    finally:
        for handle in handles:
            handle.remove()
