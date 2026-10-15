# Bounded paired-intervention adapter

The adapter is an exploratory integration seam, not a new experiment framework.
Study orchestration, approved input selection, split assignment, targets, scoring,
controls and publication gates stay in the calling study repository. It does not
fit masks or infer neuron identities from layer-level lens readouts.

## Audited implementation and compatibility

Audited 2026-10-15: fork and upstream `goodfire-ai/causalab` both at
`8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e`; no fork changes preceded this work.
No AGENTS.md files were present in this checkout. The `uv.lock` environment pins
Torch 2.9.0, Transformers 5.16.1, Accelerate 1.14.0, safetensors 0.8.0 and NumPy
2.3.4. Keep this environment separate from callers' inference environments.
The compiled safetensors backend needs Rust (toolchain file: 1.98.1).

- `protocol/registry/families.py`: the structural `llama_tree` resolver maps
  `block_output` to the block output and `mlp_neuron_output` to `down_proj` input.
  `mlp_activation` is only the output of `act_fn`, not the complete gated product.
- `protocol/registry/models.py`: the new exact `Qwen/Qwen3-4B` static entry is
  sourced from [this pinned configuration](https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/config.json).
  Its fixture SHA256 is `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a`.
  The entry is **shape support**, not released-weight parity certification. It
  does not alias `-Base`, `-Instruct-2507`, Qwen3.5 or another model family.
- `demos/papers/arithmetic_neurons.md` and its fit/apply/harvest protocols use
  different-answer interchange objectives. The new adapter imports neither
  those objectives nor that paper's channel indices. Same-answer pairs remain
  same-answer pairs; both unpatched outputs are retained, including failures.
- `demos/methods/protocols/minimal_cpu.json` is protocol v4 with an explicit
  model revision. Its pure validation initially refuses an unregistered tiny
  model unless that model is registered; validation never fetches its config.
- `neural/engines/pytorch_hooks/loading.py` exposes the supported caller-owned
  `ModelBundle.from_model`; `PytorchHooksEngine(bundle=...)` avoids model loading.
  `run_protocol` remains the primary execution path. Existing model identity
  assertions are supplemented by exact config geometry and tokenizer checks.

The Rust workspace declares Apache-2.0 in Cargo.toml. No root LICENSE/COPYING or
Python package license declaration was found; preserve upstream attribution and
resolve that licensing gap before distributing copied upstream implementation.
This adapter calls public interfaces and does not copy upstream algorithm bodies.

## Contract and validation

`causalab.analysis.paired_intervention.build_pair_protocol(spec)` emits an actual
protocol-v4 document; `validate_pair(spec)` also runs Causalab's compiler. The
versioned spec has:

- `schema: causalab.paired-intervention/v1`;
- `model: {key, revision, dtype}` and `tokenizer: {key, revision}`, both revision
  values full commit hashes;
- `inputs.source` and `inputs.recipient`, each `{text, token_ids}`;
- `tokenization: {add_special_tokens: true, chat_template: false, truncation: false}`;
- `intervention: {component, kind, layer, source_position, recipient_position,
  channels, alignment}`. Components are `block_output` or `mlp_neuron_output`;
  kinds are `replacement` or `zero_ablation`; `channels` is null for a complete
  component or a nonempty list of unique indices; `alignment` records the
  prespecified correspondence, not an automatically guessed alignment.

Indices are zero-based, with specials included. Donor and recipient positions
may differ. Runtime tokenization must reproduce every declared token ID exactly.
No chat template is applied. Unsupported chat/think/truncation conventions must
be handled explicitly in a later adapter change, not silently substituted.
Full model/output logits at **all input positions** are retained separately for
source, recipient and patched executions. Selected activation tensors are saved
separately. There is no answer filter, normalization, task metric or correctness
selection in this adapter; calling-study scoring owns those decisions.

Validation returns `validated_not_executed` or a structured `unsupported` result
with a named code. The validation CLI reports malformed JSON, filesystem errors
and an existing protocol output as JSON `error` results (`invalid_json`,
`io_error`, `output_exists`) with exit status 2; it never overwrites that output.
Malformed specification shapes return named `unsupported` results. Runtime
preconditions use `PairRefusal` with `.code` and a
message. A missing execution dependency returns `missing_runtime_dependency`.
Model-forward exceptions propagate, preserving the partial output directory and
Causalab's hook cleanup; callers must record them as failed executions.

`run_pair(spec, *, model, tokenizer, output_directory, environment)` accepts only
a prepared caller-owned model. Model config name/revision and geometry must match;
tokenizer name and its `init_kwargs` resolved `_commit_hash` or explicit pinned
`revision` must match. Parameter bytes are **not cryptographically verified** by
this check; the result says so. Caller-supplied provenance is never upgraded to
verified weight identity. A reused output directory is refused. Actual model
logits and runtime are returned/stamped separately from lens scores; peak memory
is null with an explicit `not_measured_by_this_adapter` status.

From the fork checkout, this command is offline and runs no model:

```bash
python -m causalab.analysis.paired_intervention \
  tests/analysis/fixtures/synthetic_pair.json \
  --protocol-out /tmp/synthetic_pair.protocol.json
python -m causalab.cli validate /tmp/synthetic_pair.protocol.json \
  --engine pytorch_hooks
```

The fixture contains synthetic strings and invented token IDs for contract
validation only. It is not a real-model protocol; runtime token parity will
refuse those IDs. A calling study must produce its own approved, hashed spec.
Real study execution requires its authorized CI scheduler and provenance checks;
these commands do not authorize local measurement or a production trigger.

## Native diagnostic and tests

`neural/shared/pair_probe.py` provides a bounded single-forward diagnostic over
Causalab's public `resolve_site` registry. This location avoids importing the
Rust-dependent full engine merely to test tensor hooks. It is not a substitute
for a successful public protocol run. `temporary_pair_patch` handles only
unsharded block outputs and down-projection inputs, validates finite floating
activations, preserves unselected coordinates and reports edit L2. All hooks
are removed on ordinary and exceptional exits.

`capture_qwen_dense_product` checks the actual Qwen3 module tree and captures raw
gate, activated gate, up/value and the input of `down_proj`. It verifies exact
`activation(gate) * value` equality, retaining positive and negative values.
The protocol itself still has no dense raw-gate/up-value component names, so its
validation report lists those two captures as unsupported; the diagnostic does
not invent protocol enums or extend support to Gemma by width.

The offline tests use a **random tiny Qwen3 architecture**, constructed from a
config, and no pretrained checkpoint or network. They exercise self-patch
identity, separate replacement/zero-ablation behavior, selected-position/channel
isolation, signed products, nonfinite refusals, cleanup, model/tokenizer identity
rejection, and exact-model static geometry. Full `run_protocol` tests are present
but skip if the compiled safetensors backend is unavailable. The full repository
suite and released-weight baseline parity remain required in the complete locked
environment before a research run is considered ready.
