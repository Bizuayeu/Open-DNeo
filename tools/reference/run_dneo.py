"""run_dneo.py PKG RUNTIME_PY OUT.npz: the public DNeo definition (the model repository's configuration_dneo.py,
modeling_dneo.py and reference.py as the package PKG/dneo_ref, and the head of the code repository's runtime
RUNTIME_PY) run at a small configuration with every parameter random, its inputs, parameters and outputs written to OUT
for a port to be compared with. Dimensions differ where a port could confuse them: 6 heads over 2 key-value heads, 6
convolution groups, 3 candidate target layers of 6 for 2 draft layers, three blocks a row at different anchors, rotary
frequencies rounded to bf16. No rejection memory. Run in an environment with transformers 4.57.1, the definition's."""

import importlib.util
import json
import sys
import types

import numpy as np
import torch

pkg, runtime_py, out = sys.argv[1:4]
sys.path.insert(0, pkg)
from dneo_ref.configuration_dneo import DNeoConfig  # noqa: E402
from dneo_ref.modeling_dneo import DNeoDraftModel  # noqa: E402

# the runtime imports its package's checkpoint loader at module level; the head's class does not use it
loading = types.ModuleType("dneo.loading")
loading.load_checkpoint = None
sys.modules.setdefault("dneo", types.ModuleType("dneo"))
sys.modules["dneo.loading"] = loading
spec = importlib.util.spec_from_file_location("dneo_runtime", runtime_py)
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

H, VOCAB, BLOCK, TARGET_LAYERS, PREFIX = 48, 97, 4, 6, 20
target = dict(
    hidden_size=H,
    intermediate_size=80,
    num_hidden_layers=TARGET_LAYERS,
    num_attention_heads=6,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=4096,
    rope_theta=1000000.0,
    vocab_size=VOCAB,
    tie_word_embeddings=True,
    rms_norm_eps=1e-6,
)
settings = dict(
    target_config=target,
    block_size=BLOCK,
    num_hidden_layers=2,
    target_layer_ids=[0, 3, 5],
    head_rank=5,
    folded_rank=7,
    conv_group_size=8,
    mask_token_id=96,
    rotary_frequency_dtype="bfloat16",
    shared_layer_mofa=True,
)
torch.manual_seed(20261119)
config = DNeoConfig(**settings)
model = DNeoDraftModel(config).eval()
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.randn_like(p) * 0.3)
    lm_head = torch.nn.Linear(H, VOCAB, bias=False)
    lm_head.weight.copy_(torch.randn_like(lm_head.weight) * 0.3)

    block_embeddings = torch.randn(2, 3, BLOCK, H)
    prefix_features = torch.randn(2, PREFIX, TARGET_LAYERS, H)
    anchor_positions = torch.tensor([[5, 12, 20], [3, 19, 20]])
    anchor_ids = torch.tensor([[3, 77, 41], [9, 0, 95]])

    kept = {}
    for i, layer in enumerate(model.draft.layers):
        layer.register_forward_hook(
            lambda mod, args, output, i=i: kept.__setitem__(
                f"out.layer{i}", output.clone()
            )
        )
    hidden = model(block_embeddings, prefix_features, anchor_positions)
    kept["out.hidden"] = hidden
    kept["out.context"] = model.conditioner(prefix_features)[0]

    folded = rt.FoldedDF2Head(model.ar_head)
    drafts = hidden[:, :, 1:].reshape(-1, BLOCK - 1, H)
    base, projected = lm_head(drafts), folded.project_hidden(drafts)
    previous = anchor_ids.reshape(-1)
    rows, path = [], []
    for position in range(BLOCK - 1):
        logits = folded(base[:, position], projected[:, position], previous, position)
        previous = logits.argmax(-1)
        rows.append(logits)
        path.append(previous)
    kept["out.logits"] = torch.stack(rows, 1)
    kept["out.path"] = torch.stack(path, 1)

arrays = {f"state.{name}": value for name, value in model.state_dict().items()}
arrays["state.lm_head.weight"] = lm_head.weight
arrays |= {
    "in.block_embeddings": block_embeddings,
    "in.prefix_features": prefix_features,
    "in.anchor_positions": anchor_positions,
    "in.anchor_ids": anchor_ids,
}
arrays |= kept
np.savez_compressed(
    out,
    config=np.array(json.dumps(config.to_dict(), default=str)),
    **{k: v.detach().numpy() for k, v in arrays.items()},
)
print(
    "torch", torch.__version__, "transformers", __import__("transformers").__version__
)
for k in sorted(arrays):
    print(f"{k:<58} {tuple(arrays[k].shape)} {arrays[k].dtype}")
