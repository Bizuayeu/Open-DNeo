"""run_dflash2.py MODEL_PY OUT.npz: the public DFlash2 definition (z-lab/dflash's dflash/model.py at a pinned commit) run
at a small configuration with every parameter random, its inputs, parameters and outputs written to OUT for a port to
be compared with. Dimensions differ where a port could confuse them: 6 heads over 2 key-value heads, 6 convolution
groups, 3 target layers for 2 draft layers, a context (50 rows) longer than the sliding window (40), two rows of the
batch at different positions. Run in an environment with the definition's own torch and transformers."""

import importlib.util
import json
import sys

import numpy as np
import torch

src, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("dflash_model", src)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

H, VOCAB, BLOCK, CONTEXT, TAPS = 48, 97, 4, 50, [1, 3, 4]
settings = dict(
    hidden_size=H,
    intermediate_size=80,
    num_hidden_layers=2,
    num_attention_heads=6,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=VOCAB,
    rms_norm_eps=1e-5,
    max_position_embeddings=4096,
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    sliding_window=40,
    use_sliding_window=True,
    layer_types=["sliding_attention"] * 2,
    max_window_layers=2,
    is_causal=False,
    num_target_layers=6,
    tie_word_embeddings=False,
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    dflash_config=dict(
        block_size=BLOCK,
        conv_group_size=8,
        conv_kernel_size=2,
        mask_token_id=96,
        selector_rank=5,
        selector_top_k=7,
        target_layer_ids=TAPS,
    ),
)
config = m.Qwen3Config(**settings)
torch.manual_seed(20261119)
model = m.DFlash2DraftModel(config).eval()
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.randn_like(p) * 0.3)
    head = torch.nn.Linear(H, VOCAB, bias=False)
    head.weight.copy_(torch.randn_like(head.weight) * 0.3)

    target_hidden = torch.randn(2, CONTEXT, len(TAPS) * H)
    noise = torch.randn(2, BLOCK, H)
    starts = [
        CONTEXT,
        1000,
    ]  # each row's anchor position: its context is the CONTEXT positions before it
    position_ids = torch.stack([torch.arange(s - CONTEXT, s + BLOCK) for s in starts])
    anchor_ids = torch.tensor([3, 77])

    kept = {}
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(
            lambda mod, args, output, i=i: kept.__setitem__(
                f"out.layer{i}", output.clone()
            )
        )
    hidden = model(
        position_ids=position_ids, noise_embedding=noise, target_hidden=target_hidden
    )
    kept["out.hidden"] = hidden
    kept["out.entrance"] = model.hidden_norm(model.fc(target_hidden))
    kept["out.logits"] = model.compute_logits(hidden[:, 1:], head)
    path, candidates, _ = model.propose(hidden[:, 1:], anchor_ids, head, 0.0)
    kept["out.path"] = path
    kept["out.candidates"] = torch.sort(
        candidates, dim=-1
    ).values  # topk(sorted=False) leaves the order open

arrays = {f"state.{name}": value for name, value in model.state_dict().items()}
arrays["state.head.weight"] = head.weight
arrays |= {
    "in.target_hidden": target_hidden,
    "in.noise": noise,
    "in.position_ids": position_ids,
    "in.anchor_ids": anchor_ids,
}
arrays |= kept
np.savez_compressed(
    out,
    config=np.array(json.dumps(config.to_dict(), default=str)),
    **{k: v.detach().numpy() for k, v in arrays.items()},
)
print(
    "torch",
    torch.__version__,
    "transformers",
    m.Qwen3Config.__module__.split(".")[0],
    __import__("transformers").__version__,
)
for k in sorted(arrays):
    print(f"{k:<58} {tuple(arrays[k].shape)}")
