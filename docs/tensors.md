# Prepared tensor interface

Each `.safetensors` file contains one batch. Floating tensors must be finite. Masks are boolean and use `True` for valid entries. Use a consistent coordinate layout and normalization across samples. Fill invalid target and state coordinates with zeros before serialization.

| Key | Shape | Meaning |
| --- | --- | --- |
| `latents` | `B, C, T, H, W` | Encoded video with one clean observation frame followed by future frames |
| `latent_mask` | `B, T` | Valid latent frames; each sample includes a valid future frame |
| `actions` | `B, A, D` | Future action chunk in a shared coordinate layout |
| `action_mask` | `B, A, D` | Valid action coordinates |
| `text` | `B, L, E` | Encoded instruction |
| `text_mask` | `B, L` | Valid text tokens, with at least one per sample |
| `state` | `B, S` | Optional proprioceptive state |
| `state_mask` | `B, S` | Required when `state` is supplied |
| `consequence` | `B, M, 3K` | Motion Track displacement and visibility coordinates |
| `consequence_mask` | `B, M, 3K` | Valid Motion Track coordinates |

`A`, `D`, and `S` match `chunk`, `action_dim`, and `state_dim` in the agent configuration. `M` matches `consequence_steps`; `3K` matches `consequence_dim`. `C` and `E` match the supplied video backbone. Spatial latent dimensions must be divisible by its patch size. The temporal patch size is one.

With a configured Motion Track head, every batch includes `consequence` and `consequence_mask`. Samples without track labels use a zero tensor and an all-false mask. The same convention applies to missing action labels. These modalities are replaced by noise at level one before the forward pass. Their supervised losses are masked out. Missing state uses the learned null token.

For partially labelled modalities, the objective averages squared error over valid coordinates within each sample, then averages the weighted per-sample loss over samples carrying that modality. The world loss averages over valid future frames in each sample. The clean observation prefix is excluded from the world loss.

The model keeps its parameter names compatible with the core implementation. `agent` names the action stream, and `consequence` names its Motion Track pathway. World and action heads have separate residual widths and a shared attention-head geometry.
