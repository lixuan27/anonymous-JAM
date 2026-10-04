# JAM

Core implementation for joint action and world pretraining.

[Interactive demo](https://anonymous.4open.science/api/repo/anonymous-JAM-B9BF/file/demo/index.html) · [Local demo](demo/index.html)

## Implementation

The world and action streams keep separate projections and residual states. At each paired layer, both attend to the combined keys and values. Clean observation tokens attend to the observed prefix. Motion Track tokens share the action stream and receive their own noise-level conditioning.

Training samples independent noise levels for future video, actions, and Motion Tracks. The objective combines flow-matching losses over valid labels. A modality without labels enters as pure noise and contributes no supervised loss.

| File | Purpose |
| --- | --- |
| `src/jam/model/jam.py` | Reciprocal attention and paired stream execution |
| `src/jam/model/action_stream.py` | State, action, and Motion Track tokens |
| `src/jam/model/world_stream.py` | Video backbone interface and attention processors |
| `src/jam/model/flow.py` | Noise schedules and flow interpolation |
| `src/jam/model/objective.py` | Joint objective with label masks |
| `src/jam/train/` | Prepared-batch training, optimizer, FSDP, and checkpoints |
| `tests/` | Numerical and training-flow checks on small synthetic tensors |

## Install and verify

Use Python 3.11 or newer.

```bash
python -m pip install -e '.[test]'
JAM_BACKBONE_CONFIG="$BACKBONE_DIR/transformer/config.json" python -m pytest -q
```

Use the local video-backbone configuration specified in the manuscript. Tests read its model type, instantiate a small model with synthetic tensors, and need no weights or datasets. `requirements-tested.txt` records the dependency versions used for CPU verification. Distributed CUDA execution is provided separately and has not been exercised by these CPU tests.

## Training interface

The training entry accepts prepared tensor batches in `.safetensors` files. Each file holds one batch. [Tensor shapes and mask semantics](docs/tensors.md) specify this boundary. Video and text encoders run before this interface; their outputs must match the backbone's latent scaling and text embeddings.

Supply a compatible local video checkpoint with a `transformer/` subdirectory. Model loading uses local files only and resolves the installed model class from the checkpoint's original `_class_name` metadata. Set the dimensions in `configs/model.json` to match the prepared tensors. The example includes Motion Track tokens; the internal name `consequence` refers to those tokens throughout the implementation.

```bash
python -m jam.train \
  --backbone "$BACKBONE_DIR" \
  --config configs/model.json \
  --batches "$BATCH_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --output "$OUTPUT_DIR"
```

The caller supplies the training duration and learning rate. The entry uses constant learning rates with separate world and action parameter groups. `--lr-agent` sets the action-stream rate. `--grad-accum`, `--gradient-checkpointing`, and `--bf16` control memory use. Under `torchrun`, prepared batch files are partitioned evenly across ranks and the model uses FSDP. The batch count must be divisible by the process count.

Checkpoints contain model, optimizer, and random-generator state. Add `--resume "$CHECKPOINT_DIR"` to continue from a completed epoch with the same process count and prepared batches. The epoch limit refers to the total number of epochs, including those already completed.

This release contains the core implementation and its prepared-tensor training interface. Dataset preparation, benchmark runners, experimental variants, pretrained weights, and hardware control software are outside its scope. The existing qualitative demo is included in `demo/`.

## License

JAM source is distributed under the [MIT license](LICENSE). Dependencies and separately supplied weights retain their respective licenses.
