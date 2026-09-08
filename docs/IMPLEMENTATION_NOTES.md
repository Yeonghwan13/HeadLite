# Implementation notes

The model uses ACC/GYR branch widths 32/64/128 and PRS widths 48/96/192, with three valid Conv1D layers of kernel size 30. For the required input length of 300 normalized positions, the convolutional output length is 213. Each branch is pooled and projected to 128 features. A 6-to-64-to-64 metadata encoder produces a 448-feature fused representation.

The mean head has widths 448/256/128/1, with BatchNorm, ReLU and dropout after both hidden layers. Its exact dropout probability is 0.21469034765110603 (0.21 when rounded in the article). The variance head uses the same fully connected widths, BatchNorm and ReLU but no dropout; its final activation is Softplus.

Figure 3 depicts one mean-head dropout block; the implementation has two. The released model keeps both, because they are what the archived checkpoints were trained with; the drawing was not used to change the network. Dropout does not appear in the state dictionary or the parameter count and is inactive in `eval()` mode, so a strict load and a matching evaluation output do not by themselves establish that the training-time behaviour was identical.

The tensor module names and shapes are preserved for state-dictionary compatibility. The public constructor supplies the fixed model configuration and validates tensor inputs. No loss function, training engine, raw preprocessing, dataset partition or trained weights are distributed in this model-only release.

Use `model.eval()` with `torch.inference_mode()` for inference. BatchNorm requires a batch of at least two samples in training mode. A saved state dictionary can be loaded with `torch.load(path, map_location="cpu", weights_only=True)` followed by `model.load_state_dict(state, strict=True)`. This instruction assumes a tensor state dictionary, not an arbitrary serialized model object. The repository provides no such file. `weights_only=True` is the right default but is not a guarantee against every malformed file, and it has itself been the subject of PyTorch security advisories, so load only checkpoints you produced or otherwise trust, on a PyTorch inside the supported range.
