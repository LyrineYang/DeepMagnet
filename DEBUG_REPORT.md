# DeepMagnet Troubleshooting & Debugging Report

**Last Updated:** 2026-01-03

This document summarizes the debugging process, issues encountered during 4-GPU distributed training setup, and the applied fixes.

## 1. Summary of Fixes

| Issue Category | Error / Symptom | Root Cause | Fix Applied |
| :--- | :--- | :--- | :--- |
| **Logic (Loss)** | `RuntimeError: grad can be implicitly created only for scalar outputs` | `dice_loss` returned a vector `(batch,)` instead of a scalar mean. | Added `.mean()` to `src/models/losses.py`. |
| **Serialization** | `yaml.representer.RepresenterError` | `yaml.safe_dump` cannot serialize nested `dataclass` objects (`MnistConfig`). | Used `dataclasses.asdict` in `src/data/generate.py`. |
| **Compatibility** | `AttributeError: module 'torch.amp' has no attribute ...` | Code used PyTorch 2.3+ `torch.amp` API, but environment is 2.1.0. | Reverted to `torch.cuda.amp` in `scripts/train_ddp.py`. |
| **Deadlock** | Training stuck at `Epoch 0: 0%` | `num_workers: 16` caused CPU resource contention/deadlock with 64 total workers. | Reduced `num_workers` to 4 in `configs/train_4gpu.yaml`. |
| **OOM (Init)** | `exitcode: 1` (Silent crash during init) | `Dataset` loaded all 40GB+ of shards into RAM during `__init__`. | Implemented lazy loading in `src/data/dataset.py`; only check last shard size. |
| **OOM (Runtime)** | `RuntimeError: DefaultCPUAllocator: can't allocate memory` | `torch.load` loaded full shards (1GB) into RAM for every sample access. | Enabled `mmap=True` in `src/utils/io.py` to map files instead of copying. |
| **Invalid Arg** | `ValueError: f must be a string ...` | `torch.load` with `mmap=True` requires string path, not `pathlib.Path`. | Cast path to `str()` in `src/utils/io.py`. |
| **Logic (Data)** | `KeyError: 'center'` in DataLoader | Inconsistent `meta` keys (MNIST vs Shapes) caused `default_collate` to fail. | Removed `meta` from `__getitem__` return value (unused in training). |

## 2. Detailed Verification

### Smoke Test
- **Status:** PASSED
- **Command:** `bash scripts/smoke_test.sh` (Manual equivalent)
- **Verified:** Data generation (tiny), Training loop (CPU/GPU), Validation metrics.

### 4-GPU Distributed Training
- **Status:** READY (Pending final restart)
- **Configuration:** `configs/train_4gpu.yaml`
- **Command:** `torchrun --nproc_per_node=4 scripts/train_ddp.py`
- **Environment:**
  - PyTorch 2.1.0+cu121
  - Mixed Precision (AMP) Enabled
  - DDP Backend: NCCL

## 3. Recommended Git Commit

```bash
git add DEV_STATUS.md DEBUG_REPORT.md configs/train_4gpu.yaml scripts/train_ddp.py src/data/dataset.py src/data/generate.py src/models/losses.py src/utils/io.py
git commit -m "Fix serialization, OOM, PyTorch compat, and data loading issues"
git push
```
