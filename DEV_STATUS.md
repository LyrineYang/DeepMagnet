# DeepMagnet 开发状态

> 最后更新: 2026-01-04

## 快速开始

```bash
# 1. 环境 (CUDA 12.x)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "numpy<2"

# 2. 单元测试
python tests/test_core.py

# 3. Smoke Test
bash scripts/smoke_test.sh

# 4. 4卡训练
python scripts/gen_data.py --config configs/data_h100.yaml --device cuda
torchrun --nproc_per_node=4 scripts/train_ddp.py
```

## 关键配置

| 配置 | 用途 |
|------|------|
| `configs/data_h100.yaml` | 10K 样本 |
| `configs/train_4gpu.yaml` | 4卡 DDP |
| `configs/model.yaml` | DeepONet (hidden=128) |

## 已知问题 (已修复)

- ✅ 模块导入 (sys.path)
- ✅ 张量形状 (physics.py, deeponet.py)
- ✅ YAML 类型转换 (lr → float)
- ✅ hidden 维度匹配 (128)
- ✅ Loss backward scalar error (losses.py)
