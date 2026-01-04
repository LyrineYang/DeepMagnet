## 任务概览
- 目标：基于 DeepONet 的电磁逆向成像/金属探测可视化，快速交付可演示的端到端管线。
- 关注：高层架构与实现计划，时间紧，强调分阶段完成代码。

## 阶段划分与代码计划
- **Phase 0 / 仓库骨架**
  - 创建目录：`configs/`、`src/data/`、`src/models/`、`src/trainers/`、`src/viz/`、`notebooks/`、`scripts/`、`reports/`。
  - 配置草稿：`configs/data.yaml`（网格、线圈参数）、`configs/model.yaml`（结构与损失权重）、`configs/train.yaml`（优化器、batch、epoch、日志）、`configs/paths.yaml`（数据/模型/输出路径）。
  - 工具：`src/utils/io.py`（读写分片与元数据）、`src/utils/config.py`（Hydra/OmegaConf 封装）、`src/utils/seed.py`。

- **Phase 1 / 数据工厂（正向仿真 + 数据集）**
  - `src/data/coil.py`：双D线圈几何、采样轨迹、姿态与波形设置。
  - `src/data/physics.py`：毕奥-萨伐尔 + 互感/阻抗计算；torch/cupy GPU 广播路径。
  - `src/data/shapes.py`：程序化几何（盒/柱/组合），体素化到网格；新增 `MnistShape`（读取 torchvision MNIST，二值化后挤压为 3D 体素），支持混合采样权重。
  - `src/data/generate.py`：组合形状+线圈+物理 → `signals`、`mask`、可选 `Bfield`；噪声注入；分片写出 `.pt/.npy` 与 `metadata.json`。
  - 脚本：`scripts/gen_small.sh`（200–500 样本）与 `scripts/gen_full.sh`（5k+）。
  - `src/data/dataset.py`：PyTorch Dataset/Loader，读取分片，归一化与 collate。

- **Phase 2 / 模型**
  - `src/models/deeponet.py`：Branch（1D CNN/Transformer）编码信号，Trunk（MLP）编码坐标；预测 `mask` 标量与可选 `B` 向量；体素推理坐标批处理辅助。
  - `src/models/encoder_decoder.py`：Seq-to-Vol（信号编码 → 3D 反卷积解码），多头输出 `mask` + `Bfield`。
  - `src/models/baseline.py`：线性/MLP 基线，用于对照实验。
  - `src/models/losses.py`：Dice/BCE，L2/Huber（`B`），可选 SSIM；加权和。
  - `src/models/metrics.py`：Dice、PSNR/SSIM、FPS 计算。

- **Phase 3 / 训练与评估**
  - `src/trainers/train.py`：配置驱动训练循环（混合精度、checkpoint、early stop），日志（CSV/W&B 可选），周期性评估与可视化输出。
  - `src/trainers/eval.py`：离线评估验证/测试分片；保存指标表与样例重建。
  - `src/trainers/ablation.py`：运行线圈类型/噪声/辅助头开关的配置；聚合结果。

- **Phase 4 / 可视化与演示**
  - `src/viz/volume.py`：PyVista 体渲染 + 等值面，颜色/透明度预设；预留 live 模式接口（可在有 GUI 的环境直接 `.show()`）。
  - `src/viz/panel.py`：双视窗（线圈轨迹 + 信号曲线、重建体渲染），“显影”步进模式。
  - `scripts/demo_infer.py`：加载 checkpoint，推理样本，导出帧/视频（ffmpeg）用于展示；可选实时渲染模式（参数开关）。

- **Phase 5 / Notebooks（快速迭代）**
  - `notebooks/01_sanity_forward.ipynb`：线圈场可视化与简单形状校验。
  - `notebooks/02_model_smoke.ipynb`：小数据过拟合检查；绘制 loss/指标。
  - `notebooks/03_viz_story.ipynb`：生成演示图（基线 vs 模型、双D vs 圆线圈）。

- **Phase 6 / 报告资产**
  - `reports/figs/`：对比切片、重建 GIF。
  - `reports/notes.md`：实验记录、超参、失败案例。

## 时间线（可并行压缩）
- D0：骨架 + 配置 + 工具。
- D1：完成 `coil.py`、`physics.py`、`shapes.py`；跑 `gen_small`；实现 `dataset.py`。
- D2：完成 `deeponet.py`、`encoder_decoder.py`、`losses/metrics`；小样本 smoke train；`notebooks/02` 过拟合检查。
- D3：全量数据生成；全量训练；调权重与噪声；启动消融。
- D4：评估 + 消融；完成可视化（`volume.py`、`panel.py`）；`demo_infer.py` 生成演示素材。
- D5：打磨视觉稿；收集图表入 `reports/`；准备讲解脚本。
