# DeepMagnet Cloud 使用指南

## 1. 更新摘要
我们已将可视化层从桌面端 PyVista 应用迁移至 **DeepMagnet Cloud**，一个基于 Streamlit 的现代 Web 应用。

### 核心特性
- **服务器友好**: 支持 Headless 模式，可部署在无显示器的远程服务器上。
- **交互式 3D**: 使用 Plotly WebGL 渲染，支持浏览器内丝滑旋转/缩放。
- **手写输入**: 集成 Canvas 画板，支持在浏览器中直接绘制目标。

---

## 2. 前端对后端的依赖 (关键)

> [!IMPORTANT]
> **要让前端正常运行，后端必须满足以下条件：**

### 2.1 模型权重文件
- **路径**: `outputs/checkpoints/best_model.pth`
- **要求**: 必须存在一个有效的 DeepONet 模型权重文件。
- **如何获取**:
    - 运行 `bash scripts/smoke_test.sh` 可生成一个临时模型。
    - 或从完整训练流程获取。

### 2.2 后端 API 函数签名
前端通过以下接口调用后端逻辑，请确保这些函数可用且签名正确：

| 模块 | 函数 | 输入 | 输出 |
| --- | --- | --- | --- |
| `src.models.inference` | `load_model(ckpt_path, device)` | 权重路径, 设备字符串 | `DeepONet` 模型实例 |
| `src.models.inference` | `run_inference(model, signal, grid_coords)` | 模型, 信号Tensor `(steps, samples)`, 坐标Tensor `(D,H,W,3)` | 体素Tensor `(D,H,W)` |
| `src.data.physics` | `add_realistic_noise(signal, noise_level, drift_weight)` | 信号, 噪声级别 (0-1), 漂移权重 | 带噪信号 Tensor |
| `src.data.physics` | `synthesize_signal(cfg, traj, mask, coords, noise_std)` | 线圈配置, 轨迹, 3D Mask, 网格坐标, 噪声标准差 | 信号 Tensor `(steps, samples)` |
| `src.data.generate` | `CoilConfig` | (dataclass 配置) | - |
| `src.data.physics` | `GridConfig` | (dataclass 配置) | - |

> 上述函数已存在于项目中，无需额外开发。确保它们的签名和逻辑未被修改即可。

### 2.3 Python 环境
确保安装了以下依赖：
```bash
pip install -r requirements.txt
```
*(包含 `streamlit`, `plotly`, `streamlit-drawable-canvas`, `opencv-python` 等)*

---

## 3. 启动应用
```bash
streamlit run src/viz/web_demo.py
```

- **本地**: 浏览器将自动打开 `http://localhost:8501`。
- **远程服务器**: 使用 SSH 端口转发：
  ```bash
  ssh -L 8501:localhost:8501 user@your_server_ip
  ```
  然后在本地访问 `http://localhost:8501`。

---

## 4. 验证清单
- [ ] **页面加载**: 是否显示 "DeepMagnet Cloud" 标题和侧边栏？
- [ ] **绘图功能**: 能否在画布上绘制图形？
- [ ] **推理流程**: 点击 "Reconstruct" 后，是否出现加载动画？
- [ ] **3D 可视化**:
    - "Holographic View" 选项卡是否显示 3D 体素？能否旋转？
    - "Sensor Data" 选项卡是否显示热力图？
- [ ] **噪声控制**: 将 "Soil Mineralization" 调至 50%，再次点击 Reconstruct，图像是否变模糊？

---

## 5. 生成/修改的文件
| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `src/viz/web_demo.py` | [NEW] | Streamlit 主应用代码 |
| `src/viz/interactive_demo.py` | [DELETE] | 旧版 PyVista 应用 (已删除) |
| `requirements.txt` | [MODIFY] | 新增 Streamlit 相关依赖 |
| `README.md` | [MODIFY] | 更新启动命令 |
