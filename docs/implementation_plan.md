# DeepMagnet Cloud Implementation Plan

## Goal Description
Build "DeepMagnet Cloud", a web-based anti-interference electromagnetic imaging console using **Streamlit** and **Plotly**.
This shift from a desktop app to a web app enables easier server deployment (headless execution), smoother 3D interaction (client-side WebGL by Plotly), and a modern, clean UI.

**Architecture Principle**: Strict Decoupling.
- **Backend (Python)**: Pure logic for physics simulation and model inference. No UI dependencies.
- **Frontend (Streamlit)**: purely for presentation and state management. Calls backend APIs.

## User Review Required
> [!IMPORTANT]
> **New Dependencies**: We are switching stack.
> - `streamlit`: Web framework
> - `plotly`: 3D rendering (Client-side WebGL)
> - `streamlit-drawable-canvas`: Hand-drawn input component
>
> Please allow installation of these packages.

## Proposed Changes

### 1. Project Configuration
#### [MODIFY] [requirements.txt](file:///Users/nianzhen/Desktop/miniproject/requirements.txt)
- Remove `opencv-python` (optional, or keep for utils, but remove from core UI reqs).
- Add `streamlit`, `plotly`, `streamlit-drawable-canvas`, `pandas`.

### 2. Logic Layer (Backend) - *Pure Python*
*Ensure these modules remain free of any UI code.*
#### [Keep/Verify] [src/models/inference.py](file:///Users/nianzhen/Desktop/miniproject/src/models/inference.py)
- `load_model(ckpt, device)` -> returns PyTorch model.
- `run_inference(model, signal, coords)` -> returns pure Tensor volume.

#### [Keep/Verify] [src/data/physics.py](file:///Users/nianzhen/Desktop/miniproject/src/data/physics.py)
- `add_realistic_noise(...)` -> returns pure Tensor signal.

### 3. Visualization Layer (Frontend) - *Streamlit*
#### [DELETE] src/viz/interactive_demo.py (PyVista version)
#### [NEW] [src/viz/web_demo.py](file:///Users/nianzhen/Desktop/miniproject/src/viz/web_demo.py)
- **Design**: Sidebar for controls (Model, Noise), Main area for Input/Output.
- **Components**:
    - `InputSection`: `st_canvas` for drawing 2D digits.
    - `SignalView`: `plotly.graph_objects.Heatmap` for opaque/clean signal visualization.
    - `HologramView`: `plotly.graph_objects.Volume` for semi-transparent 3D rendering.
- **State Management**: Use `st.session_state` to decouple "drawing" from "processing".

### 4. Documentation
#### [MODIFY] [README.md](file:///Users/nianzhen/Desktop/miniproject/README.md)
- Update "Quick Start" to use `streamlit run src/viz/web_demo.py`.
- Update screenshots/description to reflect Web UI.

## Verification Plan

### Automated Tests
- **Backend Unit Tests**: Verify `physics` and `inference` functions still work without UI context.

### Manual Verification
1. **Web Launch**: Run `streamlit run src/viz/web_demo.py`.
2. **Browser Check**: Open `http://localhost:8501`.
3. **Flow Test**:
   - Draw '8' on canvas.
   - Click "Reconstruct".
   - Check if 3D plot appears and is rotatable.
   - Adjust Noise Slider -> Click Reconstruct -> Verify degradation.
