"""
Inference module for DeepMagnet models.
Supports: DeepONet, GridSegNet
"""
import torch
import yaml
from pathlib import Path
from typing import Tuple, Dict, Union

from .deeponet import DeepONet, BranchConfig, TrunkConfig
from .grid_segnet import GridSegNet, Conv2dEncoderConfig, Decoder3dConfig


def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str | Path, model_config_path: str | Path | None = None, device: str = "cpu") -> torch.nn.Module:
    """
    Load a model from a checkpoint. Supports DeepONet and GridSegNet.
    
    Args:
        ckpt_path: Path to the model checkpoint (.pt file).
        model_config_path: Path to the model config yaml. If None, tries to find 'configs/model.yaml'.
        device: 'cpu', 'cuda', or 'mps'.
        
    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Try to determine config
    if model_config_path is None:
        root = Path(__file__).parents[2]
        model_config_path = root / "configs" / "model.yaml"
        if not model_config_path.exists():
            print(f"Warning: Config not found at {model_config_path}, using defaults.")
    
    cfg = {}
    if model_config_path and Path(model_config_path).exists():
        cfg = load_config(model_config_path)
    
    # Also check if config was saved in checkpoint
    if "config" in checkpoint and checkpoint["config"]:
        cfg = checkpoint["config"]
    
    # Determine architecture
    arch_name = cfg.get("arch", {}).get("name", "deeponet")
    
    if arch_name == "grid_segnet":
        # Build GridSegNet
        arch_cfg = cfg.get("arch", {})
        enc_cfg = Conv2dEncoderConfig(
            base_channels=arch_cfg.get("grid_encoder", {}).get("base_channels", 64),
            depth=arch_cfg.get("grid_encoder", {}).get("depth", 3),
            latent_dim=arch_cfg.get("grid_encoder", {}).get("latent_dim", 512)
        )
        dec_cfg = Decoder3dConfig(
            base_channels=arch_cfg.get("decoder3d", {}).get("base_channels", 64),
            depth=arch_cfg.get("decoder3d", {}).get("depth", 4),
            output_size=(64, 64, 64)  # Default output size
        )
        input_grid_size = arch_cfg.get("input_grid_size", 24)
        model = GridSegNet(enc_cfg, dec_cfg, grid_size=input_grid_size)
    else:
        # Default to DeepONet
        branch_dict = cfg.get("arch", {}).get("branch", cfg.get("branch", {}))
        trunk_dict = cfg.get("arch", {}).get("trunk", cfg.get("trunk", {}))
        branch_cfg = BranchConfig(**branch_dict) if branch_dict else BranchConfig()
        trunk_cfg = TrunkConfig(**trunk_dict) if trunk_dict else TrunkConfig()
        model = DeepONet(branch_cfg, trunk_cfg)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
        
    # Clean state dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_signal_to_grid(signals: torch.Tensor, traj: torch.Tensor, grid_size: int = 24) -> torch.Tensor:
    """
    Preprocess raw signal + trajectory to 4-channel grid input for GridSegNet.
    
    Args:
        signals: (B, steps, samples) raw signal
        traj: (B, steps, 3) trajectory coordinates
        grid_size: Target grid size (default 24 for 24x24)
        
    Returns:
        (B, 4, grid_size, grid_size) preprocessed input
    """
    B = signals.size(0)
    device = signals.device
    
    # Aggregate signal across samples dimension -> (B, steps)
    signal_agg = signals.mean(dim=2)
    
    # Log + MinMax normalization
    signal_agg = torch.sign(signal_agg) * torch.log1p(torch.abs(signal_agg) * 100)
    sig_min = signal_agg.min(dim=1, keepdim=True)[0]
    sig_max = signal_agg.max(dim=1, keepdim=True)[0]
    signal_norm = (signal_agg - sig_min) / (sig_max - sig_min + 1e-8)
    
    # Reshape to 2D grid
    steps = signal_norm.size(1)
    side = int(steps ** 0.5)
    signal_2d = signal_norm.view(B, side, side)
    
    # Resize if needed
    if side != grid_size:
        import torch.nn.functional as F
        signal_2d = F.interpolate(signal_2d.unsqueeze(1), size=(grid_size, grid_size), mode='bilinear', align_corners=False).squeeze(1)
    
    # Trajectory -> 2D grids
    traj_2d = traj.view(B, side, side, 3)
    if side != grid_size:
        traj_2d = traj_2d.permute(0, 3, 1, 2)
        traj_2d = F.interpolate(traj_2d, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
        traj_2d = traj_2d.permute(0, 2, 3, 1)
    
    # Normalize trajectory to [-1, 1]
    traj_norm = traj_2d / 0.2
    
    # Stack: (B, 4, grid_size, grid_size)
    inputs = torch.cat([
        signal_2d.unsqueeze(1),
        traj_norm.permute(0, 3, 1, 2),
    ], dim=1)
    
    return inputs


def run_inference(
    model: torch.nn.Module, 
    signal: torch.Tensor, 
    traj: torch.Tensor = None,
    grid_coords: torch.Tensor = None,
    grid_size: int = 24
) -> torch.Tensor:
    """
    Run inference on a signal to reconstruct the 3D volume.
    
    Args:
        model: Loaded model (DeepONet or GridSegNet).
        signal: For GridSegNet: (4, H, W) preprocessed OR (steps, samples) raw.
                For DeepONet: (steps, samples) raw signal.
        traj: Trajectory (steps, 3). Required for raw signal with GridSegNet.
        grid_coords: (D, H, W, 3) coordinate grid. Required for DeepONet.
        grid_size: Grid size for GridSegNet preprocessing.
        
    Returns:
        (D, H, W) volume probability/intensity.
    """
    device = next(model.parameters()).device
    
    if isinstance(model, GridSegNet):
        # Check if signal is already preprocessed (4, H, W)
        if signal.dim() == 3 and signal.size(0) == 4:
            # Already preprocessed
            inputs = signal.unsqueeze(0).to(device)  # (1, 4, H, W)
        else:
            # Need preprocessing
            if signal.dim() == 2:
                signal = signal.unsqueeze(0)  # (1, steps, samples)
            if traj is not None and traj.dim() == 2:
                traj = traj.unsqueeze(0)  # (1, steps, 3)
            signal = signal.to(device)
            traj = traj.to(device) if traj is not None else None
            
            if traj is None:
                raise ValueError("GridSegNet requires trajectory for raw signal input")
            inputs = preprocess_signal_to_grid(signal, traj, grid_size)
        
        with torch.no_grad():
            outputs = model(inputs)
            vol = torch.sigmoid(outputs["mask_logits"]).squeeze(0)  # (D, H, W)
    else:
        # DeepONet path
        if signal.dim() == 1:
            signal = signal.unsqueeze(0).unsqueeze(0)
        elif signal.dim() == 2:
            signal = signal.unsqueeze(0)
        
        signal = signal.to(device)
        
        if grid_coords is None:
            raise ValueError("DeepONet requires grid_coords")
            
        spatial_shape = grid_coords.shape[:-1]
        flat_coords = grid_coords.reshape(-1, 3).to(device)
        
        with torch.no_grad():
            preds = model(signal, flat_coords)
            vol_flat = torch.sigmoid(preds["mask_logits"])
            vol = vol_flat.view(spatial_shape)
    
    return vol
