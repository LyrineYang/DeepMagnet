import torch
import yaml
from pathlib import Path
from typing import Tuple, Dict

from .deeponet import DeepONet, BranchConfig, TrunkConfig

def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model(ckpt_path: str | Path, model_config_path: str | Path | None = None, device: str = "cpu") -> DeepONet:
    """
    Load a DeepONet model from a checkpoint.
    
    Args:
        ckpt_path: Path to the model checkpoint (.pt file).
        model_config_path: Path to the model config yaml. If None, tries to find 'configs/model.yaml' relative to project root.
        device: 'cpu', 'cuda', or 'mps'.
        
    Returns:
        Loaded DeepONet model in eval mode.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Try to determine config
    if model_config_path is None:
        root = Path(__file__).parents[2]
        model_config_path = root / "configs" / "model.yaml"
        if not model_config_path.exists():
             print(f"Warning: Config not found at {model_config_path}, using defaults.")
    
    if model_config_path and Path(model_config_path).exists():
        cfg = load_config(model_config_path)
        # Handle both flat config and nested arch.branch config
        branch_dict = cfg.get("arch", {}).get("branch", cfg.get("branch", {}))
        trunk_dict = cfg.get("arch", {}).get("trunk", cfg.get("trunk", {}))
        branch_cfg = BranchConfig(**branch_dict) if branch_dict else BranchConfig()
        trunk_cfg = TrunkConfig(**trunk_dict) if trunk_dict else TrunkConfig()
    else:
        branch_cfg = BranchConfig() 
        trunk_cfg = TrunkConfig()
        
    model = DeepONet(branch_cfg, trunk_cfg)
    
    # Handling state dict key mismatch - train.py saves as "model_state", DDP saves as "model_state_dict"
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint  # Assume it's the raw state dict
        
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "").replace("module.", "")  # Remove torch.compile and DDP prefixes
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def run_inference(model: DeepONet, signal: torch.Tensor, grid_coords: torch.Tensor) -> torch.Tensor:
    """
    Run inference on a single signal to reconstruct the volume.
    
    Args:
        model: Loaded DeepONet model.
        signal: (samples,) or (1, samples)
        grid_coords: (D, H, W, 3) coordinate grid.
        
    Returns:
        (D, H, W) volume intensity.
    """
    device = next(model.parameters()).device
    
    # Preprocess signal
    if signal.dim() == 1:
        signal = signal.unsqueeze(0).unsqueeze(0) # (1, 1, samples) -> (B, steps, samples)
    elif signal.dim() == 2:
        signal = signal.unsqueeze(0) # Assumes input is (steps, samples) -> (1, steps, samples) (1 batch)
    
    signal = signal.to(device)
    
    # Preprocess coords
    # Model expects coords: (P, 3) or (B, P, 3)
    # The trunk net in deeponet.py takes (P, 3) and flattens it.
    spatial_shape = grid_coords.shape[:-1]
    flat_coords = grid_coords.reshape(-1, 3).to(device)
    
    with torch.no_grad():
        # Forward pass
        # Note: DeepONet forward signature: forward(signals, coords)
        # signals: (B, steps, samples)
        # coords: (P, 3)
        # Output: dict with 'mask_logits' -> (B, P)
        
        preds = model(signal, flat_coords)
        mask_logits = preds["mask_logits"] # (1, P)
        
        # Apply sigmoid to get probability/intensity
        vol_flat = torch.sigmoid(mask_logits)
        
        # Reshape to grid
        vol = vol_flat.view(spatial_shape)
        
    return vol
