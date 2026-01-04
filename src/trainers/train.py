import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import VolumeDataset
from ..data.shapes import create_grid
from ..models.baseline import LinearBaseline
from ..models.deeponet import BranchConfig, DeepONet, TrunkConfig
from ..models.encoder_decoder import Decoder3DConfig, SeqToVol
from ..models.grid_segnet import GridSegNet # Added top-level import
from ..models.losses import LossComputer
from ..models.metrics import compute_metrics
from ..utils.config import ensure_dir, load_yaml, merge_dicts
from ..utils.seed import set_seed


def build_model(model_cfg: Dict, grid_shape, signal_shape) -> torch.nn.Module:
    arch = model_cfg["arch"]["name"]
    if arch == "deeponet":
        branch_cfg = BranchConfig(
            hidden=model_cfg["arch"]["branch"]["hidden"],
            layers=model_cfg["arch"]["branch"]["layers"],
            kernel_size=model_cfg["arch"]["branch"]["kernel_size"],
        )
        trunk_cfg = TrunkConfig(
            hidden=model_cfg["arch"]["trunk"]["hidden"],
            layers=model_cfg["arch"]["trunk"]["layers"],
        )
        model = DeepONet(branch_cfg, trunk_cfg, out_channels=4 if model_cfg["arch"]["heads"]["predict_Bfield"] else 1)
        model.build(signal_shape)
    elif arch == "encoder_decoder":
        dec_cfg = Decoder3DConfig(
            base_channels=model_cfg["arch"]["decoder3d"]["base_channels"],
            depth=model_cfg["arch"]["decoder3d"]["depth"],
        )
        model = SeqToVol(latent_dim=model_cfg["arch"]["branch"]["hidden"], decoder_cfg=dec_cfg, grid_shape=grid_shape)
    elif arch == "grid_segnet":
        from ..models.grid_segnet import GridSegNet, Conv2dEncoderConfig, Decoder3dConfig as GridDecConfig
        enc_cfg = Conv2dEncoderConfig(
            base_channels=model_cfg["arch"]["grid_encoder"]["base_channels"],
            depth=model_cfg["arch"]["grid_encoder"]["depth"],
            latent_dim=model_cfg["arch"]["grid_encoder"]["latent_dim"]
        )
        dec_cfg = GridDecConfig(
            base_channels=model_cfg["arch"]["decoder3d"]["base_channels"],
            depth=model_cfg["arch"]["decoder3d"]["depth"],
            output_size=grid_shape
        )
        # grid_size=24 derived from 576 sweep steps (sqrt(576)=24)
        # We assume dataset provides square crops. The encoder will adapt if input matches config.
        # But GridSegNet takes `grid_size` in init for FC calculation. 
        # We should infer it or set it. Standard overfit uses 24.
        # Let's use 24 as default or infer from data config? 
        # build_model receives grid_shape (64,64,64) which is output. Not input grid size.
        # Let's hardcode 24 for now as per successful test, or add to config.
        input_grid_size = model_cfg["arch"].get("input_grid_size", 24)
        model = GridSegNet(enc_cfg, dec_cfg, grid_size=input_grid_size)
    else:
        input_len = signal_shape[0] * signal_shape[1]
        output_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]
        model = LinearBaseline(input_len, output_voxels)
    return model


def train(config_paths):
    data_cfg = load_yaml(config_paths.data)
    model_cfg = load_yaml(config_paths.model)
    train_cfg = load_yaml(config_paths.train)

    set_seed(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = VolumeDataset(train_cfg["paths"]["train_shards"])
    val_ds = VolumeDataset(train_cfg["paths"]["val_shards"])

    train_loader = DataLoader(train_ds, batch_size=train_cfg["training"]["batch_size"], shuffle=True, num_workers=train_cfg["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=train_cfg["training"]["batch_size"], shuffle=False, num_workers=train_cfg["training"]["num_workers"])

    grid_shape = tuple(data_cfg["grid"]["size"])
    grid_bounds = tuple(data_cfg["grid"]["bounds"])
    grid_coords = create_grid(grid_shape, grid_bounds, device)
    flat_coords = grid_coords.view(-1, 3)

    signal_shape = train_ds[0]["signals"].shape
    model = build_model(model_cfg, grid_shape, signal_shape).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["optimizer"]["lr"]), weight_decay=float(train_cfg["optimizer"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("mixed_precision", True))
    loss_fn = LossComputer(
        mask_weight=model_cfg["loss"]["mask"]["weight"], 
        b_weight=model_cfg["loss"]["bfield"]["weight"],
        pos_weight=model_cfg["loss"]["mask"].get("pos_weight", 20.0)  # Added pos_weight from config
    )

    best_val = float("inf")
    ckpt_dir = ensure_dir(train_cfg["training"]["ckpt_dir"])

    for epoch in range(train_cfg["training"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            signals = batch["signals"].to(device)
            mask = batch["mask"].to(device)
            bfield = batch.get("bfield")
            if bfield is not None:
                bfield = bfield.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=train_cfg.get("mixed_precision", True)):
                if isinstance(model, DeepONet):
                    outputs = model(signals, flat_coords)
                    outputs["mask_logits"] = outputs["mask_logits"].view(signals.size(0), *grid_shape)
                    outputs["bfield"] = outputs.get("bfield", torch.zeros(signals.size(0), grid_shape[0] * grid_shape[1] * grid_shape[2], 3, device=device)).view(signals.size(0), grid_shape[0], grid_shape[1], grid_shape[2], 3).permute(0, 4, 1, 2, 3)
                else:
                    # SeqToVol only takes signals; LinearBaseline needs grid_shape
                    if isinstance(model, SeqToVol):
                        outputs = model(signals)
                    elif isinstance(model, GridSegNet):
                        outputs = model(signals)
                    else:
                        outputs = model(signals, grid_shape)
                batch_targets = {"mask": mask}
                if bfield is not None and "bfield" in outputs:
                    # Permute target from (B,D,H,W,3) to (B,3,D,H,W) to match model output
                    batch_targets["bfield"] = bfield.permute(0, 4, 1, 2, 3)
                losses = loss_fn(outputs, batch_targets)
                loss = losses["total"]
            scaler.scale(loss).backward()
            if train_cfg["training"].get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"loss": loss.item()})

        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        with torch.no_grad():
            for batch in val_loader:
                signals = batch["signals"].to(device)
                mask = batch["mask"].to(device)
                bfield = batch.get("bfield")
                if bfield is not None:
                    bfield = bfield.to(device)
                if isinstance(model, DeepONet):
                    outputs = model(signals, flat_coords)
                    outputs["mask_logits"] = outputs["mask_logits"].view(signals.size(0), *grid_shape)
                    if "bfield" in outputs:
                        outputs["bfield"] = outputs["bfield"].view(signals.size(0), *grid_shape, 3).permute(0, 4, 1, 2, 3)
                else:
                    if isinstance(model, SeqToVol):
                        outputs = model(signals)
                    elif isinstance(model, GridSegNet): # Added GridSegNet check
                        outputs = model(signals)
                    else:
                        outputs = model(signals, grid_shape)
                targets = {"mask": mask}
                if bfield is not None and "bfield" in outputs:
                    # Permute target from (B,D,H,W,3) to (B,3,D,H,W) to match model output
                    targets["bfield"] = bfield.permute(0, 4, 1, 2, 3)
                losses = loss_fn(outputs, targets)
                val_loss += losses["total"].item()
                batch_metrics = compute_metrics(outputs, targets, thresh=train_cfg["metrics"]["threshold"] if "metrics" in train_cfg else 0.5)
                for k, v in batch_metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v.item()
        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        print(f"Epoch {epoch} val_loss={val_loss:.4f} metrics={val_metrics}")

        # checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = ckpt_dir / f"model_epoch{epoch}.pt"
            torch.save({"model_state": model.state_dict(), "config": model_cfg}, ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--model", default="configs/model.yaml")
    parser.add_argument("--train", default="configs/train.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
