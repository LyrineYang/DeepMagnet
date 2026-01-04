#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.dataset import VolumeDataset
from src.data.shapes import create_grid
from src.models.deeponet import BranchConfig, DeepONet, TrunkConfig
from src.models.encoder_decoder import Decoder3DConfig, SeqToVol
from src.models.baseline import LinearBaseline
from src.utils.config import load_yaml
from src.viz.panel import render_panel


def build_model(model_cfg, grid_shape, signal_shape):
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
    else:
        input_len = signal_shape[0] * signal_shape[1]
        output_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]
        model = LinearBaseline(input_len, output_voxels)
    return model


def main(args):
    data_cfg = load_yaml(args.data)
    model_cfg = load_yaml(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = VolumeDataset(args.sample_dir, load_bfield=True)
    sample = ds[0]
    grid_shape = tuple(data_cfg["grid"]["size"])
    grid_bounds = tuple(data_cfg["grid"]["bounds"])
    grid_coords = create_grid(grid_shape, grid_bounds, device)
    flat_coords = grid_coords.view(-1, 3)

    signal_shape = sample["signals"].shape
    model = build_model(model_cfg, grid_shape, signal_shape).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        signals = sample["signals"].unsqueeze(0).to(device)
        if isinstance(model, DeepONet):
            outputs = model(signals, flat_coords)
            mask = torch.sigmoid(outputs["mask_logits"]).view(1, *grid_shape)
            mask_np = mask.squeeze(0).cpu().numpy()
            bfield = outputs.get("bfield")
            b_np = None
            if bfield is not None:
                bfield = bfield.view(1, *grid_shape, 3).permute(0, 4, 1, 2, 3)
                b_np = bfield.squeeze(0).cpu().numpy()
        else:
            outputs = model(signals, grid_shape)
            mask_np = torch.sigmoid(outputs["mask_logits"]).squeeze(0).cpu().numpy()
            b_np = outputs["bfield"].squeeze(0).cpu().numpy() if "bfield" in outputs else None
    render_panel(sample["signals"].numpy(), mask_np, b_np, sample.get("traj"), out_dir=args.out_dir, live=args.live)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--model", default="configs/model.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sample_dir", default="data/processed/val")
    parser.add_argument("--out_dir", default="outputs/demo")
    parser.add_argument("--live", action="store_true", help="开启交互式渲染（需要 GUI 支持）")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
