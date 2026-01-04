import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.dataset import VolumeDataset
from ..data.shapes import create_grid
from ..models.deeponet import BranchConfig, DeepONet, TrunkConfig
from ..models.encoder_decoder import Decoder3DConfig, SeqToVol
from ..models.baseline import LinearBaseline
from ..models.metrics import compute_metrics
from ..utils.config import load_yaml
from ..utils.seed import set_seed


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


def evaluate(args):
    data_cfg = load_yaml(args.data)
    model_cfg = load_yaml(args.model)
    set_seed(42)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    ds = VolumeDataset(args.split_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    grid_shape = tuple(data_cfg["grid"]["size"])
    grid_bounds = tuple(data_cfg["grid"]["bounds"])
    grid_coords = create_grid(grid_shape, grid_bounds, device)
    flat_coords = grid_coords.view(-1, 3)

    signal_shape = ds[0]["signals"].shape
    model = build_model(model_cfg, grid_shape, signal_shape).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    metrics_accum = {}
    with torch.no_grad():
        for batch in loader:
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
                else:
                    outputs = model(signals, grid_shape)
            targets = {"mask": mask}
            if bfield is not None and "bfield" in outputs:
                # Permute target from (B,D,H,W,3) to (B,3,D,H,W) to match model output
                targets["bfield"] = bfield.permute(0, 4, 1, 2, 3)
            batch_metrics = compute_metrics(outputs, targets, thresh=args.threshold)
            for k, v in batch_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v.item()
    for k in metrics_accum:
        metrics_accum[k] /= len(loader)
    print("Evaluation metrics:", metrics_accum)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--model", default="configs/model.yaml")
    parser.add_argument("--split_dir", default="data/processed/val")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
