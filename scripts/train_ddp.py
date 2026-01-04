#!/usr/bin/env python
"""
Multi-GPU Distributed Data Parallel (DDP) Training for DeepMagnet.
Usage: torchrun --nproc_per_node=4 scripts/train_ddp.py --data configs/data_h100.yaml --model configs/model.yaml --train configs/train_4gpu.yaml
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.data.dataset import VolumeDataset
from src.data.shapes import create_grid
from src.models.baseline import LinearBaseline
from src.models.deeponet import BranchConfig, DeepONet, TrunkConfig
from src.models.encoder_decoder import Decoder3DConfig, SeqToVol
from src.models.losses import LossComputer
from src.models.metrics import compute_metrics
from src.utils.config import ensure_dir, load_yaml
from src.utils.seed import set_seed


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP / torch.compile wrappers to get the underlying model."""
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    if hasattr(model, "_orig_mod"):
        return unwrap_model(model._orig_mod)
    return model


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
    else:
        input_len = signal_shape[0] * signal_shape[1]
        output_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]
        model = LinearBaseline(input_len, output_voxels)
    return model


def train_ddp(args):
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main = local_rank == 0
    
    data_cfg = load_yaml(args.data)
    model_cfg = load_yaml(args.model)
    train_cfg = load_yaml(args.train)

    set_seed(train_cfg.get("seed", 42) + local_rank)  # Different seed per rank
    device = torch.device(f"cuda:{local_rank}")

    # Dataset with DistributedSampler
    shard_size = data_cfg["dataset"].get("shard_size")
    train_ds = VolumeDataset(train_cfg["paths"]["train_shards"], shard_size=shard_size)
    val_ds = VolumeDataset(train_cfg["paths"]["val_shards"], shard_size=shard_size)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False)

    # Effective batch size = batch_size * world_size
    batch_size = train_cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=train_cfg["training"]["num_workers"], pin_memory=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=train_cfg["training"]["num_workers"], pin_memory=True
    )

    grid_shape = tuple(data_cfg["grid"]["size"])
    grid_bounds = tuple(data_cfg["grid"]["bounds"])
    grid_coords = create_grid(grid_shape, grid_bounds, device)
    flat_coords = grid_coords.view(-1, 3)

    signal_shape = train_ds[0]["signals"].shape
    model = build_model(model_cfg, grid_shape, signal_shape).to(device)
    
    compile_enabled = bool(train_cfg.get("compile", False)) and hasattr(torch, "compile")
    if compile_enabled:
        model = torch.compile(model)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["optimizer"]["lr"]), weight_decay=float(train_cfg["optimizer"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("mixed_precision", True))
    loss_fn = LossComputer(mask_weight=model_cfg["loss"]["mask"]["weight"], b_weight=model_cfg["loss"]["bfield"]["weight"])

    best_val = float("inf")
    ckpt_dir = ensure_dir(train_cfg["training"]["ckpt_dir"]) if is_main else None

    for epoch in range(train_cfg["training"]["epochs"]):
        train_sampler.set_epoch(epoch)  # Important for shuffling
        model.train()
        
        pbar = tqdm(train_loader, desc=f"[GPU{local_rank}] Epoch {epoch}", disable=not is_main)
        for batch in pbar:
            signals = batch["signals"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            bfield = batch.get("bfield")
            if bfield is not None:
                bfield = bfield.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            
            with torch.cuda.amp.autocast(enabled=train_cfg.get("mixed_precision", True)):
                # Unwrap DDP module for isinstance check
                base_model = unwrap_model(model)
                if isinstance(base_model, DeepONet):
                    outputs = model(signals, flat_coords)
                    outputs["mask_logits"] = outputs["mask_logits"].view(signals.size(0), *grid_shape)
                    outputs["bfield"] = outputs.get("bfield", torch.zeros(signals.size(0), grid_shape[0] * grid_shape[1] * grid_shape[2], 3, device=device)).view(signals.size(0), grid_shape[0], grid_shape[1], grid_shape[2], 3).permute(0, 4, 1, 2, 3)
                elif isinstance(base_model, SeqToVol):
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
            
            if is_main:
                pbar.set_postfix({"loss": loss.item()})

        # Validation - ALL ranks participate to avoid barrier deadlock
        model.eval()
        val_loss = 0.0
        val_metrics = {"dice": 0.0, "psnr_b": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                signals = batch["signals"].to(device)
                mask = batch["mask"].to(device)
                bfield = batch.get("bfield")
                if bfield is not None:
                    bfield = bfield.to(device)
                
                base_model = unwrap_model(model)
                if isinstance(base_model, DeepONet):
                    outputs = model(signals, flat_coords)
                    outputs["mask_logits"] = outputs["mask_logits"].view(signals.size(0), *grid_shape)
                    if "bfield" in outputs:
                        outputs["bfield"] = outputs["bfield"].view(signals.size(0), *grid_shape, 3).permute(0, 4, 1, 2, 3)
                elif isinstance(base_model, SeqToVol):
                    outputs = model(signals)
                else:
                    outputs = model(signals, grid_shape)
                    
                targets = {"mask": mask}
                if bfield is not None and "bfield" in outputs:
                    targets["bfield"] = bfield.permute(0, 4, 1, 2, 3)
                losses = loss_fn(outputs, targets)
                val_loss += losses["total"].item()
                batch_metrics = compute_metrics(outputs, targets, thresh=train_cfg.get("metrics", {}).get("threshold", 0.5))
                for k, v in batch_metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v.item()
                    
        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # Reduce metrics across all ranks
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_loss_tensor.item()
        
        if is_main:
            print(f"Epoch {epoch} val_loss={val_loss:.4f} metrics={val_metrics}")

            # Checkpoint
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = ckpt_dir / f"best_model.pth"
                torch.save({
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "config": model_cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, ckpt_path)
                print(f"Saved best model to {ckpt_path}")

        dist.barrier()  # Sync all processes

    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data_h100.yaml")
    parser.add_argument("--model", default="configs/model.yaml")
    parser.add_argument("--train", default="configs/train_4gpu.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ddp(args)
