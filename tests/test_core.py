#!/usr/bin/env python
"""
Unit tests for DeepMagnet - run these BEFORE pushing to catch errors early.
Usage: python -m pytest tests/test_core.py -v
Or:    python tests/test_core.py
"""
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import unittest


class TestPhysics(unittest.TestCase):
    """Test physics module tensor shapes."""
    
    def test_coil_moments_shape(self):
        from src.data.coil import CoilConfig, coil_moments
        cfg = CoilConfig()
        result = coil_moments(cfg, device=torch.device("cpu"))
        # Should be (n_coils, 6) where 6 = 3 (center) + 3 (moment)
        self.assertEqual(result.shape[1], 6)
        self.assertIn(result.shape[0], [1, 2])  # circle=1, double_d=2
    
    def test_sample_trajectory_shape(self):
        from src.data.coil import CoilConfig, sample_trajectory
        cfg = CoilConfig(sweep_steps=16)
        traj = sample_trajectory(cfg, device=torch.device("cpu"))
        self.assertEqual(traj.shape, (16, 3))
    
    def test_synthesize_signal_shape(self):
        from src.data.coil import CoilConfig, sample_trajectory
        from src.data.physics import synthesize_signal
        from src.data.shapes import create_grid
        
        cfg = CoilConfig(sweep_steps=8, samples=32)
        device = torch.device("cpu")
        traj = sample_trajectory(cfg, device)
        grid = create_grid((8, 8, 8), (-0.1, 0.1), device)
        mask = torch.zeros(8, 8, 8, device=device)
        mask[3:5, 3:5, 3:5] = 1.0
        
        signal = synthesize_signal(cfg, traj, mask, grid)
        self.assertEqual(signal.shape, (8, 32))  # (steps, samples)
    
    def test_coil_field_at_points_1d_input(self):
        """Test that 1D centroid input is handled correctly."""
        from src.data.coil import CoilConfig, sample_trajectory
        from src.data.physics import coil_field_at_points
        
        cfg = CoilConfig(sweep_steps=8)
        device = torch.device("cpu")
        traj = sample_trajectory(cfg, device)
        centroid = torch.tensor([0.0, 0.0, 0.05], device=device)  # 1D input
        
        b = coil_field_at_points(cfg, traj, centroid)
        self.assertEqual(b.shape, (8, 3))


class TestModels(unittest.TestCase):
    """Test model construction and forward pass."""
    
    def test_deeponet_hidden_dims_match(self):
        """Verify branch and trunk hidden dims match."""
        from src.utils.config import load_yaml
        cfg = load_yaml(PROJECT_ROOT / "configs" / "model.yaml")
        branch_hidden = cfg["arch"]["branch"]["hidden"]
        trunk_hidden = cfg["arch"]["trunk"]["hidden"]
        self.assertEqual(branch_hidden, trunk_hidden, 
                        f"Branch hidden ({branch_hidden}) != Trunk hidden ({trunk_hidden})")
    
    def test_deeponet_forward(self):
        from src.models.deeponet import DeepONet, BranchConfig, TrunkConfig
        
        hidden = 64
        branch_cfg = BranchConfig(hidden=hidden, layers=2)
        trunk_cfg = TrunkConfig(hidden=hidden, layers=2)
        model = DeepONet(branch_cfg, trunk_cfg)
        model.build(torch.Size([8, 32]))  # (steps, samples)
        
        signals = torch.randn(2, 8, 32)  # (batch, steps, samples)
        coords = torch.randn(100, 3)  # (num_points, 3)
        
        outputs = model(signals, coords)
        self.assertIn("mask_logits", outputs)
        self.assertEqual(outputs["mask_logits"].shape, (2, 100))
    
    def test_seqtovol_forward(self):
        from src.models.encoder_decoder import SeqToVol, Decoder3DConfig
        
        dec_cfg = Decoder3DConfig(base_channels=16, depth=3)
        model = SeqToVol(latent_dim=64, decoder_cfg=dec_cfg, grid_shape=(16, 16, 16))
        
        signals = torch.randn(2, 8, 32)
        outputs = model(signals)
        
        self.assertIn("mask_logits", outputs)
        self.assertEqual(outputs["mask_logits"].shape, (2, 16, 16, 16))


class TestDataGeneration(unittest.TestCase):
    """Test data generation pipeline."""
    
    def test_generate_sample(self):
        from src.data.generate import DataGenerator
        from src.data.physics import GridConfig
        from src.data.coil import CoilConfig
        from src.data.shapes import ShapeConfig
        
        grid_cfg = GridConfig(size=(8, 8, 8), bounds=(-0.1, 0.1), voxel_size=0.025)
        coil_cfg = CoilConfig(sweep_steps=4, samples=16)
        shape_cfg = ShapeConfig(types=["box"], size_range=[0.02, 0.05], min_distance_from_coil=0.01)
        
        gen = DataGenerator(grid_cfg, coil_cfg, shape_cfg, device="cpu")
        sample = gen.generate_sample()
        
        self.assertIn("signals", sample)
        self.assertIn("mask", sample)
        self.assertEqual(sample["signals"].shape, (4, 16))
        self.assertEqual(sample["mask"].shape, (8, 8, 8))


if __name__ == "__main__":
    unittest.main()
