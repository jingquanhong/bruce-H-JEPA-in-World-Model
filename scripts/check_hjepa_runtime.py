#!/usr/bin/env python3
"""Exercise model construction, hierarchical loss/backward, rollout and serialization."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'h_jepa'))
from hjepa_utils import create_world_model, hjepa_forward
from hierarchical_solver import build_hierarchical_solver


class TrainingHarness(torch.nn.Module):
    def __init__(self, model, losses):
        super().__init__()
        self.model = model
        for name, value in losses.items():
            setattr(self, name, value)

    def log_dict(self, *args, **kwargs):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    results = []
    for config_name in ('ant_hjepa_l3', 'ant_hwm_l3', 'droid_hjepa_l2'):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'h_jepa/config/train')):
            cfg = compose(config_name=config_name)
        if config_name.startswith('droid'):
            cfg.img_size = 64
            cfg.level1.encoder.image_size = 64
        model, losses = create_world_model(cfg)
        model.train()
        n = int(cfg.num_levels)
        length = int(cfg[f'level{n}'].wm.history_size + cfg[f'level{n}'].wm.rollout_n)
        for level in range(n, 1, -1):
            c = cfg[f'level{level}']
            length = (length - 1) * int(c.stride) + int(c.window_size)
            lower = cfg[f'level{level-1}'].wm
            length = max(length, int(lower.history_size + lower.rollout_n))
        batch = {'pixels_level1': torch.randn(2, length, 3, int(cfg.img_size), int(cfg.img_size)),
                 'action_level1': torch.randn(2, length, int(cfg.data.dataset.level1.frameskip * cfg.level1.wm.action_dim))}
        if cfg.level1.wm.get('use_proprio'):
            batch['proprio_level1'] = torch.randn(2, length, int(cfg.level1.wm.proprio_dim))
        harness = TrainingHarness(model, losses)
        output = hjepa_forward(harness, batch, 'train', cfg, normalize_batch=lambda b: None)
        loss = output['loss']
        assert loss.isfinite()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model[0].encoder.parameters())
        assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
        torch.optim.AdamW(model.parameters(), lr=1e-4).step()
        model.eval()
        # Test projection-space action gradients without requiring an environment rollout.
        from hierarchical_solver import LevelCostModel
        adapter = LevelCostModel(model[0], list(model.jepas)[1:])
        dim = int(cfg.data.dataset.level1.frameskip * cfg.level1.wm.action_dim)
        actions = torch.randn(1, 2, 4, dim, requires_grad=True)
        info = {'pixels': torch.zeros(1, 2, 1, 3, int(cfg.img_size), int(cfg.img_size)),
                'embed_0': output['embed_1'][:1, :1].detach().unsqueeze(1).expand(1, 2, 1, -1),
                'goal_embed_0': output['embed_2'][:1, :1].detach().unsqueeze(1).expand(1, 2, 1, -1),
                'goal_embed_0_in_upper_space': True}
        costs = adapter.get_cost(info, actions)
        assert costs.shape == (1, 2) and costs.isfinite().all()
        grad, = torch.autograd.grad(costs.sum(), actions)
        assert grad.isfinite().all() and grad.abs().sum() > 0
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'model.pt'
            torch.save(model, path)
            restored = torch.load(path, weights_only=False)
            assert restored.num_levels == n
        solved_shapes = []
        if config_name == 'ant_hjepa_l3':
            import gymnasium as gym
            import numpy as np
            from eval_config_utils import _build_policy_plan_config
            plan_cfg = OmegaConf.load(ROOT / 'h_jepa/config/eval/ant_l3.yaml')
            for solver_cfg in plan_cfg.solver.solvers.values():
                solver_cfg.n_steps = 2
                solver_cfg.num_samples = 2
                solver_cfg.device = 'cpu'
            model.requires_grad_(False)
            solver = build_hierarchical_solver(plan_cfg, model)
            solver.configure(action_space=gym.spaces.Box(-1., 1., shape=(1, 8), dtype=np.float32),
                             n_envs=1, config=_build_policy_plan_config(plan_cfg))
            obs = {'pixels': batch['pixels_level1'][:1, :1],
                   'goal': batch['pixels_level1'][:1, -1:],
                   'proprio': batch['proprio_level1'][:1, :1],
                   'goal_proprio': batch['proprio_level1'][:1, -1:]}
            for step in (0, 340):
                planned = solver.solve(dict(obs), steps_taken=step, eval_budget=350)
                assert planned['actions'].isfinite().all()
                solved_shapes.append(list(planned['actions'].shape))
        results.append(dict(config=config_name, loss=float(loss.detach()),
                            levels=n, input_frames=length, backward=True,
                            planning_action_gradient=True, serialization=True, solver_shapes=solved_shapes))
        print(results[-1], flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
