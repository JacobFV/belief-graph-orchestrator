from __future__ import annotations

import argparse
from pathlib import Path

from .config import RuntimeConfig, TrainConfig, load_json_config
from .runtime import Brain
from .serialization import load_session_bundle
from .training.synthetic import generate_mock_sessions
from .training.replay import replay_bundle
from .training.trainer import BrainTrainer
from .training.dataset import VerifierDataset, TargetSelectionDataset


def _make_brain(backend: str, cfg: RuntimeConfig) -> Brain:
    if backend == "mock-desktop":
        from .backends.mock_desktop import MockDesktop
        return Brain(target_key="mock-desktop", target_instance=MockDesktop("mock-desktop"), use_metadata_hints=cfg.use_metadata_hints)
    if backend == "playwright-desktop":
        from .backends.playwright_desktop import PlaywrightDesktop
        return Brain(target_key="pw-desktop", target_instance=PlaywrightDesktop("pw-desktop"), use_metadata_hints=cfg.use_metadata_hints)
    if backend == "playwright-phone":
        from .backends.playwright import PlaywrightPhone
        return Brain(target_key="pw-phone", target_instance=PlaywrightPhone("pw-phone"), use_metadata_hints=cfg.use_metadata_hints)
    # default: mock phone
    from .backends.mock import MockPhone
    return Brain('mock-key', target_cls=MockPhone, use_metadata_hints=cfg.use_metadata_hints)


def cmd_demo(args):
    cfg = load_json_config(args.config, RuntimeConfig) if args.config else RuntimeConfig()
    brain = _make_brain(args.backend, cfg)
    for i in range(args.steps or cfg.demo_steps):
        brain.step()
        if i % 10 == 0:
            print(f"step={i} summary={brain.summary()}")
    if args.out:
        brain.save_bundle(args.out)
    print('final:', brain.summary())


def cmd_record_mock(args):
    out_dir = Path(args.out)
    generate_mock_sessions(out_dir, num_sessions=args.num_sessions, steps=args.steps)
    print(f'wrote sessions to {out_dir}')


def cmd_replay(args):
    bundle = load_session_bundle(args.bundle)
    summary = replay_bundle(bundle, steps=args.steps)
    print(summary)


def cmd_train_verifier(args):
    cfg = load_json_config(args.config, TrainConfig) if args.config else TrainConfig()
    ds = VerifierDataset.from_directory(args.data_dir)
    trainer = BrainTrainer(device=cfg.device, lr=cfg.lr)
    trainer.train_verifier(ds, batch_size=cfg.batch_size, epochs=cfg.epochs)
    trainer.save(args.out)
    print(f'saved checkpoint to {args.out}')


def cmd_train_node_scorer(args):
    cfg = load_json_config(args.config, TrainConfig) if args.config else TrainConfig()
    ds = TargetSelectionDataset.from_directory(args.data_dir)
    trainer = BrainTrainer(device=cfg.device, lr=cfg.lr)
    trainer.train_node_scorer(ds, batch_size=cfg.batch_size, epochs=cfg.epochs)
    trainer.save(args.out)
    print(f'saved checkpoint to {args.out}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='belief-graph-orchestrator')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('demo')
    s.add_argument('--steps', type=int, default=120)
    s.add_argument('--out', type=str, default=None)
    s.add_argument('--config', type=str, default=None)
    s.add_argument('--backend', type=str, default='mock-phone',
                   choices=['mock-phone', 'mock-desktop', 'playwright-phone', 'playwright-desktop'],
                   help='Backend to use for the demo')
    s.set_defaults(func=cmd_demo)
    s = sub.add_parser('record-mock')
    s.add_argument('--out', type=str, required=True)
    s.add_argument('--num-sessions', type=int, default=4)
    s.add_argument('--steps', type=int, default=120)
    s.set_defaults(func=cmd_record_mock)
    s = sub.add_parser('replay')
    s.add_argument('--bundle', type=str, required=True)
    s.add_argument('--steps', type=int, default=120)
    s.set_defaults(func=cmd_replay)
    s = sub.add_parser('train-verifier')
    s.add_argument('--data-dir', type=str, required=True)
    s.add_argument('--out', type=str, required=True)
    s.add_argument('--config', type=str, default=None)
    s.set_defaults(func=cmd_train_verifier)
    s = sub.add_parser('train-node-scorer')
    s.add_argument('--data-dir', type=str, required=True)
    s.add_argument('--out', type=str, required=True)
    s.add_argument('--config', type=str, default=None)
    s.set_defaults(func=cmd_train_node_scorer)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
