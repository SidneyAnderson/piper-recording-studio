#!/usr/bin/env python3
"""Patch Piper training code for PyTorch 2.x + pytorch-lightning 1.7 compatibility.

Applies the following patches:
1. torch.load safe_globals for checkpoint loading (PyTorch 2.6+)
2. Manual optimization in VitsModel (PyTorch 2.x multi-optimizer fix)
3. Custom SimpleCheckpoint callback (replaces broken ModelCheckpoint)
4. ONNX export legacy exporter (PyTorch 2.6+ dynamo fix)

Run from the piper/src/python directory:
    python3 ../../train/patch_piper.py
"""

import re
import sys
from pathlib import Path


def patch_file(filepath: Path, patches: list[tuple[str, str]], description: str):
    """Apply text replacements to a file."""
    if not filepath.exists():
        print(f"  SKIP: {filepath} not found")
        return False

    content = filepath.read_text()
    original = content

    for old, new in patches:
        if new in content:
            continue  # Already patched
        if old not in content:
            print(f"  WARN: Pattern not found in {filepath.name}: {old[:60]}...")
            continue
        content = content.replace(old, new, 1)

    if content != original:
        filepath.write_text(content)
        print(f"  PATCHED: {filepath.name} — {description}")
        return True
    else:
        print(f"  OK: {filepath.name} — already patched or no changes needed")
        return False


def main():
    # Determine piper python directory
    piper_dir = Path.cwd()
    if not (piper_dir / "piper_train" / "__main__.py").exists():
        # Try to find it
        for candidate in [
            Path(__file__).parent.parent / "piper" / "src" / "python",
            Path.cwd() / "piper" / "src" / "python",
        ]:
            if (candidate / "piper_train" / "__main__.py").exists():
                piper_dir = candidate
                break
        else:
            print("ERROR: Could not find piper_train. Run from piper/src/python/")
            sys.exit(1)

    print(f"Patching Piper in: {piper_dir}")

    # 1. Patch __main__.py — safe_globals + SimpleCheckpoint
    main_py = piper_dir / "piper_train" / "__main__.py"
    patch_file(main_py, [
        # Add safe_globals for torch.load
        (
            "import torch\n",
            "import torch\nimport pathlib\nif hasattr(torch.serialization, 'add_safe_globals'):\n    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])\n",
        ),
        # Replace ModelCheckpoint with SimpleCheckpoint
        (
            """    trainer = Trainer.from_argparse_args(args)
    if args.checkpoint_epochs is not None:
        trainer.callbacks = [ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)]
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )""",
            """    # Custom checkpoint callback compatible with PyTorch 2.x + PL 1.7
    from pytorch_lightning.callbacks import Callback
    class SimpleCheckpoint(Callback):
        def __init__(self, every_n_epochs=10):
            super().__init__()
            self.every_n_epochs = every_n_epochs

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            if (epoch + 1) % self.every_n_epochs == 0:
                ckpt_dir = Path(trainer.log_dir) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"epoch={epoch}-step={trainer.global_step}.ckpt"
                trainer.save_checkpoint(str(ckpt_path))
                _LOGGER.debug("Saved checkpoint: %s", ckpt_path)

    trainer = Trainer.from_argparse_args(args)
    if args.checkpoint_epochs is not None:
        trainer.callbacks.append(SimpleCheckpoint(every_n_epochs=args.checkpoint_epochs))
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )""",
        ),
    ], "safe_globals + SimpleCheckpoint")

    # 2. Patch lightning.py — manual optimization
    lightning_py = piper_dir / "piper_train" / "vits" / "lightning.py"
    patch_file(lightning_py, [
        # Replace automatic training_step with manual optimization
        (
            """    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            return self.training_step_g(batch)

        if optimizer_idx == 1:
            return self.training_step_d(batch)""",
            """    @property
    def automatic_optimization(self):
        return False

    def training_step(self, batch: Batch, batch_idx: int):
        opts = self.optimizers()
        opt_g, opt_d = opts[0], opts[1]

        # Generator step
        opt_g.zero_grad()
        loss_g = self.training_step_g(batch)
        self.manual_backward(loss_g)
        opt_g.step()

        # Discriminator step
        opt_d.zero_grad()
        loss_d = self.training_step_d(batch)
        self.manual_backward(loss_d)
        opt_d.step()

        # Step LR schedulers
        if hasattr(self, '_sch_g'):
            self._sch_g.step()
        if hasattr(self, '_sch_d'):
            self._sch_d.step()""",
        ),
        # Replace configure_optimizers to store schedulers as attributes
        (
            """    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(
                self.model_g.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            ),
            torch.optim.AdamW(
                self.model_d.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            ),
        ]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[0], gamma=self.hparams.lr_decay
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[1], gamma=self.hparams.lr_decay
            ),
        ]

        return optimizers, schedulers""",
            """    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.model_g.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
        )
        opt_d = torch.optim.AdamW(
            self.model_d.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
        )
        # Store schedulers as attributes — stepped manually in training_step
        self._sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.hparams.lr_decay)
        self._sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.hparams.lr_decay)

        return [opt_g, opt_d]""",
        ),
    ], "manual optimization + scheduler fix")

    # 3. Patch export_onnx.py — safe_globals + legacy exporter
    export_py = piper_dir / "piper_train" / "export_onnx.py"
    patch_file(export_py, [
        (
            "import torch\n",
            "import torch\nimport pathlib\nif hasattr(torch.serialization, 'add_safe_globals'):\n    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])\n",
        ),
        (
            "    torch.onnx.export(\n        model=model_g,",
            "    try:\n        torch.onnx.export(\n            dynamo=False,\n            model=model_g,",
        ),
    ], "safe_globals + legacy ONNX exporter")

    print("\nAll patches applied. Training should now work with PyTorch 2.x.")


if __name__ == "__main__":
    main()
