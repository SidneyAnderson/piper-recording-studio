#!/usr/bin/env python3
"""Patch Piper training code for PyTorch 2.x + pytorch-lightning 1.7 compatibility.

Applies the following patches:
1. torch.load safe_globals for checkpoint loading (PyTorch 2.6+)
2. LR scheduler step override (PyTorch 2.x API change)
3. Custom SimpleCheckpoint callback + disable default ModelCheckpoint
4. ONNX export legacy exporter (PyTorch 2.6+ dynamo fix)

Run from the piper/src/python directory:
    python3 ../../train/patch_piper.py
"""

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
    """Apply all PyTorch 2.x compatibility patches to the Piper training code."""
    # Determine piper python directory
    piper_dir = Path.cwd()
    if not (piper_dir / "piper_train" / "__main__.py").exists():
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

    # 1. Patch __main__.py — safe_globals + SimpleCheckpoint + disable default checkpointing
    main_py = piper_dir / "piper_train" / "__main__.py"
    patch_file(main_py, [
        # Add safe_globals for torch.load
        (
            "import torch\n",
            "import torch\nimport pathlib\nif hasattr(torch.serialization, 'add_safe_globals'):\n    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])\n",
        ),
        # Replace ModelCheckpoint with SimpleCheckpoint and disable default checkpointing
        (
            """    trainer = Trainer.from_argparse_args(args)
    if args.checkpoint_epochs is not None:
        trainer.callbacks = [ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)]
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )""",
            """    # Custom checkpoint callback — PL's default ModelCheckpoint (save_top_k=1)
    # deletes old checkpoints, preventing epoch comparison. SimpleCheckpoint
    # keeps all checkpoints. enable_checkpointing=False disables the default.
    from pytorch_lightning.callbacks import Callback
    class SimpleCheckpoint(Callback):
        def __init__(self, every_n_epochs=10):
            super().__init__()
            self.every_n_epochs = every_n_epochs

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            if (epoch + 1) % self.every_n_epochs == 0:
                try:
                    ckpt_dir = Path(trainer.log_dir) / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"epoch={epoch}-step={trainer.global_step}.ckpt"
                    trainer.save_checkpoint(str(ckpt_path))
                    _LOGGER.info("Saved checkpoint: %s", ckpt_path)
                except Exception as e:
                    _LOGGER.error("Failed to save checkpoint at epoch %d: %s", epoch, e)

    # NOTE: Must set in Namespace directly because from_argparse_args gives
    # Namespace values priority over kwargs, and PL's argparse default is True
    args.enable_checkpointing = False
    trainer = Trainer.from_argparse_args(args, enable_checkpointing=False)
    if args.checkpoint_epochs is not None:
        trainer.callbacks.append(SimpleCheckpoint(every_n_epochs=args.checkpoint_epochs))
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )""",
        ),
    ], "safe_globals + SimpleCheckpoint + disable default checkpointing")

    # 2. Patch lightning.py — LR scheduler step override
    lightning_py = piper_dir / "piper_train" / "vits" / "lightning.py"
    patch_file(lightning_py, [
        # Add lr_scheduler_step after configure_optimizers
        (
            "        return optimizers, schedulers\n\n    @staticmethod",
            "        return optimizers, schedulers\n\n    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):\n        scheduler.step()\n\n    @staticmethod",
        ),
    ], "LR scheduler step override")

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
