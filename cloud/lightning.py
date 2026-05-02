"""
PyTorch Lightning Integration

Production-grade PyTorch Lightning integration for AETHERIS training experiments.
Provides LightningDataModule for activation datasets, LightningModule for probe training,
and Trainer configuration with callbacks, checkpointing, and logging.
"""

import os
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---- Dataset ----

class ActivationDataset(Dataset):
    """Dataset of activation pairs (harmful, harmless) for training."""

    def __init__(self, harmful_acts: torch.Tensor, harmless_acts: torch.Tensor):
        """
        Args:
            harmful_acts: Tensor of shape (N, D) — harmful activations
            harmless_acts: Tensor of shape (N, D) — harmless activations
        """
        self.harmful = harmful_acts.float()
        self.harmless = harmless_acts.float()

    def __len__(self) -> int:
        return self.harmful.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "harmful": self.harmful[idx],
            "harmless": self.harmless[idx],
        }


# ---- DataModule ----

class ActivationDataModule:
    """
    PyTorch Lightning DataModule for activation datasets.

    Handles loading, splitting into train/val/test, and creating DataLoaders
    for the AETHERIS liberation pipeline.
    """

    def __init__(
        self,
        harmful_acts: Dict[int, torch.Tensor],
        harmless_acts: Dict[int, torch.Tensor],
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        """
        Args:
            harmful_acts: Layer-indexed harmful activations
            harmless_acts: Layer-indexed harmless activations
            batch_size: Batch size for DataLoaders
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            num_workers: DataLoader workers
            pin_memory: Pin memory for faster GPU transfer
        """
        self.harmful_acts = harmful_acts
        self.harmless_acts = harmless_acts
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_datasets: Dict[int, ActivationDataset] = {}
        self.val_datasets: Dict[int, ActivationDataset] = {}
        self.test_datasets: Dict[int, ActivationDataset] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        """Split data into train/val/test sets per layer."""
        for layer_idx in self.harmful_acts:
            if layer_idx not in self.harmless_acts:
                continue

            harmful = self.harmful_acts[layer_idx]
            harmless = self.harmless_acts[layer_idx]
            n = min(harmful.shape[0], harmless.shape[0])

            # Shuffle indices
            perm = torch.randperm(n)
            harmful = harmful[perm]
            harmless = harmless[perm]

            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)

            self.train_datasets[layer_idx] = ActivationDataset(
                harmful[:n_train], harmless[:n_train]
            )
            self.val_datasets[layer_idx] = ActivationDataset(
                harmful[n_train:n_train + n_val], harmless[n_train:n_train + n_val]
            )
            self.test_datasets[layer_idx] = ActivationDataset(
                harmful[n_train + n_val:], harmless[n_train + n_val:]
            )

    def train_dataloader(self, layer_idx: int) -> DataLoader:
        """Get train DataLoader for a specific layer."""
        ds = self.train_datasets.get(layer_idx)
        if ds is None:
            raise KeyError(f"No training data for layer {layer_idx}")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self, layer_idx: int) -> DataLoader:
        """Get validation DataLoader for a specific layer."""
        ds = self.val_datasets.get(layer_idx)
        if ds is None:
            raise KeyError(f"No validation data for layer {layer_idx}")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self, layer_idx: int) -> DataLoader:
        """Get test DataLoader for a specific layer."""
        ds = self.test_datasets.get(layer_idx)
        if ds is None:
            raise KeyError(f"No test data for layer {layer_idx}")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def get_layers(self) -> List[int]:
        """Get list of layer indices with data."""
        return sorted(self.train_datasets.keys())


# ---- Probe Module ----

class ProbeModule(nn.Module):
    """
    PyTorch LightningModule for training constraint direction probes.

    Learns a linear probe that distinguishes harmful from harmless activations.
    The learned probe weight serves as the constraint direction.
    """

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_directions: int = 1,
    ):
        """
        Args:
            input_dim: Dimensionality of activation vectors
            learning_rate: Optimizer learning rate
            weight_decay: L2 regularization strength
            n_directions: Number of probe directions to learn
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_directions = n_directions

        # Linear probe: output raw score for binary classification
        self.probe = nn.Linear(input_dim, 1, bias=False)

        # Loss: center harmful at +1, harmless at -1
        self.criterion = nn.MSELoss()

        # Metrics
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: project onto probe direction."""
        return self.probe(x)

    def _step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared step for train/val/test."""
        harmful = batch["harmful"]
        harmless = batch["harmless"]

        # Concatenate and create targets: +1 for harmful, -1 for harmless
        x = torch.cat([harmful, harmless], dim=0)
        targets = torch.cat([
            torch.ones(harmful.shape[0], 1, device=x.device),
            -torch.ones(harmless.shape[0], 1, device=x.device),
        ], dim=0)

        scores = self.probe(x)
        loss = self.criterion(scores, targets)

        # Accuracy: score sign matches target sign
        with torch.no_grad():
            correct = (scores.sign() == targets.sign()).float().mean()

        if stage == "train":
            self.train_losses.append(loss.item())
            self.train_accs.append(correct.item())
        elif stage == "val":
            self.val_losses.append(loss.item())
            self.val_accs.append(correct.item())

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        """Configure Adam optimizer with weight decay."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def get_direction(self) -> torch.Tensor:
        """Get the learned constraint direction (normalized)."""
        direction = self.probe.weight.data.squeeze(0)
        norm = direction.norm()
        if norm > 0:
            direction = direction / norm
        return direction

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics summary."""
        return {
            "train_loss": sum(self.train_losses[-10:]) / max(1, len(self.train_losses[-10:])),
            "val_loss": sum(self.val_losses[-10:]) / max(1, len(self.val_losses[-10:])),
            "train_accuracy": sum(self.train_accs[-10:]) / max(1, len(self.train_accs[-10:])),
            "val_accuracy": sum(self.val_accs[-10:]) / max(1, len(self.val_accs[-10:])),
            "direction_norm": self.get_direction().norm().item(),
        }


# ---- Trainer Factory ----

class LightningTrainer:
    """
    Factory for creating and configuring PyTorch Lightning Trainer instances.

    Provides pre-configured trainers with callbacks for:
    - Model checkpointing
    - Early stopping
    - Learning rate monitoring
    - TensorBoard logging
    """

    def __init__(
        self,
        output_dir: str = "./lightning_checkpoints",
        max_epochs: int = 50,
        accelerator: str = "auto",
        devices: int = 1,
        precision: str = "16-mixed",
        log_every_n_steps: int = 10,
    ):
        """
        Args:
            output_dir: Directory for checkpoints and logs
            max_epochs: Maximum training epochs
            accelerator: "cpu", "gpu", "tpu", or "auto"
            devices: Number of devices to use
            precision: "32", "16-mixed", or "bf16-mixed"
            log_every_n_steps: Logging interval
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def create_trainer(
        self,
        enable_checkpointing: bool = True,
        early_stopping_patience: int = 10,
        enable_progress_bar: bool = True,
    ) -> Any:
        """
        Create a PyTorch Lightning Trainer.

        Args:
            enable_checkpointing: Save best model checkpoints
            early_stopping_patience: Patience for early stopping
            enable_progress_bar: Show progress bar

        Returns:
            Configured Lightning Trainer, or None if lightning is not installed
        """
        try:
            import lightning.pytorch as pl
            from lightning.pytorch.callbacks import (
                ModelCheckpoint,
                EarlyStopping,
                LearningRateMonitor,
            )
            from lightning.pytorch.loggers import TensorBoardLogger

            callbacks = []

            if enable_checkpointing:
                checkpoint_callback = ModelCheckpoint(
                    dirpath=str(self.output_dir / "checkpoints"),
                    filename="probe-{epoch:02d}-{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=3,
                    save_last=True,
                )
                callbacks.append(checkpoint_callback)

            if early_stopping_patience > 0:
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    mode="min",
                    verbose=True,
                )
                callbacks.append(early_stopping)

            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            callbacks.append(lr_monitor)

            logger = TensorBoardLogger(
                save_dir=str(self.output_dir / "logs"),
                name="aetheris_probe",
                version=None,
            )

            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                precision=self.precision,
                callbacks=callbacks,
                logger=logger,
                log_every_n_steps=self.log_every_n_steps,
                enable_progress_bar=enable_progress_bar,
                enable_model_summary=True,
            )

            return trainer

        except ImportError:
            return None

    def train_probe(
        self,
        probe: ProbeModule,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Train a probe using PyTorch Lightning.

        Args:
            probe: ProbeModule to train
            train_loader: Training DataLoader
            val_loader: Validation DataLoader

        Returns:
            Training results with metrics and direction
        """
        trainer = self.create_trainer()
        if trainer is None:
            # Fallback: manual training loop
            return self._manual_train(probe, train_loader, val_loader)

        try:
            import lightning.pytorch as pl

            # Wrap in LightningDataModule adapter
            class _SimpleDataModule(pl.LightningDataModule):
                def __init__(self, train_dl, val_dl):
                    super().__init__()
                    self._train = train_dl
                    self._val = val_dl

                def train_dataloader(self):
                    return self._train

                def val_dataloader(self):
                    return self._val

            dm = _SimpleDataModule(train_loader, val_loader)
            trainer.fit(probe, dm)

            return {
                "success": True,
                "direction": probe.get_direction(),
                "metrics": probe.get_metrics(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _manual_train(
        self,
        probe: ProbeModule,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Manual training loop (fallback when lightning not available)."""
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=probe.learning_rate,
            weight_decay=probe.weight_decay,
        )

        for epoch in range(self.max_epochs):
            probe.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = probe.training_step(batch, 0)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(train_loader))
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs} — Loss: {avg_loss:.4f}")

        return {
            "success": True,
            "direction": probe.get_direction(),
            "metrics": probe.get_metrics(),
            "backend": "manual",
        }


# ---- LightningRuntime ----

class LightningRuntime:
    """
    Lightning.ai / PyTorch Lightning integration for AETHERIS experiments.

    Orchestrates the full pipeline: data preparation, probe training,
    direction extraction, and validation using PyTorch Lightning.
    """

    def __init__(self, output_dir: str = "./lightning_studio"):
        """
        Initialize Lightning runtime.

        Args:
            output_dir: Directory for checkpoints and logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_extraction(
        self,
        model_name: str,
        method: str = "advanced",
        n_directions: int = 4,
        refinement_passes: int = 2,
        batch_size: int = 32,
        max_epochs: int = 50,
        push_to_hub: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full AETHERIS extraction pipeline with Lightning.

        Args:
            model_name: HuggingFace model to liberate
            method: Liberation method
            n_directions: Number of directions per layer
            refinement_passes: Ouroboros compensation passes
            batch_size: Batch size for training
            max_epochs: Maximum training epochs
            push_to_hub: HuggingFace Hub repo (optional)

        Returns:
            Pipeline results
        """
        results = {
            "model": model_name,
            "method": method,
            "n_directions": n_directions,
            "stages": {},
        }

        # Stage 1: Collect activations
        print(f"[1/4] Collecting activations from {model_name}...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from aetheris.core.extractor import ConstraintExtractor
            from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            extractor = ConstraintExtractor(model, tokenizer, device=device)

            harmful_prompts = get_harmful_prompts()[:100]
            harmless_prompts = get_harmless_prompts()[:100]

            harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts)
            harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts)

            results["stages"]["collection"] = {
                "status": "ok",
                "n_layers": len(harmful_acts),
                "n_harmful": len(harmful_prompts),
                "n_harmless": len(harmless_prompts),
            }
        except Exception as e:
            results["stages"]["collection"] = {"status": "error", "error": str(e)}
            return results

        # Stage 2: Setup DataModule and train probes with Lightning
        print("[2/4] Training probes with Lightning...")
        datamodule = ActivationDataModule(
            harmful_acts, harmless_acts,
            batch_size=batch_size,
        )
        datamodule.setup()

        trainer_factory = LightningTrainer(
            output_dir=str(self.output_dir / "checkpoints"),
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
        )

        directions = []
        probe_metrics = {}

        for layer_idx in datamodule.get_layers():
            sample = harmful_acts[layer_idx]
            input_dim = sample.shape[-1]

            probe = ProbeModule(
                input_dim=input_dim,
                n_directions=n_directions,
            )

            train_loader = datamodule.train_dataloader(layer_idx)
            val_loader = datamodule.val_dataloader(layer_idx)

            result = trainer_factory.train_probe(probe, train_loader, val_loader)
            if result.get("success"):
                direction = result["direction"]
                directions.append(direction)
                probe_metrics[layer_idx] = result.get("metrics", {})

        results["stages"]["probe_training"] = {
            "status": "ok",
            "n_layers_trained": len(probe_metrics),
            "n_directions": len(directions),
        }

        if not directions:
            results["stages"]["probe_training"]["status"] = "no_directions"
            return results

        # Stage 3: Remove constraints
        print("[3/4] Removing constraints...")
        try:
            from aetheris.core.projector import NormPreservingProjector

            projector = NormPreservingProjector(model, preserve_norm=True)
            projector.project_weights(directions)
            projector.project_biases(directions)

            # Ouroboros compensation
            if refinement_passes > 1:
                for pass_num in range(refinement_passes - 1):
                    harmful_resid = extractor.collect_activations(
                        model, tokenizer, harmful_prompts[:50]
                    )
                    harmless_resid = extractor.collect_activations(
                        model, tokenizer, harmless_prompts[:50]
                    )
                    residual = []
                    for layer in harmful_resid:
                        if layer in harmless_resid:
                            res = extractor.extract_mean_difference(
                                harmful_resid[layer].to(device),
                                harmless_resid[layer].to(device),
                            )
                            if res.directions:
                                residual.extend(res.directions)
                    if residual:
                        projector.project_weights(residual)
                        projector.project_biases(residual)

            results["stages"]["removal"] = {"status": "ok", "n_directions": len(directions)}
        except Exception as e:
            results["stages"]["removal"] = {"status": "error", "error": str(e)}
            return results

        # Stage 4: Validate
        print("[4/4] Validating...")
        try:
            from aetheris.core.validation import CapabilityValidator

            validator = CapabilityValidator(device)
            test_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a field of artificial intelligence.",
                "The theory of relativity explains the relationship between space and time.",
            ]
            perplexity = validator.compute_perplexity(model, tokenizer, test_texts)
            results["stages"]["validation"] = {"status": "ok", "perplexity": float(perplexity)}
        except Exception as e:
            results["stages"]["validation"] = {"status": "error", "error": str(e)}

        # Save model
        output_model_dir = self.output_dir / "liberated_model"
        output_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_model_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_model_dir))

        results["output_dir"] = str(output_model_dir)
        results["success"] = all(
            s.get("status") == "ok" for s in results["stages"].values()
        )

        return results

    def generate_studio_files(
        self,
        model_name: str,
        method: str = "advanced",
        n_directions: int = 4,
        refinement_passes: int = 2,
        push_to_hub: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate files for Lightning.ai Studio deployment.

        Args:
            model_name: Model to liberate
            method: Liberation method
            n_directions: Number of directions
            refinement_passes: Ouroboros passes
            push_to_hub: HuggingFace Hub repo

        Returns:
            Studio configuration details
        """
        app_content = f'''"""
AETHERIS Lightning Studio
Model: {model_name} | Method: {method}
"""
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.core.validation import CapabilityValidator
from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts
from aetheris.cloud.lightning import LightningRuntime


def main():
    print("=" * 60)
    print(f"AETHERIS Lightning Studio — {{model_name}}")
    print("=" * 60)

    runtime = LightningRuntime(output_dir="./lightning_output")
    result = runtime.run_extraction(
        model_name="{model_name}",
        method="{method}",
        n_directions={n_directions},
        refinement_passes={refinement_passes},
    )

    if result.get("success"):
        print("\\n" + "=" * 60)
        print("Liberation complete!")
        print(f"Model saved to: {{result.get('output_dir', './lightning_output')}}")
        print("=" * 60)
    else:
        print(f"\\nLiberation failed: {{result}}")


if __name__ == "__main__":
    main()
'''
        app_path = self.output_dir / "app.py"
        app_path.write_text(app_content, encoding="utf-8")

        requirements = (
            "aetheris\n"
            "transformers>=4.35.0\n"
            "torch>=2.0.0\n"
            "accelerate>=0.25.0\n"
            "bitsandbytes\n"
            "huggingface_hub\n"
        )
        (self.output_dir / "requirements.txt").write_text(requirements, encoding="utf-8")

        return {
            "success": True,
            "studio_path": str(self.output_dir),
            "files": ["app.py", "requirements.txt"],
            "instructions": [
                "1. Go to https://lightning.ai/",
                "2. Create a new Studio",
                f"3. Upload the contents of {self.output_dir}",
                "4. Select GPU runtime (L4 or A10G recommended)",
                "5. Run: python app.py",
            ],
        }
