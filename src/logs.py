from typing import Any

import wandb
import torch
import matplotlib.cm as cm
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback


class LogPredictionsCallback(Callback):
    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        batch = next(iter(trainer.val_dataloaders))
        images, labels = batch
        image, label = images[0].to(pl_module.device), labels[0].to(pl_module.device)

        self._log_prediction(trainer, pl_module, image, label, step=trainer.current_epoch)

    def on_test_batch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule",
                            batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        images, labels = batch

        for i, (image, label) in enumerate(zip(images, labels)):
            step = batch_idx * pl_module.cfg.train.batch_size + i
            image, label = image.to(pl_module.device), label.to(pl_module.device)
            self._log_prediction(trainer, pl_module, image, label, step=step)

    @staticmethod
    def _log_prediction(trainer: "L.Trainer", pl_module: "L.LightningModule",
                        image: torch.Tensor, label: torch.Tensor = None, step: int = 0):
        logits = pl_module(image.unsqueeze(0)).squeeze(0)

        # Apply inverse normalization
        mean = torch.tensor(pl_module.cfg.ds.mean).view(1, 3, 1, 1).to(pl_module.device)
        std = torch.tensor(pl_module.cfg.ds.std).view(1, 3, 1, 1).to(pl_module.device)
        image = image * std + mean
        image = image.squeeze(0)

        class_labels = {0: "void", 1: "feasible", 2: "infeasible", 3: "other"}

        # Log prediction mask
        prediction_mask = logits.argmax(0).float().cpu().detach().numpy()
        prediction_image = wandb.Image(
            image.permute(1, 2, 0).cpu().detach().numpy(),
            masks={"prediction": {"mask_data": prediction_mask, "class_labels": class_labels}},
            caption="Prediction",
        )
        trainer.logger.experiment.log({"prediction": prediction_image}, step=step)

        # Calculate entropy
        prob = torch.nn.functional.softmax(logits, dim=0)
        entropy = -torch.sum(prob * torch.log(prob), dim=0)

        # Create RGB from entropy values
        entropy = entropy.cpu().detach().numpy()
        entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min())
        entropy = cm.viridis(entropy)[..., :3]

        # Log entropy image
        entropy_image = wandb.Image(entropy, caption="Entropy")
        trainer.logger.experiment.log({"entropy": entropy_image}, step=step)

        if label is not None:
            # Log ground truth mask
            ground_truth_mask = label.float().cpu().detach().numpy()
            ground_truth_image = wandb.Image(
                image.permute(1, 2, 0).cpu().detach().numpy(),
                masks={"ground_truth": {"mask_data": ground_truth_mask, "class_labels": class_labels}},
                caption="Ground truth",
            )
            trainer.logger.experiment.log({"ground_truth": ground_truth_image}, step=step)

            # Log error mask
            error_mask = torch.abs(label - logits.argmax(0)).cpu().detach().numpy()
            error_image = wandb.Image(
                image.permute(1, 2, 0).cpu().detach().numpy(),
                masks={"error": {"mask_data": error_mask, "class_labels": class_labels}},
                caption="Error",
            )
            trainer.logger.experiment.log({"error": error_image}, step=step)
