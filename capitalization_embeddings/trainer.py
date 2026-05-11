"""Trainer utilities for capitalization-aware MLM pretraining."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer


class CapitalizedMLMTrainer(Trainer):
    """Trainer that logs separate token and capitalization metrics."""

    max_capitalization_eval_batches: int = 64

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        outputs = model(**inputs)
        loss = outputs.loss

        logs = {}
        if getattr(outputs, "token_loss", None) is not None:
            logs["token_loss"] = outputs.token_loss.detach().float().item()
        if getattr(outputs, "capitalization_loss", None) is not None:
            logs["capitalization_loss"] = (
                outputs.capitalization_loss.detach().float().item()
            )
        if logs:
            self.log(logs)

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        cap_metrics = self.evaluate_capitalization_metrics(
            eval_dataset=eval_dataset,
            metric_key_prefix=metric_key_prefix,
        )
        metrics.update(cap_metrics)
        self.log(cap_metrics)
        return metrics

    def evaluate_capitalization_metrics(
        self,
        eval_dataset=None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        totals = torch.zeros(19, dtype=torch.float64, device=self.args.device)
        batches = 0

        for inputs in dataloader:
            if batches >= self.max_capitalization_eval_batches:
                break

            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs)
            totals += self._capitalization_batch_sums(outputs, inputs)
            batches += 1

        if batches == 0 or totals[4].item() == 0:
            return {}

        token_loss_sum, cap_loss_sum, correct, none_correct, first_correct = totals[:5]
        all_correct, total, none_total, first_total, all_total = totals[5:10]
        confusion = totals[10:].reshape(3, 3)

        def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
            if denominator.item() == 0:
                return 0.0
            return float((numerator / denominator).item())

        prefix = metric_key_prefix
        metrics = {
            f"{prefix}_token_loss_unweighted": ratio(token_loss_sum, total),
            f"{prefix}_capitalization_loss_unweighted": ratio(cap_loss_sum, total),
            f"{prefix}_capitalization_accuracy": ratio(correct, total),
            f"{prefix}_capitalization_none_accuracy": ratio(none_correct, none_total),
            f"{prefix}_capitalization_first_cap_accuracy": ratio(first_correct, first_total),
            f"{prefix}_capitalization_all_caps_accuracy": ratio(all_correct, all_total),
            f"{prefix}_capitalization_eval_count": float(total.item()),
            f"{prefix}_capitalization_none_count": float(none_total.item()),
            f"{prefix}_capitalization_first_cap_count": float(first_total.item()),
            f"{prefix}_capitalization_all_caps_count": float(all_total.item()),
            f"{prefix}_capitalization_eval_batches": float(batches),
        }
        labels = ("none", "first_cap", "all_caps")
        for true_index, true_label in enumerate(labels):
            for pred_index, pred_label in enumerate(labels):
                metrics[
                    f"{prefix}_capitalization_confusion_true_{true_label}_pred_{pred_label}"
                ] = float(confusion[true_index, pred_index].item())
        return metrics

    def _capitalization_batch_sums(
        self,
        outputs: Any,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        capitalization_labels = inputs.get("capitalization_labels")
        labels = inputs.get("labels")
        if capitalization_labels is None or outputs.capitalization_logits is None:
            return torch.zeros(19, dtype=torch.float64, device=self.args.device)

        mask = capitalization_labels != -100
        cap_logits = outputs.capitalization_logits
        cap_preds = cap_logits.argmax(dim=-1)

        total = mask.sum().to(torch.float64)
        if total.item() == 0:
            return torch.zeros(19, dtype=torch.float64, device=cap_logits.device)

        cap_loss = torch.tensor(0.0, device=cap_logits.device)
        cap_loss = CrossEntropyLoss(ignore_index=-100, reduction="sum")(
            cap_logits.view(-1, cap_logits.shape[-1]),
            capitalization_labels.view(-1),
        )

        token_loss = torch.tensor(0.0, device=cap_logits.device)
        if labels is not None and outputs.logits is not None:
            token_loss = CrossEntropyLoss(ignore_index=-100, reduction="sum")(
                outputs.logits.view(-1, outputs.logits.shape[-1]),
                labels.view(-1),
            )

        none_mask = mask & (capitalization_labels == 0)
        first_mask = mask & (capitalization_labels == 1)
        all_mask = mask & (capitalization_labels == 2)
        confusion_values = []
        for true_index in range(3):
            true_mask = mask & (capitalization_labels == true_index)
            for pred_index in range(3):
                confusion_values.append(
                    (cap_preds[true_mask] == pred_index).sum().float(),
                )

        values = [
            token_loss.detach().float(),
            cap_loss.detach().float(),
            (cap_preds[mask] == capitalization_labels[mask]).sum().float(),
            (cap_preds[none_mask] == capitalization_labels[none_mask]).sum().float(),
            (cap_preds[first_mask] == capitalization_labels[first_mask]).sum().float(),
            (cap_preds[all_mask] == capitalization_labels[all_mask]).sum().float(),
            total.float(),
            none_mask.sum().float(),
            first_mask.sum().float(),
            all_mask.sum().float(),
            *confusion_values,
        ]
        return torch.stack(values).to(dtype=torch.float64)
