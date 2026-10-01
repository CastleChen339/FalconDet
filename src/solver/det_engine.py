"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ..misc import MetricLogger, SmoothedValue, dist
from ..optim import ModelEMA, Warmup
from .validator import Validator
from ._utils import box_cxcyczwhd_to_xyzxyz, scale_boxes


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    """
    Enable or disable gradients for all parameters in a module.

    Args:
        module: Module to update.
        requires_grad: Flag indicating whether gradients should be tracked.

    Returns:
        None.
    """
    for param in module.parameters():
        param.requires_grad = requires_grad


def _get_eval_canvas_size(target: Dict[str, torch.Tensor], resized_shape: List[int]) -> List[int]:
    """
    Resolve the canvas size for evaluation metrics.

    Args:
        target: Target dictionary for one sample.
        resized_shape: Model input shape used for evaluation fallback.

    Returns:
        Canvas size as [W, H, D].
    """
    for key in ("bboxes", "points"):
        canvas_size = getattr(target.get(key), "canvas_size", None)
        if canvas_size is not None:
            return list(canvas_size)

    # Current dataloader / transform path materializes custom tv_tensors as plain
    # tensors, so fall back to the model-input canvas used by both GT and preds.
    return list(resized_shape)


def _set_module_requires_grad(model: torch.nn.Module, module_name: str, requires_grad: bool) -> None:
    """
    Enable or disable gradients for a named submodule if it exists.

    Args:
        model: Model containing the submodule.
        module_name: Attribute name of the submodule.
        requires_grad: Flag indicating whether gradients should be tracked.

    Returns:
        None.
    """
    module = getattr(model, module_name, None)
    if module is not None:
        set_requires_grad(module, requires_grad)


def _sum_losses(loss_dict, predicate):
    selected = [value for key, value in loss_dict.items() if predicate(key)]
    if selected:
        return sum(selected)
    return next(iter(loss_dict.values())).new_zeros(())


def _loss_diagnostics(loss_dict):
    point = _sum_losses(loss_dict, lambda key: key.startswith("loss_point"))
    density = _sum_losses(loss_dict, lambda key: key.startswith("loss_vox_density"))
    tube_aux = _sum_losses(loss_dict, lambda key: "_aux_" in key)
    tube_enc = _sum_losses(loss_dict, lambda key: "_enc_" in key)
    tube_dn = _sum_losses(loss_dict, lambda key: "_dn_" in key)
    tube_final = _sum_losses(
        loss_dict,
        lambda key: (
            key.startswith("loss_")
            and not key.startswith(("loss_point", "loss_vox_density"))
            and "_aux_" not in key
            and "_enc_" not in key
            and "_dn_" not in key
        ),
    )
    return {
        "diag_tube_final": tube_final,
        "diag_tube_aux": tube_aux,
        "diag_tube_enc": tube_enc,
        "diag_tube_dn": tube_dn,
        "diag_point": point,
        "diag_density": density,
    }


def _branch_gradient_norm(loss, module):
    if module is None or not loss.requires_grad:
        return loss.new_zeros(())
    parameters = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    if not parameters:
        return loss.new_zeros(())
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared_norm = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared_norm += gradient.detach().float().pow(2).sum()
    return squared_norm.sqrt()


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        epochs: int = None,
        print_freq: int = 1,
        ema: ModelEMA = None,
        writer: SummaryWriter = None,
        lr_warmup_scheduler: Warmup = None,
        max_norm: float = 0,
        freeze_backbone_epochs: int = 0,
        freeze_encoder_epochs: int = 0,
        log_branch_grad_norms: bool = False,
        branch_grad_norm_interval: int = 50,
):
    """
    Train the detector for a single epoch.

    Args:
        model: Detection model.
        criterion: Loss module.
        data_loader: Training dataloader.
        optimizer: Optimizer instance.
        device: Device for computation.
        epoch: Current epoch index.
        epochs: Total number of epochs, optional.
        print_freq: Logging frequency.
        ema: EMA wrapper, optional.
        writer: TensorBoard writer, optional.
        lr_warmup_scheduler: Warmup scheduler, optional.
        max_norm: Gradient clipping norm.
        freeze_backbone_epochs: Number of epochs to freeze the backbone.
        freeze_encoder_epochs: Number of epochs to freeze the encoder.
        log_branch_grad_norms: Whether to measure tube and point gradient norms
            entering the encoder.
        branch_grad_norm_interval: Number of training steps between gradient
            norm measurements.

    Returns:
        Dict of averaged training metrics.
    """
    if epoch < freeze_backbone_epochs:
        _set_module_requires_grad(model, "backbone", False)
        if epoch == 0:
            print(f"Freezing backbone for first {freeze_backbone_epochs} epochs")
    elif epoch == freeze_backbone_epochs:
        _set_module_requires_grad(model, "backbone", True)
        print("Unfreezing backbone, now training full model")

    if epoch < freeze_encoder_epochs:
        _set_module_requires_grad(model, "encoder", False)
        if epoch == 0:
            print(f"Freezing encoder for first {freeze_encoder_epochs} epochs")
    elif epoch == freeze_encoder_epochs:
        _set_module_requires_grad(model, "encoder", True)
        print("Unfreezing encoder, now training full model")

    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    losses = []

    for i, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples, targets=targets)
        loss_dict = criterion(outputs, targets, **metas)

        loss: torch.Tensor = sum(loss_dict.values())
        diagnostics = _loss_diagnostics(loss_dict)
        if (
            log_branch_grad_norms
            and global_step % max(int(branch_grad_norm_interval), 1) == 0
        ):
            model_module = model.module if hasattr(model, "module") else model
            encoder = getattr(model_module, "encoder", None)
            tube_objective = (
                diagnostics["diag_tube_final"]
                + diagnostics["diag_tube_aux"]
                + diagnostics["diag_tube_enc"]
                + diagnostics["diag_tube_dn"]
            )
            diagnostics["diag_encoder_grad_tube"] = _branch_gradient_norm(
                tube_objective, encoder
            )
            diagnostics["diag_encoder_grad_point"] = _branch_gradient_norm(
                diagnostics["diag_point"], encoder
            )
        optimizer.zero_grad()
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist.reduce_dict(loss_dict)
        diagnostics_reduced = dist.reduce_dict(diagnostics)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value.item()):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(
            loss=loss_value,
            **loss_dict_reduced,
            **diagnostics_reduced,
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)
            for k, v in diagnostics_reduced.items():
                writer.add_scalar(f"LossDiagnostics/{k}", v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        postprocessor,
        data_loader,
        device,
        print_freq: int = 1,
        max_eval_batches: Optional[int] = None,
):
    """
    Run evaluation for the detector.

    Args:
        model: Detection model.
        criterion: Loss module.
        postprocessor: Postprocessor callable.
        data_loader: Validation dataloader.
        device: Device for computation.
        print_freq: Logging frequency.
        max_eval_batches: Limit evaluation to a number of batches.

    Returns:
        Dict of evaluation metrics.
    """
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    if max_eval_batches is not None and max_eval_batches > 0:
        print(f"Fast eval enabled: limiting validation to first {max_eval_batches} batches")

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if (
            max_eval_batches is not None
            and max_eval_batches > 0
            and i >= max_eval_batches
        ):
            break
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        outputs = model(samples)

        results = postprocessor(outputs)
        resized_shape = [samples.shape[-1], samples.shape[-2], samples.shape[-3]] if samples.dim() == 5 else [
            samples.shape[-1], samples.shape[-2], 1]

        # validator format for metrics
        for target, result in zip(targets, results):
            canvas_size = _get_eval_canvas_size(target, resized_shape)
            gt.append(
                {
                    "bboxes": scale_boxes(
                        boxes=box_cxcyczwhd_to_xyzxyz(target["bboxes"].detach().cpu()),
                        orig_shape=canvas_size,  # [W,H,D]
                        resized_shape=resized_shape,
                    ),
                    "bboxes_labels": target["bboxes_labels"].detach().cpu(),
                    "points": target["points"].detach().cpu(),
                    "points_group_ids": target["points_group_ids"].detach().cpu(),
                    "bboxes_group_ids": target["bboxes_group_ids"].detach().cpu(),
                    "canvas_size": canvas_size,
                }
            )

            preds.append(
                {
                    # tube-level
                    "bboxes": result["bboxes"].detach().cpu(),
                    "bboxes_labels": result["bboxes_labels"].detach().cpu(),
                    "bboxes_scores": result["bboxes_scores"].detach().cpu(),
                    "bboxes_group_ids": result["bboxes_group_ids"].detach().cpu(),

                    # point-level
                    "points": result["points"].detach().cpu(),
                    "points_scores": result["points_scores"].detach().cpu(),
                    "points_labels": result["points_labels"].detach().cpu(),
                    "points_group_ids": result["points_group_ids"].detach().cpu(),
                }
            )

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return metrics
