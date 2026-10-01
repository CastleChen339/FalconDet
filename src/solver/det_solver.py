"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import math
import time
from ..misc import dist
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


def _initial_best_value(mode):
    if mode == "max":
        return -math.inf
    if mode == "min":
        return math.inf
    raise ValueError(f"best_mode must be 'max' or 'min', got {mode!r}")


def _is_better_metric(value, best_value, mode):
    if mode == "max":
        return value > best_value
    if mode == "min":
        return value < best_value
    raise ValueError(f"best_mode must be 'max' or 'min', got {mode!r}")


def _dedupe_metrics(metrics):
    deduped = []
    for metric in metrics:
        if metric not in deduped:
            deduped.append(metric)
    return deduped


def _configured_best_metrics(args):
    primary_metric = getattr(args, "best_metric", "point_f1")
    configured = getattr(args, "best_metrics", None)
    if configured is None:
        configured = [primary_metric, "f1", "point_f1_tube_gated"]
    elif isinstance(configured, str):
        configured = [configured]
    return _dedupe_metrics([primary_metric, *configured])


def _metric_mode(args, metric):
    best_mode = getattr(args, "best_mode", "max")
    if isinstance(best_mode, dict):
        return best_mode.get(metric, "max")
    return best_mode


class DetSolver(BaseSolver):
    def fit(self):
        """
        Train the detector with periodic evaluation and checkpointing.

        Returns:
            None.
        """
        self.train()
        args = self.cfg

        print("-" * 42 + "Start training" + "-" * 43)
        best_metric = getattr(args, "best_metric", "point_f1")
        best_metrics = _configured_best_metrics(args)
        best_stats = {}
        best_values = {}
        best_checkpoints = {}
        for metric in best_metrics:
            mode = _metric_mode(args, metric)
            best_value = _initial_best_value(mode)
            best_values[metric] = best_value
            best_stats[metric] = {
                "epoch": -1,
                "metric": metric,
                "mode": mode,
                "value": best_value,
            }
            best_checkpoints[metric] = self.output_dir / f"best_{metric}.pth"

        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            metrics = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.device,
                print_freq=args.print_freq,
                max_eval_batches=args.max_eval_batches,
            )
            for metric in best_metrics:
                if metric not in metrics:
                    raise KeyError(
                        f"Configured best metric {metric!r} is missing from "
                        f"evaluation metrics: {sorted(metrics)}"
                    )
                best_values[metric] = metrics[metric]
                best_stats[metric].update(metrics)
                best_stats[metric]["epoch"] = self.last_epoch
                best_stats[metric]["value"] = metrics[metric]
            print(f"Initial eval from epoch {self.last_epoch}: {metrics}")

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        eval_interval = max(int(getattr(args, "eval_interval", 1)), 1)

        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                freeze_backbone_epochs=args.freeze_backbone_epochs,
                freeze_encoder_epochs=args.freeze_encoder_epochs,
                log_branch_grad_norms=getattr(args, "log_branch_grad_norms", False),
                branch_grad_norm_interval=getattr(
                    args, "branch_grad_norm_interval", 50
                ),
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < args.epochs:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(), checkpoint_path)

            metrics = {}
            should_eval = ((epoch + 1) % eval_interval == 0) or (epoch == args.epochs - 1)
            if should_eval:
                module = self.ema.module if self.ema else self.model
                metrics = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.device,
                    print_freq=args.print_freq,
                    max_eval_batches=args.max_eval_batches,
                )

            # ---- save log to tensorboard ----
            if metrics and self.writer and dist.is_main_process():
                for k, v in metrics.items():
                    self.writer.add_scalar(f"Test/{k}", v, epoch)

            # ---- update best stat ----
            if metrics:
                for metric in best_metrics:
                    if metric not in metrics:
                        raise KeyError(
                            f"Configured best metric {metric!r} is missing from "
                            f"evaluation metrics: {sorted(metrics)}"
                        )
                    metric_value = metrics[metric]
                    mode = best_stats[metric]["mode"]
                    if _is_better_metric(metric_value, best_values[metric], mode):
                        best_values[metric] = metric_value
                        best_stats[metric].update(metrics)
                        best_stats[metric]["epoch"] = epoch
                        best_stats[metric]["value"] = metric_value

                        if self.output_dir:
                            dist.save_on_master(
                                self.state_dict(), best_checkpoints[metric]
                            )

            best_stat_print = best_stats[best_metric].copy()
            print(f"best_stat: {best_stat_print}")
            if len(best_metrics) > 1:
                print(f"best_stats: {best_stats}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            if metrics:
                log_stats.update({f"test_{k}": v for k, v in metrics.items()})

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        """
        Run evaluation on the validation set.

        Returns:
            Dict of evaluation metrics.
        """
        self.eval()
        args = self.cfg

        module = self.ema.module if self.ema else self.model
        metrics = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.device,
            max_eval_batches=args.max_eval_batches,
        )

        # Print evaluation results to console
        print("\nValidation Metrics:")
        for k, v in metrics.items():
            print(f"  {k:<12s}: {v:.4f}" if isinstance(v, (float, int)) else f"  {k:<12s}: {v}")

        # Save evaluation results when output_dir is set
        if self.output_dir and dist.is_main_process():
            eval_path = self.output_dir / "eval_metrics.json"
            with eval_path.open("w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Saved validation metrics to {eval_path}")

        return metrics
