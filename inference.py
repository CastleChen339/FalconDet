"""Batch inference, evaluation, and visualization for FalconDet checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.core import YAMLConfig
from src.solver.point_metrics import match_points, normalized_time_to_frame
from src.zoo.FalconDet.utils import box_cxcyczwhd_to_xyzxyz, relative_points_to_global


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIS_COLORS = {
    "tp": (0, 220, 0),
    "fp": (255, 40, 40),
    "fn": (255, 220, 0),
    "det": (0, 220, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FalconDet inference and save visualizations and predictions."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/custom/FalconDet_eval.yml"),
        help="FalconDet YAML configuration.",
    )
    parser.add_argument(
        "--weights",
        "-r",
        type=Path,
        default=Path("checkpoints/FalconDet_v2_1_best_box_best_stg1_epoch17.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default=Path("test_samples/testset_1"),
        help="A sequence directory, or a directory containing sequence directories.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/inference"),
        help="Output directory.",
    )
    parser.add_argument("--device", default="cuda", help="Inference device, such as cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=1, help="Sequence windows per batch.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length.")
    parser.add_argument("--stride", type=int, default=None, help="Window stride; defaults to seq-len.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square model input size; defaults to config eval_spatial_size.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Use original image size as model input.",
    )
    parser.add_argument("--conf", type=float, default=0.40, help="Tube confidence threshold.")
    parser.add_argument(
        "--point-conf",
        type=float,
        default=0.40,
        help="Point confidence threshold after tube-score blending.",
    )
    parser.add_argument(
        "--point-match",
        type=float,
        default=12.8,
        help="GT point matching distance in original-image pixels.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=30,
        help="Maximum tube detections retained per sequence.",
    )
    parser.add_argument("--radius", type=int, default=6, help="Visualization circle radius.")
    parser.add_argument("--line-width", type=int, default=2, help="Visualization line width.")
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer EMA weights when available.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")
    return checkpoint


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith("total_ops") or key.endswith("total_params"):
            continue
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def load_model(
    config_path: Path,
    checkpoint: Dict[str, Any],
    device: torch.device,
    use_ema: bool,
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any]]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = YAMLConfig(str(config_path))
    # The checkpoint replaces every parameter, so loading backbone pretraining is unnecessary.
    cfg.yaml_cfg.setdefault("Backbone3D", {})["pretrained"] = False
    model = cfg.model
    postprocessor = cfg.postprocessor

    state_dict = None
    weight_source = "model"
    if use_ema and isinstance(checkpoint.get("ema"), dict):
        state_dict = checkpoint["ema"].get("module")
        if isinstance(state_dict, dict):
            weight_source = "ema"
    if not isinstance(state_dict, dict):
        state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint contains neither model nor EMA weights.")

    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.to(device).eval()
    postprocessor.eval()
    return model, postprocessor, {
        "config": cfg.yaml_cfg,
        "weight_source": weight_source,
        "last_epoch": checkpoint.get("last_epoch"),
    }


def discover_sequences(source: Path) -> List[Tuple[str, Path, Optional[Path], List[Path]]]:
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not source.is_dir():
        raise ValueError(f"Source must be a directory: {source}")

    direct_images = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )

    candidates: List[Tuple[str, Path, Path, Optional[Path]]] = []
    if direct_images:
        # Passing an image directory explicitly always means inference without GT.
        candidates.append((source.name, source, source, None))
    elif (source / "images").is_dir():
        json_dir = source / "json"
        candidates.append(
            (
                source.name,
                source,
                source / "images",
                json_dir if json_dir.is_dir() else None,
            )
        )
    else:
        for sequence_dir in sorted(path for path in source.iterdir() if path.is_dir()):
            image_dir = sequence_dir / "images"
            if not image_dir.is_dir():
                continue
            json_dir = sequence_dir / "json"
            candidates.append(
                (
                    sequence_dir.name,
                    sequence_dir,
                    image_dir,
                    json_dir if json_dir.is_dir() else None,
                )
            )

    sequences = []
    for sequence_name, sequence_dir, image_dir, annotation_dir in candidates:
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        if not image_paths:
            continue
        if annotation_dir is not None:
            missing = [
                path.name
                for path in image_paths
                if not (annotation_dir / f"{path.stem}.json").is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{sequence_name} has images without JSON annotations: "
                    + ", ".join(missing[:5])
                )
        sequences.append((sequence_name, sequence_dir, annotation_dir, image_paths))

    if not sequences:
        raise ValueError(f"No image sequences found under {source}")
    return sequences


def make_windows(num_images: int, seq_len: int, stride: int) -> List[Tuple[int, int]]:
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq-len and stride must be positive.")
    if num_images < seq_len:
        return []
    return [
        (start, start + seq_len)
        for start in range(0, num_images - seq_len + 1, stride)
    ]


def read_frame(path: Path, image_size: int, resize: bool) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if resize:
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float() / 255.0


def load_sequence(
    image_paths: Sequence[Path],
    image_size: int,
    resize: bool,
) -> torch.Tensor:
    frames = [read_frame(path, image_size=image_size, resize=resize) for path in image_paths]
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        raise ValueError("All frames in one sequence window must have the same size.")
    return torch.stack(frames, dim=1)


def load_canvases(image_paths: Iterable[Path]) -> Dict[str, Image.Image]:
    canvases = {}
    for path in image_paths:
        with Image.open(path) as image:
            canvases[path.name] = image.convert("RGB")
    return canvases


def postprocess_outputs(
    outputs: Dict[str, Any],
    postprocessor: torch.nn.Module,
    conf_threshold: float,
    point_conf_threshold: float,
    max_detections: int,
) -> List[Dict[str, torch.Tensor]]:
    logits = outputs["pred_logits"].detach().cpu()
    boxes = box_cxcyczwhd_to_xyzxyz(outputs["pred_boxes"].detach().cpu()).clamp(0, 1)
    points = outputs.get("pred_points")
    point_scores = outputs.get("pred_point_scores")
    num_classes = int(postprocessor.num_classes)

    if postprocessor.use_focal_loss:
        all_scores = logits.sigmoid()
        top_k = min(int(postprocessor.num_top_queries), all_scores.shape[1] * num_classes)
        scores, flat_indices = torch.topk(all_scores.flatten(1), top_k, dim=-1)
        labels = flat_indices % num_classes
        query_indices = flat_indices // num_classes
    else:
        all_scores = torch.softmax(logits, dim=-1)[..., :-1]
        scores, labels = all_scores.max(dim=-1)
        top_k = min(int(postprocessor.num_top_queries), scores.shape[1])
        scores, query_indices = torch.topk(scores, top_k, dim=-1)
        labels = labels.gather(1, query_indices)

    results = []
    for batch_idx in range(boxes.shape[0]):
        keep = scores[batch_idx] >= conf_threshold
        selected_scores = scores[batch_idx][keep][:max_detections]
        selected_labels = labels[batch_idx][keep][:max_detections]
        selected_queries = query_indices[batch_idx][keep][:max_detections]
        selected_boxes = boxes[batch_idx][selected_queries]

        out_points = []
        out_point_scores = []
        out_point_group_ids = []
        if points is not None and point_scores is not None and points[batch_idx] is not None:
            sample_points = points[batch_idx].detach().cpu()
            sample_point_scores = point_scores[batch_idx].detach().cpu()
            for group_id, (query_idx, tube_box, tube_score) in enumerate(
                zip(selected_queries, selected_boxes, selected_scores)
            ):
                query_idx_int = int(query_idx)
                global_points = relative_points_to_global(
                    sample_points[query_idx_int],
                    tube_box,
                ).clamp(0, 1)
                raw_scores = sample_point_scores[query_idx_int].clamp_min(1e-6)
                tube_scale = (
                    (1.0 - float(postprocessor.point_tube_score_blend))
                    + float(postprocessor.point_tube_score_blend)
                    * tube_score.clamp_min(1e-6).pow(float(postprocessor.tube_score_power))
                )
                blended_scores = (
                    raw_scores.pow(float(postprocessor.point_score_power)) * tube_scale
                )
                point_keep = blended_scores >= point_conf_threshold
                out_points.append(global_points[point_keep])
                out_point_scores.append(blended_scores[point_keep])
                out_point_group_ids.append(
                    torch.full((int(point_keep.sum()),), group_id, dtype=torch.long)
                )

        results.append(
            {
                "bboxes": selected_boxes,
                "bboxes_labels": selected_labels,
                "bboxes_scores": selected_scores,
                "points": torch.cat(out_points) if out_points else torch.empty((0, 3)),
                "points_scores": (
                    torch.cat(out_point_scores) if out_point_scores else torch.empty((0,))
                ),
                "points_group_ids": (
                    torch.cat(out_point_group_ids)
                    if out_point_group_ids
                    else torch.empty((0,), dtype=torch.long)
                ),
            }
        )
    return results


def prediction_points_to_records(
    result: Dict[str, torch.Tensor],
    width: int,
    height: int,
    seq_len: int,
) -> List[Dict[str, Any]]:
    records = []
    for point, score, group_id in zip(
        result["points"], result["points_scores"], result["points_group_ids"]
    ):
        frame = normalized_time_to_frame(float(point[2]), seq_len)
        records.append(
            {
                "frame": frame,
                "x": float(point[0]) * width,
                "y": float(point[1]) * height,
                "t": float(point[2]),
                "score": float(score),
                "group_id": int(group_id),
            }
        )
    return records


def load_gt_points(json_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records = []
    for frame_idx, path in enumerate(json_paths):
        with path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)
        for shape in annotation.get("shapes", []):
            points = shape.get("points") or []
            if shape.get("label") != "debris" or not points:
                continue
            records.append(
                {
                    "frame": frame_idx,
                    "x": float(points[0][0]),
                    "y": float(points[0][1]),
                    "group_id": shape.get("group_id"),
                }
            )
    return records


def draw_circle(
    canvas: Image.Image,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    radius: int,
    line_width: int,
) -> None:
    x = max(0.0, min(float(canvas.width - 1), x))
    y = max(0.0, min(float(canvas.height - 1), y))
    ImageDraw.Draw(canvas).ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=color,
        width=line_width,
    )


def draw_prediction_label(
    canvas: Image.Image,
    x: float,
    y: float,
    tube_score: float,
    point_score: float,
    color: Tuple[int, int, int],
    radius: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    text = f"score: {tube_score:.2f}\nvisibility: {point_score:.2f}"
    bbox = draw.multiline_textbbox((0, 0), text, spacing=1, stroke_width=1)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x0 = max(2, min(canvas.width - text_width - 2, int(x - text_width / 2)))
    y0 = max(2, min(canvas.height - text_height - 2, int(y - radius - text_height - 4)))
    draw.rectangle((x0 - 2, y0 - 2, x0 + text_width + 2, y0 + text_height + 2), fill=(0, 0, 0))
    draw.multiline_text(
        (x0, y0),
        text,
        fill=color,
        spacing=1,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )


def tensor_list(tensor: torch.Tensor) -> List[Any]:
    return tensor.detach().cpu().tolist()


def run_inference(args: argparse.Namespace) -> Dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.max_detections <= 0:
        raise ValueError("max-detections must be positive.")

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.weights)
    model, postprocessor, model_info = load_model(
        args.config, checkpoint, device=device, use_ema=args.use_ema
    )
    config = model_info["config"]
    eval_spatial_size = config["eval_spatial_size"]
    configured_seq_len = int(eval_spatial_size[0])
    configured_height = int(eval_spatial_size[1])
    configured_width = int(eval_spatial_size[2])
    seq_len = int(args.seq_len) if args.seq_len is not None else configured_seq_len
    if seq_len != configured_seq_len:
        raise ValueError(
            f"This model was built for {configured_seq_len} frames; got seq-len={seq_len}."
        )
    if configured_height != configured_width:
        raise ValueError(
            "inference.py currently expects square model input, but "
            f"config eval_spatial_size is {eval_spatial_size}."
        )
    image_size = configured_height if args.image_size is None else int(args.image_size)
    if image_size <= 0:
        raise ValueError("image-size must be positive.")
    if not args.no_resize and image_size != configured_height:
        raise ValueError(
            "image-size must match config eval_spatial_size for this model: "
            f"got {image_size}, expected {configured_height}."
        )
    stride = int(args.stride) if args.stride is not None else seq_len
    resize = not args.no_resize
    sequences = discover_sequences(args.source)
    if not resize:
        for sequence_name, _, _, image_paths in sequences:
            with Image.open(image_paths[0]) as image:
                if image.size != (configured_width, configured_height):
                    raise ValueError(
                        "--no-resize requires source images to match config "
                        "eval_spatial_size: "
                        f"{sequence_name} has {image.size}, expected "
                        f"{(configured_width, configured_height)}."
                    )

    jobs = []
    sequence_canvases = {}
    for name, sequence_dir, json_dir, image_paths in sequences:
        windows = make_windows(len(image_paths), seq_len, stride)
        if not windows:
            print(f"Skipping {name}: found {len(image_paths)} images, need {seq_len}.")
            continue
        sequence_canvases[name] = load_canvases(image_paths)
        for start, end in windows:
            jobs.append(
                {
                    "sequence": name,
                    "sequence_dir": sequence_dir,
                    "json_dir": json_dir,
                    "image_paths": image_paths,
                    "start": start,
                    "end": end,
                }
            )
    if not jobs:
        raise ValueError("No valid sequence windows found.")

    args.output.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "weights": str(args.weights),
        "weight_source": model_info["weight_source"],
        "checkpoint_epoch": model_info["last_epoch"],
        "config": str(args.config),
        "source": str(args.source),
        "output": str(args.output),
        "device": str(device),
        "seq_len": seq_len,
        "stride": stride,
        "image_size": None if args.no_resize else image_size,
        "conf_threshold": args.conf,
        "point_conf_threshold": args.point_conf,
        "point_match_threshold_px": args.point_match,
        "windows": [],
        "totals": {
            "detections": 0,
            "detected_points": 0,
            "gt_points": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        },
    }

    with torch.inference_mode():
        for batch_start in range(0, len(jobs), args.batch_size):
            batch_jobs = jobs[batch_start : batch_start + args.batch_size]
            batch = torch.stack(
                [
                    load_sequence(
                        job["image_paths"][job["start"] : job["end"]],
                        image_size=image_size,
                        resize=resize,
                    )
                    for job in batch_jobs
                ]
            ).to(device, non_blocking=True)
            outputs = model(batch)
            results = postprocess_outputs(
                outputs,
                postprocessor,
                conf_threshold=args.conf,
                point_conf_threshold=args.point_conf,
                max_detections=args.max_detections,
            )

            for job, result in zip(batch_jobs, results):
                sequence_paths = job["image_paths"][job["start"] : job["end"]]
                canvases = sequence_canvases[job["sequence"]]
                width, height = canvases[sequence_paths[0].name].size
                pred_points = prediction_points_to_records(result, width, height, seq_len)
                gt_points = []
                has_ground_truth = job["json_dir"] is not None
                if has_ground_truth:
                    gt_points = load_gt_points(
                        [
                            job["json_dir"] / f"{path.stem}.json"
                            for path in sequence_paths
                        ]
                    )
                matches, unmatched_pred, unmatched_gt = match_points(
                    pred_points, gt_points, args.point_match
                ) if has_ground_truth else ([], list(range(len(pred_points))), [])
                matched_pred = {pred_idx for pred_idx, _, _ in matches}

                for pred_idx, point in enumerate(pred_points):
                    status = (
                        "tp"
                        if pred_idx in matched_pred
                        else ("fp" if has_ground_truth else "det")
                    )
                    canvas = canvases[sequence_paths[point["frame"]].name]
                    draw_circle(
                        canvas,
                        point["x"],
                        point["y"],
                        VIS_COLORS[status],
                        args.radius,
                        args.line_width,
                    )
                    tube_score = float(result["bboxes_scores"][point["group_id"]])
                    draw_prediction_label(
                        canvas,
                        point["x"],
                        point["y"],
                        tube_score,
                        point["score"],
                        VIS_COLORS[status],
                        args.radius,
                    )

                for gt_idx in unmatched_gt:
                    point = gt_points[gt_idx]
                    draw_circle(
                        canvases[sequence_paths[point["frame"]].name],
                        point["x"],
                        point["y"],
                        VIS_COLORS["fn"],
                        args.radius,
                        args.line_width,
                    )

                detections = []
                for idx in range(len(result["bboxes"])):
                    box = result["bboxes"][idx]
                    detections.append(
                        {
                            "score": float(result["bboxes_scores"][idx]),
                            "label": int(result["bboxes_labels"][idx]),
                            "bbox_normalized_xyzxyz": tensor_list(box),
                            "bbox_original_xyzxyz": [
                                float(box[0]) * width,
                                float(box[1]) * height,
                                float(box[2]) * seq_len,
                                float(box[3]) * width,
                                float(box[4]) * height,
                                float(box[5]) * seq_len,
                            ],
                        }
                    )

                point_rows = []
                for idx, point in enumerate(pred_points):
                    point_rows.append(
                        {
                            **point,
                            "status": (
                                "tp"
                                if idx in matched_pred
                                else ("fp" if has_ground_truth else "det")
                            ),
                            "tube_score": float(result["bboxes_scores"][point["group_id"]]),
                        }
                    )
                window_summary = {
                    "sequence": job["sequence"],
                    "start": job["start"],
                    "end": job["end"],
                    "images": [str(path) for path in sequence_paths],
                    "has_ground_truth": has_ground_truth,
                    "num_detections": len(detections),
                    "num_pred_points": len(pred_points),
                    "num_gt_points": len(gt_points),
                    "matches": [
                        {
                            "pred_point": pred_idx,
                            "gt_point": gt_idx,
                            "distance_px": distance,
                        }
                        for pred_idx, gt_idx, distance in matches
                    ],
                    "points": point_rows,
                    "detections": detections,
                }
                summary["windows"].append(window_summary)
                summary["totals"]["detections"] += len(detections)
                summary["totals"]["detected_points"] += len(pred_points)
                summary["totals"]["gt_points"] += len(gt_points)
                summary["totals"]["tp"] += len(matches)
                summary["totals"]["fp"] += len(unmatched_pred) if has_ground_truth else 0
                summary["totals"]["fn"] += len(unmatched_gt)

    visualizations_dir = args.output / "visualizations"
    for sequence_name, canvases in sequence_canvases.items():
        sequence_output = visualizations_dir / sequence_name
        sequence_output.mkdir(parents=True, exist_ok=True)
        for image_name, canvas in canvases.items():
            canvas.save(sequence_output / image_name)

    predictions_path = args.output / "predictions.json"
    with predictions_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Processed {len(jobs)} window(s) from {len(sequence_canvases)} sequence(s).")
    print(f"Weights: {model_info['weight_source']} from epoch {model_info['last_epoch']}")
    print(f"Totals: {summary['totals']}")
    print(f"Visualizations: {visualizations_dir}")
    print(f"Predictions: {predictions_path}")
    return summary


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
