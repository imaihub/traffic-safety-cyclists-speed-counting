"""Command-line interface: ``cyclist-tracker <command> ...`` (run ``cyclist-tracker -h`` for help)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("transformers", "timm", "matplotlib", "PIL", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    try:
        import transformers

        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


# ------------------------------------------------------------------------------------ commands
def cmd_gui(args) -> int:
    from .gui.app import main as gui_main

    return gui_main([])


def cmd_check(args) -> int:
    from .system import system_report

    for item, value, ok in system_report():
        print(f"{'OK ' if ok else ' - '} {item:<26} {value}")
    return 0


def cmd_demo(args) -> int:
    from .synthetic import generate_demo_dataset

    path = generate_demo_dataset(args.output, make_video=not args.no_video)
    print(f"Demo dataset created in {path}")
    return 0


def cmd_dataset(args) -> int:
    from . import dataset as ds

    if args.action == "validate":
        report = ds.validate_dataset(args.path)
        print(report.as_text())
        return 1 if report.errors else 0
    if args.action == "init":
        created = ds.create_layout(args.path)
        print(f"Created {len(created)} folder(s) in {args.path}")
    elif args.action == "extract-frames":
        n = ds.extract_frames(args.video, args.output, every_n=args.every)
        print(f"Saved {n} frames to {args.output}")
    elif args.action == "frame-list":
        n = ds.write_frame_list(args.coco, args.output, only_annotated=not args.all)
        print(f"Wrote {n} frame names to {args.output}")
    elif args.action == "coco-to-mot":
        n = ds.coco_to_mot(args.coco, args.output)
        print(f"Wrote {n} ground-truth boxes to {args.output}")
    elif args.action == "flatten":
        n = ds.flatten_file_names(args.coco, args.output, args.split)
        print(f"Rewrote {n} file names into {args.output}")
    return 0


def cmd_scene(args) -> int:
    from .config import SceneConfig, utrecht_scene

    if args.action == "paper":
        utrecht_scene().save(args.output)
        print(f"Wrote the scene of the paper to {args.output}")
        return 0
    scene = SceneConfig.load(args.path)
    problems = scene.problems()
    print("\n".join(problems) if problems else f"Scene '{scene.name}' is complete.")
    return 1 if problems else 0


def _settings_from_args(args):
    from .config import AnalysisSettings

    settings = AnalysisSettings.paper() if args.paper_settings else AnalysisSettings()
    if args.settings:
        settings = AnalysisSettings.load(args.settings)
    overrides = {
        "embedder": args.embedder, "embedder_weights": args.embedder_weights, "fps": args.fps,
        "speed_mode": args.speed_mode, "tracking_point": args.tracking_point,
        "confidence_threshold": args.confidence,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    if args.cpu:
        settings.use_gpu = False
    return settings


def cmd_analyze(args) -> int:
    from .config import SceneConfig
    from .detection import create_detector
    from .pipeline import open_source, run_analysis
    from .tracking import CyclistTracker

    scene = SceneConfig.load(args.scene)
    settings = _settings_from_args(args)
    source = open_source(args.source, args.frame_list, args.stride, natural_sort=not args.legacy_sort)
    if args.fps is None and getattr(source, "fps", None):
        settings.fps = source.fps
    problems = scene.problems() + settings.problems()
    if problems:
        print("Cannot start:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    detector = create_detector(args.detector, checkpoint=args.checkpoint, annotations=args.annotations,
                               threshold=settings.confidence_threshold, prefer_gpu=settings.use_gpu)
    tracker = CyclistTracker(settings)
    total = len(source)

    def progress(done, _total, counts):
        if done % args.progress_interval == 0 or done == total:
            print(f"  frame {done}/{total}  counts: {counts}", flush=True)

    result = run_analysis(source, detector, tracker, scene, settings, args.output_dir, args.name,
                          save_video=not args.no_video, progress=progress)
    print("\nResult")
    for direction, count in result.counts.items():
        speed = result.speeds_by_direction.get(direction)
        print(f"  {direction:<12} {count:>4} cyclists" + (f"   average {speed:.1f} km/h" if speed else ""))
    print("Files:\n  " + "\n  ".join(f"{k:<8} {v}" for k, v in result.output_files.items()))
    return 0


def cmd_train(args) -> int:
    from .training import TrainConfig, train_detector

    cfg = TrainConfig(dataset_dir=args.dataset, output_dir=args.output, epochs=args.epochs, batch_size=args.batch_size,
                      val_batch_size=args.val_batch_size, lr=args.lr, lr_backbone=args.lr_backbone,
                      weight_decay=args.weight_decay, num_workers=args.num_workers, use_gpu=not args.cpu,
                      precision=args.precision, accumulate_grad_batches=args.accumulate_grad_batches)
    best = train_detector(cfg)
    print(f"Best checkpoint: {best}")
    return 0 if best else 1


def cmd_evaluate(args) -> int:
    import pandas as pd

    from . import evaluation as ev

    pd.set_option("display.width", 160)
    if args.kind == "tracking":
        table = ev.evaluate_tracking(args.gt, args.pred, args.iou)
        shown = table[list(ev.MOT_METRICS)]
    elif args.kind == "speed":
        table, matches = ev.evaluate_speeds(args.gt, args.pred, args.names, args.tolerance)
        shown = table
        if args.plot:
            ev.speed_scatter_figure(matches).savefig(args.plot, dpi=150)
            print(f"Plot saved to {args.plot}")
    else:
        table = ev.evaluate_counts(args.gt, args.pred, args.tolerance)
        shown = table
    print(shown.round(4).to_string())
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)
        print(f"Saved to {args.output}")
    return 0


# -------------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cyclist-tracker", description="Cyclist counting and speed estimation.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser("gui", help="open the graphical user interface").set_defaults(func=cmd_gui)
    sub.add_parser("check", help="show Python, PyTorch and GPU information").set_defaults(func=cmd_check)

    s = sub.add_parser("demo", help="create a synthetic demo dataset")
    s.add_argument("output", help="folder to create")
    s.add_argument("--no-video", action="store_true", help="do not write the demo video")
    s.set_defaults(func=cmd_demo)

    d = sub.add_parser("dataset", help="check or prepare a dataset")
    dsub = d.add_subparsers(dest="action", required=True, metavar="ACTION")
    x = dsub.add_parser("validate", help="check the folder structure and annotation files")
    x.add_argument("path")
    x = dsub.add_parser("init", help="create the empty folder structure")
    x.add_argument("path")
    x = dsub.add_parser("extract-frames", help="save every N-th frame of a video as images")
    x.add_argument("video")
    x.add_argument("output")
    x.add_argument("--every", type=int, default=5, help="keep every N-th frame (25 fps / 5 = 5 fps)")
    x = dsub.add_parser("frame-list", help="list the annotated images of a COCO file")
    x.add_argument("coco")
    x.add_argument("output")
    x.add_argument("--all", action="store_true", help="also list images without boxes")
    x = dsub.add_parser("coco-to-mot", help="tracking ground truth from a COCO file with track_id attributes")
    x.add_argument("coco")
    x.add_argument("output")
    x = dsub.add_parser("flatten", help="rewrite file_name entries to '<split>/<image>'")
    x.add_argument("coco")
    x.add_argument("output")
    x.add_argument("--split", required=True, choices=["train", "val", "test"])
    d.set_defaults(func=cmd_dataset)

    c = sub.add_parser("scene", help="check a camera scene file or write the scene of the paper")
    csub = c.add_subparsers(dest="action", required=True, metavar="ACTION")
    x = csub.add_parser("check", help="check a scene file")
    x.add_argument("path")
    x = csub.add_parser("paper", help="write the Utrecht scene used in the paper")
    x.add_argument("output")
    c.set_defaults(func=cmd_scene)

    a = sub.add_parser("analyze", help="detect, track, count and measure speeds")
    a.add_argument("source", help="folder with frames or a video file")
    a.add_argument("--scene", required=True, help="scene JSON file (see the 'Camera scene' tab)")
    a.add_argument("--detector", default="checkpoint", choices=["checkpoint", "coco", "annotations"])
    a.add_argument("--checkpoint", help="fine-tuned DETR .ckpt file (detector 'checkpoint')")
    a.add_argument("--annotations", help="COCO file with boxes (detector 'annotations')")
    a.add_argument("--frame-list", help="text file with the frames to use (one name per line)")
    a.add_argument("--stride", type=int, default=1, help="for videos: use every N-th frame")
    a.add_argument("--output-dir", default="results")
    a.add_argument("--name", default="analysis", help="prefix of the output files")
    a.add_argument("--settings", help="JSON file with analysis settings")
    a.add_argument("--paper-settings", action="store_true", help="use the settings of the original paper")
    a.add_argument("--embedder", help="DeepSORT feature extractor: mobilenet, torchreid, clip_ViT-B/32, histogram")
    a.add_argument("--embedder-weights", help="weights file for the feature extractor (CLIP)")
    a.add_argument("--fps", type=float, help="frame rate of the processed frames (default 5, or from the video)")
    a.add_argument("--speed-mode", choices=["homography", "fixed_distance"])
    a.add_argument("--tracking-point", choices=["bottom", "center"])
    a.add_argument("--confidence", type=float, help="minimum detection confidence (0-1)")
    a.add_argument("--cpu", action="store_true", help="do not use the GPU")
    a.add_argument("--no-video", action="store_true", help="do not write the annotated video")
    a.add_argument("--legacy-sort", action="store_true",
                   help="sort frames alphabetically like the original code (to reproduce old results)")
    a.add_argument("--progress-interval", type=int, default=50)
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("train", help="fine-tune the DETR cyclist detector")
    t.add_argument("--dataset", required=True, help="dataset folder (see docs/DATASET.md)")
    t.add_argument("--output", default="runs/detr")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--val-batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lr-backbone", type=float, default=1e-5)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--accumulate-grad-batches", type=int, default=8)
    t.add_argument("--precision", default="32-true", help="e.g. 16-mixed for less GPU memory")
    t.add_argument("--num-workers", type=int, default=2)
    t.add_argument("--cpu", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="compare results with ground truth")
    esub = e.add_subparsers(dest="kind", required=True, metavar="KIND")
    x = esub.add_parser("tracking", help="MOT metrics (IDF1, MOTA, ...)")
    x.add_argument("--gt", required=True)
    x.add_argument("--pred", required=True)
    x.add_argument("--iou", type=float, default=0.5)
    x.add_argument("--output")
    x = esub.add_parser("speed", help="MAE, RMSE and R2 of speeds")
    x.add_argument("--gt", required=True)
    x.add_argument("--pred", required=True, nargs="+")
    x.add_argument("--names", nargs="+")
    x.add_argument("--tolerance", type=int, default=5, help="max. frame difference for matching")
    x.add_argument("--plot", help="save a scatter plot (png)")
    x.add_argument("--output")
    x = esub.add_parser("counting", help="TP/FP/FN and F1 of counted crossings")
    x.add_argument("--gt", required=True)
    x.add_argument("--pred", required=True)
    x.add_argument("--tolerance", type=int, default=10)
    x.add_argument("--output")
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args) or 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
