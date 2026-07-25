"""Unified command-line interface for the healthcare risk project.

Sub-commands::

    python main.py generate   --out data/patients.csv
    python main.py train      --model gradient_boosting
    python main.py evaluate   --model random_forest
    python main.py crossval   --model logistic --folds 5
    python main.py explain    --index 0 --plot
    python main.py serve      --port 8000

Every sub-command shares the same data/model/explainer code paths so results are
consistent across interfaces.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from config import load_config
from src.exception import HCException, format_exception
from src.logger import get_logger

logger = get_logger(__name__)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--model", default=None, help="Override model name.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level parser with all sub-commands."""
    parser = argparse.ArgumentParser(description="Healthcare Risk Prediction — CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Write a synthetic dataset to CSV.")
    p_gen.add_argument("--out", default="data/patients.csv")
    _add_common(p_gen)

    p_train = sub.add_parser("train", help="Fit a model and save it.")
    _add_common(p_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate on the test split.")
    p_eval.add_argument("--plot", action="store_true")
    _add_common(p_eval)

    p_cv = sub.add_parser("crossval", help="Stratified k-fold cross-validation.")
    p_cv.add_argument("--folds", type=int, default=5)
    _add_common(p_cv)

    p_exp = sub.add_parser("explain", help="Explain one test instance (Shapley).")
    p_exp.add_argument("--index", type=int, default=0)
    p_exp.add_argument("--samples", type=int, default=None)
    p_exp.add_argument("--plot", action="store_true")
    _add_common(p_exp)

    p_serve = sub.add_parser("serve", help="Launch the FastAPI server.")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    _add_common(p_serve)

    return parser


def _apply_model_override(cfg: object, args: argparse.Namespace) -> None:
    if getattr(args, "model", None):
        cfg.model.name = args.model  # type: ignore[attr-defined]


def _load_split(cfg: object):
    from src.data_loader import DataLoader

    return DataLoader(cfg.data).load()  # type: ignore[attr-defined]


def _cmd_generate(args: argparse.Namespace) -> int:
    import csv

    from src.data_generator import generate_clinical_data

    cfg = load_config(args.config)
    dataset = generate_clinical_data(
        n_samples=cfg.data.n_samples,
        positive_rate=cfg.data.positive_rate,
        random_state=cfg.data.random_state,
    )
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(dataset.feature_names + [cfg.data.target_column])
        for row, label in zip(dataset.X, dataset.y):
            writer.writerow([f"{v:.4f}" for v in row] + [int(label)])
    print(f"Wrote {len(dataset)} rows to {args.out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from src.trainer import Trainer

    cfg = load_config(args.config)
    _apply_model_override(cfg, args)
    split = _load_split(cfg)
    model = Trainer(cfg).fit(split.x_train, split.y_train)
    path = cfg.paths.model_dir / f"{cfg.model.name}.pkl"
    model.save(path)
    print(f"Trained '{cfg.model.name}'. Saved to {path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from src.evaluator import Evaluator
    from src.trainer import Trainer
    from src.utils import write_json

    cfg = load_config(args.config)
    _apply_model_override(cfg, args)
    split = _load_split(cfg)
    model = Trainer(cfg).fit(split.x_train, split.y_train)
    report = Evaluator(cfg).evaluate(
        model, split.x_test, split.y_test, split.feature_names, split.x_train[:200]
    )
    out = cfg.paths.report_dir / f"eval_{cfg.model.name}.json"
    write_json(report, out)
    print(f"Model: {cfg.model.name}")
    for key, value in report["metrics"].items():
        print(f"  {key:12s}: {value:.4f}")
    print("Top-5 features (permutation importance):")
    for row in report["importances"]["permutation"][:5]:
        print(f"    {row['feature']:20s} {row['importance']:.4f}")
    if args.plot:
        _make_eval_plots(cfg, model, split)
    print(f"Report saved to {out}")
    return 0


def _make_eval_plots(cfg: object, model: object, split: object) -> None:
    from src.visualizer import plot_confusion, plot_importance, plot_roc_curve

    proba = model.predict_proba(split.x_test)  # type: ignore[attr-defined]
    pred = (proba >= 0.5).astype(int)
    plot_roc_curve(split.y_test, proba, cfg.paths.plot_dir / "roc.png")
    plot_confusion(split.y_test, pred, cfg.paths.plot_dir / "confusion.png")
    native = model.native_importance()  # type: ignore[attr-defined]
    if native is not None:
        plot_importance(
            split.feature_names, native, cfg.paths.plot_dir / "importance.png"
        )
    print(f"Plots saved to {cfg.paths.plot_dir}")


def _cmd_crossval(args: argparse.Namespace) -> int:
    from src.trainer import Trainer
    from src.utils import write_json

    cfg = load_config(args.config)
    _apply_model_override(cfg, args)
    split = _load_split(cfg)
    x = np.concatenate([split.x_train, split.x_val])
    y = np.concatenate([split.y_train, split.y_val])
    result = Trainer(cfg).cross_validate(x, y, n_folds=args.folds)
    out = cfg.paths.report_dir / f"cv_{cfg.model.name}.json"
    write_json({"model": cfg.model.name, **result.to_dict()}, out)
    print(f"{cfg.model.name} — {result.n_folds}-fold CV:")
    for key, value in result.mean.items():
        print(f"  {key:12s}: {value:.4f} (+/- {result.std[key]:.4f})")
    print(f"Report saved to {out}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from src.explainer import Explainer
    from src.trainer import Trainer

    cfg = load_config(args.config)
    _apply_model_override(cfg, args)
    split = _load_split(cfg)
    model = Trainer(cfg).fit(split.x_train, split.y_train)
    background = split.x_train[: cfg.explain.background_size]
    explainer = Explainer(
        model.predict_proba, background, split.feature_names, cfg.explain
    )
    idx = min(args.index, len(split.x_test) - 1)
    explanation = explainer.shapley_values(split.x_test[idx], n_samples=args.samples)
    print(f"Instance {idx}: predicted risk = {explanation.prediction:.3f}")
    print(f"Baseline (population avg) = {explanation.base_value:.3f}")
    print("Feature contributions (sorted):")
    for row in explanation.as_ranking():
        arrow = "↑" if row["contribution"] > 0 else "↓"
        print(f"    {arrow} {row['feature']:20s} {row['contribution']:+.4f}")
    if args.plot:
        from src.visualizer import plot_shapley

        path = cfg.paths.plot_dir / f"shapley_{idx}.png"
        plot_shapley(split.feature_names, explanation.values, path)
        print(f"Plot saved to {path}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    logger.info("Starting API on %s:%d", host, port)
    uvicorn.run("api:app", host=host, port=port, reload=False)
    return 0


_DISPATCH = {
    "generate": _cmd_generate,
    "train": _cmd_train,
    "evaluate": _cmd_evaluate,
    "crossval": _cmd_crossval,
    "explain": _cmd_explain,
    "serve": _cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the chosen sub-command."""
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except HCException as exc:
        logger.error("%s failed: %s", args.command, exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001
        logger.error("Unexpected error:\n%s", format_exception())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
