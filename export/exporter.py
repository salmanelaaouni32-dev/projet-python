from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics_plots import EvaluationResult


def create_run_folder(base_dir: str | Path = "outputs", prefix: str = "run") -> Path:
    """Crée un sous-dossier horodaté dans outputs/ pour chaque exécution."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def export_evaluation_report(
    report: EvaluationResult,
    output_dir: str | Path,
    filename_stem: str = "evaluation",
) -> dict[str, Path]:
    """Exporte les métriques en JSON et en CSV."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{filename_stem}.json"
    json_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out_dir / f"{filename_stem}.csv"
    df_metrics = pd.DataFrame([
        {"model_name": report.model_name, "task_type": report.task_type, **report.metrics}
    ])
    df_metrics.to_csv(csv_path, index=False)

    return {"json": json_path, "csv": csv_path}


def export_predictions_table(
    df_results: pd.DataFrame,
    output_dir: str | Path,
    filename_stem: str = "resultats",
    export_excel: bool = True,
) -> dict[str, Path]:
    """Exporte le tableau des prédictions ou clusters en CSV (et optionnellement XLSX)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{filename_stem}.csv"
    df_results.to_csv(csv_path, index=False)
    exported: dict[str, Path] = {"csv": csv_path}

    if export_excel:
        xlsx_path = out_dir / f"{filename_stem}.xlsx"
        df_results.to_excel(xlsx_path, index=False)
        exported["xlsx"] = xlsx_path

    return exported


def export_model(
    model: Any,
    output_dir: str | Path,
    filename_stem: str = "modele_entraine",
) -> Path:
    """Sérialise un modèle ou un pipeline scikit-learn avec joblib."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{filename_stem}.joblib"
    joblib.dump(model, model_path)
    return model_path


def export_figures(
    figures: dict[str, plt.Figure],
    output_dir: str | Path,
    dpi: int = 300,
    close_after: bool = False,
) -> dict[str, Path]:
    """Sauvegarde un dictionnaire {nom_figure: fig} au format PNG."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for name, fig in figures.items():
        fig_path = out_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        saved_paths[name] = fig_path
        if close_after:
            plt.close(fig)

    return saved_paths