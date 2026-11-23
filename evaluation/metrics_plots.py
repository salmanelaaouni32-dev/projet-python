from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

TaskType = Literal["clustering", "classification", "regression"]


@dataclass
class EvaluationResult:
    """Résultat standardisé d'évaluation pour tout modèle du projet."""
    task_type: TaskType
    model_name: str
    metrics: dict[str, float]


def evaluate_clustering(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    inertia: float | None = None,
    model_name: str = "KMeans",
    already_scaled: bool = False,
) -> EvaluationResult:
    """Calcule les métriques d'évaluation d'un clustering (non supervisé)."""
    X_arr = np.asarray(X)
    labels_arr = np.asarray(labels)
    X_eval = X_arr if already_scaled else StandardScaler().fit_transform(X_arr)

    metrics: dict[str, float] = {
        "davies_bouldin": float(davies_bouldin_score(X_eval, labels_arr)),
    }
    if inertia is not None:
        metrics["inertia_wcss"] = float(inertia)

    unique_clusters = np.unique(labels_arr)
    if len(unique_clusters) > 1 and len(labels_arr) > len(unique_clusters):
        metrics["silhouette"] = float(silhouette_score(X_eval, labels_arr))

    return EvaluationResult(task_type="clustering", model_name=model_name, metrics=metrics)


def evaluate_classification(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Classifier",
) -> EvaluationResult:
    """Calcule Accuracy, Précision, Rappel et F1-score."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    average_mode = "binary" if len(np.unique(y_t)) == 2 else "weighted"

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "precision": float(precision_score(y_t, y_p, average=average_mode, zero_division=0)),
        "recall": float(recall_score(y_t, y_p, average=average_mode, zero_division=0)),
        "f1_score": float(f1_score(y_t, y_p, average=average_mode, zero_division=0)),
    }
    return EvaluationResult(task_type="classification", model_name=model_name, metrics=metrics)


def evaluate_regression(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Regressor",
) -> EvaluationResult:
    """Calcule MAE, RMSE et R²."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(y_t, y_p)),
        "rmse": float(root_mean_squared_error(y_t, y_p)),
        "r2": float(r2_score(y_t, y_p)),
    }
    return EvaluationResult(task_type="regression", model_name=model_name, metrics=metrics)


def plot_elbow_curve(
    X: pd.DataFrame | np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42,
    already_scaled: bool = False,
) -> plt.Figure:
    """Trace la méthode du coude (inertie WCSS) pour aider au choix de K."""
    X_arr = np.asarray(X)
    X_scaled = X_arr if already_scaled else StandardScaler().fit_transform(X_arr)

    ks = list(k_range)
    inertias: list[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, inertias, marker="o", linewidth=2)
    ax.set_title("Méthode du coude (K-Means)")
    ax.set_xlabel("Nombre de clusters (K)")
    ax.set_ylabel("Inertie intra-classe (WCSS)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_clusters_pca(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    title: str = "Visualisation des clusters (ACP 2D)",
    already_scaled: bool = False,
    random_state: int = 42,
) -> plt.Figure:
    """Projette les données en 2D via ACP et affiche les clusters."""
    X_arr = np.asarray(X)
    X_scaled = X_arr if already_scaled else StandardScaler().fit_transform(X_arr)
    coords = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.asarray(labels),
        cmap="tab10",
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )
    ax.set_title(title)
    ax.set_xlabel("Composante principale 1")
    ax.set_ylabel("Composante principale 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    return fig


def plot_confusion_matrix_fig(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matrice de confusion",
) -> plt.Figure:
    """Génère la figure de la matrice de confusion."""
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_regression_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    title: str = "Valeurs réelles vs Prédictions",
) -> plt.Figure:
    """Compare les valeurs réelles et les valeurs prédites en régression."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y_t, y_p, alpha=0.75, edgecolors="k", linewidths=0.4)
    min_val = min(float(np.min(y_t)), float(np.min(y_p)))
    max_val = max(float(np.max(y_t)), float(np.max(y_p)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Idéal (y = ŷ)")
    ax.set_title(title)
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Valeurs prédites")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig