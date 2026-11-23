from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics_plots import (
    EvaluationResult,
    evaluate_classification,
    evaluate_clustering,
    evaluate_regression,
    plot_clusters_pca,
    plot_confusion_matrix_fig,
    plot_elbow_curve,
    plot_regression_predictions,
)


@dataclass
class ClusteringRunOutput:
    """Résultat complet d'une exécution K-Means."""
    model_name: str
    n_clusters: int
    pipeline: Pipeline
    labels: np.ndarray
    cluster_summary: pd.DataFrame
    evaluation: EvaluationResult
    figures: dict[str, plt.Figure]


@dataclass
class SupervisedRunOutput:
    """Résultat complet d'une exécution Réseau de neurones (MLP)."""
    model_name: str
    task_type: Literal["classification", "regression"]
    pipeline: Pipeline
    X_test: pd.DataFrame | np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    evaluation: EvaluationResult
    figures: dict[str, plt.Figure]


class UnsupervisedKMeansModel:
    """Modèle de clustering K-Means avec prétraitement, évaluation et visualisation."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters doit être supérieur ou égal à 2.")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.pipeline: Pipeline | None = None

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Entraîne K-Means sur X et retourne les labels de clusters."""
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    n_init="auto",
                    random_state=self.random_state,
                ),
            ),
        ])
        return self.pipeline.fit_predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit le cluster de nouvelles observations."""
        if self.pipeline is None:
            raise RuntimeError("Le modèle K-Means doit être entraîné avant predict().")
        return self.pipeline.predict(X)

    def suggest_k_scores(
        self,
        X: pd.DataFrame,
        k_range: range = range(2, 9),
    ) -> pd.DataFrame:
        """Calcule inertie et score de silhouette pour plusieurs valeurs de K."""
        X_scaled = StandardScaler().fit_transform(X)
        rows: list[dict[str, float | int]] = []
        for k in k_range:
            km = KMeans(
                n_clusters=k,
                max_iter=self.max_iter,
                n_init="auto",
                random_state=self.random_state,
            )
            labels = km.fit_predict(X_scaled)
            rows.append({
                "k": k,
                "inertia_wcss": float(km.inertia_),
                "silhouette": float(silhouette_score(X_scaled, labels)),
            })
        return pd.DataFrame(rows)

    def compute_cluster_summary(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Calcule la taille de chaque cluster et la moyenne des variables numériques."""
        df_work = X.copy()
        df_work["cluster"] = labels
        counts = df_work.groupby("cluster").size().rename("effectif")
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            means = df_work.groupby("cluster")[numeric_cols].mean()
            return pd.concat([counts, means], axis=1).reset_index()
        return counts.reset_index()

    def run(
        self,
        X: pd.DataFrame,
        k_range: range = range(2, 9),
        model_name: str = "KMeans",
    ) -> ClusteringRunOutput:
        """Exécute le pipeline complet K-Means : entraînement, métriques et graphiques."""
        labels = self.fit_predict(X)
        assert self.pipeline is not None

        X_scaled = self.pipeline.named_steps["scaler"].transform(X)
        kmeans_est: KMeans = self.pipeline.named_steps["kmeans"]

        evaluation = evaluate_clustering(
            X=X_scaled,
            labels=labels,
            inertia=float(kmeans_est.inertia_),
            model_name=model_name,
            already_scaled=True,
        )
        summary = self.compute_cluster_summary(X, labels)

        figures: dict[str, plt.Figure] = {
            "kmeans_elbow": plot_elbow_curve(
                X=X_scaled,
                k_range=k_range,
                random_state=self.random_state,
                already_scaled=True,
            ),
            "kmeans_pca_clusters": plot_clusters_pca(
                X=X_scaled,
                labels=labels,
                title=f"{model_name} — Projection ACP (K={self.n_clusters})",
                already_scaled=True,
                random_state=self.random_state,
            ),
        }

        return ClusteringRunOutput(
            model_name=model_name,
            n_clusters=self.n_clusters,
            pipeline=self.pipeline,
            labels=labels,
            cluster_summary=summary,
            evaluation=evaluation,
            figures=figures,
        )


class NeuralNetworkModel:
    """Réseau de neurones artificiels (MLP) pour la classification et la régression."""

    def __init__(
        self,
        task_type: Literal["classification", "regression"] = "classification",
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        activation: Literal["relu", "tanh", "logistic"] = "relu",
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        self.task_type = task_type
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        if self.task_type == "classification":
            estimator = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=True,
                random_state=self.random_state,
            )
        elif self.task_type == "regression":
            estimator = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=True,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Type de tâche non supporté : {self.task_type}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", estimator),
        ])

    def _plot_loss_curve(self, model_name: str) -> plt.Figure | None:
        if self.pipeline is None:
            return None
        mlp_est = self.pipeline.named_steps["mlp"]
        loss_curve = getattr(mlp_est, "loss_curve_", None)
        if not loss_curve:
            return None

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(range(1, len(loss_curve) + 1), loss_curve, linewidth=2)
        ax.set_title(f"{model_name} — Courbe d'apprentissage (Loss)")
        ax.set_xlabel("Itérations / Époques")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        test_size: float = 0.2,
        model_name: str | None = None,
    ) -> SupervisedRunOutput:
        """Entraîne le réseau de neurones, évalue sur le jeu de test et génère les figures."""
        resolved_name = model_name or (
            "MLPClassifier" if self.task_type == "classification" else "MLPRegressor"
        )
        stratify_target = y if self.task_type == "classification" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_target,
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        y_pred = np.asarray(self.pipeline.predict(X_test))
        y_test_arr = np.asarray(y_test)

        y_proba: np.ndarray | None = None
        figures: dict[str, plt.Figure] = {}

        if self.task_type == "classification":
            if hasattr(self.pipeline, "predict_proba"):
                y_proba = np.asarray(self.pipeline.predict_proba(X_test))
            evaluation = evaluate_classification(
                y_true=y_test_arr,
                y_pred=y_pred,
                model_name=resolved_name,
            )
            figures["nn_confusion_matrix"] = plot_confusion_matrix_fig(
                y_true=y_test_arr,
                y_pred=y_pred,
                title=f"{resolved_name} — Matrice de confusion",
            )
        else:
            evaluation = evaluate_regression(
                y_true=y_test_arr,
                y_pred=y_pred,
                model_name=resolved_name,
            )
            figures["nn_pred_vs_true"] = plot_regression_predictions(
                y_true=y_test_arr,
                y_pred=y_pred,
                title=f"{resolved_name} — Réel vs Prédit",
            )

        loss_fig = self._plot_loss_curve(resolved_name)
        if loss_fig is not None:
            figures["nn_loss_curve"] = loss_fig

        return SupervisedRunOutput(
            model_name=resolved_name,
            task_type=self.task_type,
            pipeline=self.pipeline,
            X_test=X_test,
            y_test=y_test_arr,
            y_pred=y_pred,
            y_proba=y_proba,
            evaluation=evaluation,
            figures=figures,
        )  