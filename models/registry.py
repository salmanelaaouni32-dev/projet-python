from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from evaluation.metrics_plots import (
    EvaluationResult,
    evaluate_classification,
    evaluate_regression,
    plot_confusion_matrix_fig,
    plot_regression_predictions,
)
from models.unsupervised_nn import NeuralNetworkModel, UnsupervisedKMeansModel

try:
    import models.linear_bayes as linear_bayes_mod
except Exception:
    linear_bayes_mod = None  # type: ignore

try:
    import models.trees_svm_knn as trees_svm_knn_mod
except Exception:
    trees_svm_knn_mod = None  # type: ignore


@dataclass
class UnifiedModelSpec:
    key: str
    display_name: str
    task_type: Literal["regression", "classification", "clustering"]
    owner: str
    module_path: str
    class_candidates: list[str]
    description: str
    tutorial: str
    hyperparams_schema: dict[str, dict[str, Any]]


@dataclass
class UnifiedRunResult:
    model_key: str
    model_name: str
    task_type: Literal["regression", "classification", "clustering"]
    source_backend: str
    pipeline: Any
    evaluation: EvaluationResult
    figures: dict[str, plt.Figure]
    predictions_df: pd.DataFrame
    cv_scores: list[float] | None = None
    extra_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


MODEL_CATALOG: dict[str, UnifiedModelSpec] = {
    "linear_regression": UnifiedModelSpec(
        key="linear_regression",
        display_name="Régression Linéaire (Simple / Multiple)",
        task_type="regression",
        owner="Wissal (Tâche 2)",
        module_path="models/linear_bayes.py",
        class_candidates=["LinearRegressionModel", "MultipleLinearRegressionModel", "LinearModel"],
        description="Modélise une relation linéaire entre les variables explicatives et une cible continue.",
        tutorial="**Principe :** Ajuste un hyperplan minimisant la somme des carrés des résidus.\n**Conseil :** Sélectionnez 1 variable pour une régression simple, ou plusieurs pour une régression multiple.",
        hyperparams_schema={
            "fit_intercept": {"type": "bool", "default": True, "label": "Calculer l'ordonnée à l'origine (fit_intercept)"},
        },
    ),
    "polynomial_regression": UnifiedModelSpec(
        key="polynomial_regression",
        display_name="Régression Polynomiale",
        task_type="regression",
        owner="Wissal (Tâche 2)",
        module_path="models/linear_bayes.py",
        class_candidates=["PolynomialRegressionModel", "PolynomialModel"],
        description="Capture des relations non-linéaires en générant des combinaisons polynomiales des caractéristiques.",
        tutorial="**Principe :** Transforme les variables en termes polynomiaux de degré $d$ avant la régression linéaire.\n**Attention :** Un degré > 3 augmente le risque de surapprentissage.",
        hyperparams_schema={
            "degree": {"type": "int", "min": 2, "max": 5, "default": 2, "label": "Degré du polynôme"},
            "include_bias": {"type": "bool", "default": False, "label": "Inclure le terme de biais"},
        },
    ),
    "logistic_regression": UnifiedModelSpec(
        key="logistic_regression",
        display_name="Régression Logistique",
        task_type="classification",
        owner="Wissal (Tâche 2)",
        module_path="models/linear_bayes.py",
        class_candidates=["LogisticRegressionModel", "LogisticModel"],
        description="Modèle linéaire probabiliste pour la classification binaire ou multiclasse.",
        tutorial="**Principe :** Applique la fonction sigmoïde / softmax pour estimer les probabilités d'appartenance aux classes.",
        hyperparams_schema={
            "C": {"type": "float", "min": 0.01, "max": 20.0, "default": 1.0, "label": "Régularisation (C)"},
            "max_iter": {"type": "int", "min": 100, "max": 2000, "default": 500, "label": "Itérations max"},
        },
    ),
    "naive_bayes": UnifiedModelSpec(
        key="naive_bayes",
        display_name="Naïf Bayésien (Gaussian Naive Bayes)",
        task_type="classification",
        owner="Wissal (Tâche 2)",
        module_path="models/linear_bayes.py",
        class_candidates=["NaiveBayesModel", "GaussianNBModel"],
        description="Classifieur probabiliste rapide basé sur le théorème de Bayes.",
        tutorial="**Principe :** Calcule $P(y|X) \\propto P(y)\\prod_i P(x_i|y)$ sous hypothèse gaussienne.",
        hyperparams_schema={
            "var_smoothing": {"type": "float", "min": 1e-11, "max": 1e-5, "default": 1e-9, "label": "Lissage de variance"},
        },
    ),
    "decision_tree": UnifiedModelSpec(
        key="decision_tree",
        display_name="Arbre de Décision (Decision Tree)",
        task_type="classification",
        owner="Nour (Tâche 3)",
        module_path="models/trees_svm_knn.py",
        class_candidates=["DecisionTreeModel", "DecisionTreeClassifierModel"],
        description="Modèle hiérarchique interprétable fondé sur des règles de décision successives.",
        tutorial="**Principe :** Partitionne récursivement l'espace selon le critère d'impureté (Gini ou Entropie).",
        hyperparams_schema={
            "criterion": {"type": "select", "options": ["gini", "entropy", "log_loss"], "default": "gini", "label": "Critère"},
            "max_depth": {"type": "int", "min": 1, "max": 30, "default": 6, "label": "Profondeur max"},
            "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2, "label": "Min. échantillons split"},
        },
    ),
    "random_forest": UnifiedModelSpec(
        key="random_forest",
        display_name="Forêt Aléatoire (Random Forest)",
        task_type="classification",
        owner="Nour (Tâche 3)",
        module_path="models/trees_svm_knn.py",
        class_candidates=["RandomForestModel", "RandomForestClassifierModel"],
        description="Ensemble d'arbres de décision entraînés par bagging pour réduire la variance.",
        tutorial="**Principe :** Agrège les votes de nombreux arbres entraînés sur des sous-échantillons bootstrap.",
        hyperparams_schema={
            "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100, "label": "Nombre d'arbres"},
            "max_depth": {"type": "int", "min": 2, "max": 30, "default": 10, "label": "Profondeur max"},
        },
    ),
    "svm": UnifiedModelSpec(
        key="svm",
        display_name="Support Vector Machine (SVM)",
        task_type="classification",
        owner="Nour (Tâche 3)",
        module_path="models/trees_svm_knn.py",
        class_candidates=["SVMModel", "SVCModel"],
        description="Recherche l'hyperplan séparateur maximisant la marge entre les classes.",
        tutorial="**Principe :** Maximise la marge entre les classes grâce aux vecteurs supports et aux noyaux (`rbf`, `linear`, `poly`).",
        hyperparams_schema={
            "kernel": {"type": "select", "options": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf", "label": "Noyau (Kernel)"},
            "C": {"type": "float", "min": 0.01, "max": 50.0, "default": 1.0, "label": "Paramètre de marge (C)"},
        },
    ),
    "knn": UnifiedModelSpec(
        key="knn",
        display_name="K Plus Proches Voisins (KNN)",
        task_type="classification",
        owner="Nour (Tâche 3)",
        module_path="models/trees_svm_knn.py",
        class_candidates=["KNNModel", "KNeighborsModel"],
        description="Classe chaque observation selon le vote majoritaire de ses K voisins les plus proches.",
        tutorial="**Principe :** Algorithme basé sur la distance (Euclidienne, Manhattan). Nécessite une normalisation des variables.",
        hyperparams_schema={
            "n_neighbors": {"type": "int", "min": 1, "max": 25, "default": 5, "label": "Nombre de voisins (K)"},
            "weights": {"type": "select", "options": ["uniform", "distance"], "default": "uniform", "label": "Pondération"},
            "metric": {"type": "select", "options": ["euclidean", "manhattan", "minkowski"], "default": "euclidean", "label": "Distance"},
        },
    ),
    "kmeans": UnifiedModelSpec(
        key="kmeans",
        display_name="K-Means (Clustering Non Supervisé)",
        task_type="clustering",
        owner="Salmane (Tâche 4)",
        module_path="models/unsupervised_nn.py",
        class_candidates=["UnsupervisedKMeansModel"],
        description="Partitionne automatiquement les données sans étiquettes en K groupes homogènes.",
        tutorial="**Principe :** Minimise l'inertie intra-classe (WCSS). Utilisez la courbe du coude et le score de Silhouette.",
        hyperparams_schema={
            "n_clusters": {"type": "int", "min": 2, "max": 12, "default": 3, "label": "Nombre de clusters (K)"},
            "max_iter": {"type": "int", "min": 100, "max": 1000, "default": 300, "label": "Itérations max"},
        },
    ),
    "neural_network_clf": UnifiedModelSpec(
        key="neural_network_clf",
        display_name="Réseau de Neurones Artificiels (MLP - Classification)",
        task_type="classification",
        owner="Salmane (Tâche 4)",
        module_path="models/unsupervised_nn.py",
        class_candidates=["NeuralNetworkModel"],
        description="Perceptron Multicouche (MLP) capable d'apprendre des frontières non-linéaires complexes.",
        tutorial="**Principe :** Couches denses entraînées par rétropropagation du gradient.",
        hyperparams_schema={
            "hidden_layers": {"type": "select", "options": ["(32,)", "(64, 32)", "(128, 64)", "(64, 32, 16)"], "default": "(64, 32)", "label": "Couches cachées"},
            "activation": {"type": "select", "options": ["relu", "tanh", "logistic"], "default": "relu", "label": "Activation"},
            "learning_rate_init": {"type": "float", "min": 0.0001, "max": 0.1, "default": 0.001, "label": "Learning rate"},
            "max_iter": {"type": "int", "min": 100, "max": 1500, "default": 500, "label": "Époques max"},
        },
    ),
    "neural_network_reg": UnifiedModelSpec(
        key="neural_network_reg",
        display_name="Réseau de Neurones Artificiels (MLP - Régression)",
        task_type="regression",
        owner="Salmane (Tâche 4)",
        module_path="models/unsupervised_nn.py",
        class_candidates=["NeuralNetworkModel"],
        description="Perceptron Multicouche (MLP) dédié à la prédiction de variables numériques continues.",
        tutorial="**Principe :** Réseau de neurones profond avec sortie linéaire optimisant l'erreur quadratique.",
        hyperparams_schema={
            "hidden_layers": {"type": "select", "options": ["(32,)", "(64, 32)", "(128, 64)", "(64, 32, 16)"], "default": "(64, 32)", "label": "Couches cachées"},
            "activation": {"type": "select", "options": ["relu", "tanh", "logistic"], "default": "relu", "label": "Activation"},
            "learning_rate_init": {"type": "float", "min": 0.0001, "max": 0.1, "default": 0.001, "label": "Learning rate"},
            "max_iter": {"type": "int", "min": 100, "max": 1500, "default": 500, "label": "Époques max"},
        },
    ),
}


def _find_team_class(model_key: str) -> type | None:
    spec = MODEL_CATALOG[model_key]
    target_module = linear_bayes_mod if "linear_bayes" in spec.module_path else (
        trees_svm_knn_mod if "trees_svm_knn" in spec.module_path else None
    )
    if target_module is None:
        return None
    for cls_name in spec.class_candidates:
        cls = getattr(target_module, cls_name, None)
        if cls is not None and inspect.isclass(cls):
            return cls
    return None


def get_implementation_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for key, spec in MODEL_CATALOG.items():
        if key in ("kmeans", "neural_network_clf", "neural_network_reg"):
            status[key] = {"implemented_in_team_file": True, "source_label": f"Implémenté ({spec.owner})", "ready": True}
        else:
            cls = _find_team_class(key)
            if cls is not None:
                status[key] = {"implemented_in_team_file": True, "source_label": f"Implémenté ({spec.owner} - {cls.__name__})", "ready": True}
            else:
                status[key] = {"implemented_in_team_file": False, "source_label": f"Relais actif (en attente de {spec.owner})", "ready": True}
    return status


def _parse_hidden_layers(layers_str: str) -> tuple[int, ...]:
    nums = [int(x.strip()) for x in layers_str.strip("() ").split(",") if x.strip().isdigit()]
    return tuple(nums) if nums else (64, 32)


def _build_sklearn_estimator(model_key: str, params: dict[str, Any], random_state: int = 42) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if model_key == "linear_regression":
        return Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression(fit_intercept=bool(params.get("fit_intercept", True))))])
    if model_key == "polynomial_regression":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=int(params.get("degree", 2)), include_bias=bool(params.get("include_bias", False)))),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])
    if model_key == "logistic_regression":
        return Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=float(params.get("C", 1.0)), max_iter=int(params.get("max_iter", 500)), random_state=random_state))])
    if model_key == "naive_bayes":
        return Pipeline([("scaler", StandardScaler()), ("classifier", GaussianNB(var_smoothing=float(params.get("var_smoothing", 1e-9))))])
    if model_key == "decision_tree":
        return DecisionTreeClassifier(criterion=str(params.get("criterion", "gini")), max_depth=int(params.get("max_depth", 6)), min_samples_split=int(params.get("min_samples_split", 2)), random_state=random_state)
    if model_key == "random_forest":
        return RandomForestClassifier(n_estimators=int(params.get("n_estimators", 100)), max_depth=int(params.get("max_depth", 10)), random_state=random_state)
    if model_key == "svm":
        return Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel=str(params.get("kernel", "rbf")), C=float(params.get("C", 1.0)), probability=True, random_state=random_state))])
    if model_key == "knn":
        return Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=int(params.get("n_neighbors", 5)), weights=str(params.get("weights", "uniform")), metric=str(params.get("metric", "euclidean"))))])
    raise ValueError(f"Modèle inconnu : {model_key}")


def _can_stratify(y: pd.Series | np.ndarray) -> bool:
    vals, counts = np.unique(np.asarray(y), return_counts=True)
    return len(vals) >= 2 and int(counts.min()) >= 2


def execute_model(
    model_key: str,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    test_size: float = 0.2,
    cv_folds: int = 0,
    random_state: int = 42,
    target_classes: list[str] | None = None,
) -> UnifiedRunResult:
    spec = MODEL_CATALOG[model_key]
    p = params or {}

    if X is None or X.empty or X.shape[1] == 0:
        raise ValueError("La matrice de caractéristiques X est vide.")

    if model_key == "kmeans":
        n_clusters = min(int(p.get("n_clusters", 3)), max(2, len(X) - 1))
        km_model = UnsupervisedKMeansModel(n_clusters=n_clusters, max_iter=int(p.get("max_iter", 300)), random_state=random_state)
        km_out = km_model.run(X=X, k_range=range(2, min(9, max(3, len(X)))), model_name=spec.display_name)
        df_clustered = X.copy()
        df_clustered["cluster"] = km_out.labels
        return UnifiedRunResult(
            model_key=model_key,
            model_name=spec.display_name,
            task_type="clustering",
            source_backend="team_module (Salmane)",
            pipeline=km_out.pipeline,
            evaluation=km_out.evaluation,
            figures=km_out.figures,
            predictions_df=df_clustered,
            extra_tables={"Résumé des clusters": km_out.cluster_summary},
        )

    if y is None:
        raise ValueError(f"Le modèle '{spec.display_name}' nécessite une variable cible (y).")

    y_arr = np.asarray(y)
    if len(y_arr) < 6:
        raise ValueError("Au moins 6 observations sont nécessaires pour l'entraînement.")

    if model_key in ("neural_network_clf", "neural_network_reg"):
        task_t: Literal["classification", "regression"] = "classification" if model_key == "neural_network_clf" else "regression"
        if task_t == "classification" and (len(np.unique(y_arr)) < 2 or not _can_stratify(y_arr)):
            raise ValueError("La cible doit contenir au moins 2 classes avec au minimum 2 exemples chacune.")
        nn_model = NeuralNetworkModel(
            task_type=task_t,
            hidden_layer_sizes=_parse_hidden_layers(str(p.get("hidden_layers", "(64, 32)"))),
            activation=str(p.get("activation", "relu")),  # type: ignore
            learning_rate_init=float(p.get("learning_rate_init", 0.001)),
            max_iter=int(p.get("max_iter", 500)),
            random_state=random_state,
        )
        nn_out = nn_model.run(X=X, y=y_arr, test_size=test_size, model_name=spec.display_name)
        pred_df = pd.DataFrame({"y_reel": nn_out.y_test, "y_predit": nn_out.y_pred})
        if target_classes and task_t == "classification":
            mapping = dict(enumerate(target_classes))
            pred_df["classe_reelle"] = [mapping.get(int(v), v) for v in nn_out.y_test]
            pred_df["classe_predite"] = [mapping.get(int(v), v) for v in nn_out.y_pred]
        return UnifiedRunResult(
            model_key=model_key,
            model_name=spec.display_name,
            task_type=task_t,
            source_backend="team_module (Salmane)",
            pipeline=nn_out.pipeline,
            evaluation=nn_out.evaluation,
            figures=nn_out.figures,
            predictions_df=pred_df,
        )

    if spec.task_type == "classification" and len(np.unique(y_arr)) < 2:
        raise ValueError("La classification nécessite au moins 2 classes distinctes.")

    team_cls = _find_team_class(model_key)
    source_backend = f"team_module ({spec.owner})" if team_cls is not None else f"sklearn_fallback ({spec.owner})"

    if team_cls is not None:
        sig = inspect.signature(team_cls.__init__)
        valid_kwargs = {k: v for k, v in p.items() if k in sig.parameters}
        if "random_state" in sig.parameters:
            valid_kwargs["random_state"] = random_state
        estimator_obj = team_cls(**valid_kwargs)
        if hasattr(estimator_obj, "run"):
            raw_out = estimator_obj.run(X=X, y=y_arr, test_size=test_size)
            if hasattr(raw_out, "evaluation") and hasattr(raw_out, "figures"):
                pred_df = pd.DataFrame({"y_reel": raw_out.y_test, "y_predit": raw_out.y_pred})
                return UnifiedRunResult(
                    model_key=model_key,
                    model_name=spec.display_name,
                    task_type=spec.task_type,  # type: ignore
                    source_backend=source_backend,
                    pipeline=getattr(raw_out, "pipeline", estimator_obj),
                    evaluation=raw_out.evaluation,
                    figures=raw_out.figures,
                    predictions_df=pred_df,
                )
        estimator = estimator_obj
    else:
        estimator = _build_sklearn_estimator(model_key, p, random_state=random_state)

    stratify = y_arr if (spec.task_type == "classification" and _can_stratify(y_arr)) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y_arr, test_size=test_size, random_state=random_state, stratify=stratify)
    estimator.fit(X_train, y_train)
    y_pred = np.asarray(estimator.predict(X_test))

    cv_scores: list[float] | None = None
    if cv_folds and cv_folds >= 2 and len(X) >= cv_folds * 2:
        try:
            scoring = "accuracy" if spec.task_type == "classification" else "r2"
            scores = cross_val_score(clone(estimator), X, y_arr, cv=cv_folds, scoring=scoring)
            cv_scores = [round(float(s), 4) for s in scores]
        except Exception:
            cv_scores = None

    figures: dict[str, plt.Figure] = {}
    if spec.task_type == "classification":
        evaluation = evaluate_classification(y_true=y_test, y_pred=y_pred, model_name=spec.display_name)
        figures["confusion_matrix"] = plot_confusion_matrix_fig(y_true=y_test, y_pred=y_pred, title=f"{spec.display_name} — Matrice de confusion")
    else:
        evaluation = evaluate_regression(y_true=y_test, y_pred=y_pred, model_name=spec.display_name)
        figures["pred_vs_true"] = plot_regression_predictions(y_true=y_test, y_pred=y_pred, title=f"{spec.display_name} — Valeurs réelles vs Prédites")

    feat_imp_fig = _plot_feature_importance_if_available(estimator, X.columns.tolist(), spec.display_name)
    if feat_imp_fig is not None:
        figures["feature_importance"] = feat_imp_fig

    pred_df = pd.DataFrame({"y_reel": y_test, "y_predit": y_pred})
    if target_classes and spec.task_type == "classification":
        mapping = dict(enumerate(target_classes))
        pred_df["classe_reelle"] = [mapping.get(int(v), v) for v in y_test]
        pred_df["classe_predite"] = [mapping.get(int(v), v) for v in y_pred]

    return UnifiedRunResult(
        model_key=model_key,
        model_name=spec.display_name,
        task_type=spec.task_type,  # type: ignore
        source_backend=source_backend,
        pipeline=estimator,
        evaluation=evaluation,
        figures=figures,
        predictions_df=pred_df,
        cv_scores=cv_scores,
    )


def _plot_feature_importance_if_available(estimator: Any, feature_names: list[str], model_name: str) -> plt.Figure | None:
    inner = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    importances = None
    label = "Importance"
    if hasattr(inner, "feature_importances_"):
        importances = np.asarray(inner.feature_importances_)
        label = "Importance (Gini / Variance)"
    elif hasattr(inner, "coef_"):
        coef = np.asarray(inner.coef_)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        label = "Magnitude des coefficients (|w|)"

    if importances is None or len(importances) != len(feature_names) or len(feature_names) == 0:
        return None

    top_k = min(12, len(feature_names))
    idx = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="#3b82f6")
    ax.set_title(f"{model_name} — Top {top_k} variables influentes")
    ax.set_xlabel(label)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig