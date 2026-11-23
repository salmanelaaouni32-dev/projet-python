import pandas as pd
from sklearn.datasets import load_iris, make_blobs, make_regression

from export.exporter import (
    create_run_folder,
    export_evaluation_report,
    export_figures,
    export_model,
    export_predictions_table,
)
from models.unsupervised_nn import NeuralNetworkModel, UnsupervisedKMeansModel

# 1. Test K-Means
X_blobs, _ = make_blobs(n_samples=200, centers=3, n_features=4, random_state=42)
df_blobs = pd.DataFrame(X_blobs, columns=[f"feat_{i}" for i in range(4)])

km_out = UnsupervisedKMeansModel(n_clusters=3, random_state=42).run(df_blobs, model_name="KMeans")
run_dir_km = create_run_folder(base_dir="outputs", prefix="kmeans")
df_clustered = df_blobs.copy()
df_clustered["cluster"] = km_out.labels

export_evaluation_report(km_out.evaluation, run_dir_km, filename_stem="kmeans_metrics")
export_predictions_table(df_clustered, run_dir_km, filename_stem="kmeans_clusters")
export_model(km_out.pipeline, run_dir_km, filename_stem="kmeans_pipeline")
export_figures(km_out.figures, run_dir_km, close_after=True)

# 2. Test Réseau de neurones en classification
iris = load_iris(as_frame=True)
nn_clf_out = NeuralNetworkModel(task_type="classification", hidden_layer_sizes=(32, 16), random_state=42).run(
    X=iris.data,
    y=iris.target,
    model_name="MLP_Classifier",
)
run_dir_nn = create_run_folder(base_dir="outputs", prefix="mlp_clf")
df_preds = pd.DataFrame({"y_true": nn_clf_out.y_test, "y_pred": nn_clf_out.y_pred})

export_evaluation_report(nn_clf_out.evaluation, run_dir_nn, filename_stem="mlp_clf_metrics")
export_predictions_table(df_preds, run_dir_nn, filename_stem="mlp_clf_predictions")
export_model(nn_clf_out.pipeline, run_dir_nn, filename_stem="mlp_clf_pipeline")
export_figures(nn_clf_out.figures, run_dir_nn, close_after=True)

# 3. Test Réseau de neurones en régression
X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=42)
df_reg = pd.DataFrame(X_reg, columns=[f"x_{i}" for i in range(5)])
nn_reg_out = NeuralNetworkModel(task_type="regression", hidden_layer_sizes=(64, 32), random_state=42).run(
    X=df_reg,
    y=y_reg,
    model_name="MLP_Regressor",
)
print("Régression R2 :", nn_reg_out.evaluation.metrics["r2"])