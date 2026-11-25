from __future__ import annotations

import io
import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    make_blobs,
    make_classification,
    make_regression,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


@dataclass
class DataHealthReport:
    """Rapport complet d'audit et de diagnostic automatique d'un DataFrame."""
    n_rows: int
    n_cols: int
    duplicate_rows: int
    constant_cols: list[str]
    all_nan_cols: list[str]
    high_cardinality_cols: list[str]
    id_like_cols: list[str]
    mixed_type_cols: list[str]
    datetime_candidates: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    boolean_cols: list[str]
    missing_by_col: dict[str, int]
    missing_pct_by_col: dict[str, float]
    outliers_by_col: dict[str, int]
    skewness_by_col: dict[str, float]
    warnings: list[str]


@dataclass
class PreprocessingConfig:
    """Configuration complète du pipeline de nettoyage et préparation."""
    drop_duplicates: bool = True
    drop_constant_cols: bool = True
    drop_all_nan_cols: bool = True
    drop_id_cols: bool = False
    drop_high_missing_threshold: float = 0.70
    selected_features: list[str] | None = None
    target_col: str | None = None
    drop_missing_target: bool = True
    numeric_imputation: Literal["median", "mean", "knn", "zero", "drop_rows"] = "median"
    categorical_imputation: Literal["most_frequent", "constant_unknown", "drop_rows"] = "most_frequent"
    outlier_strategy: Literal["none", "clip_iqr", "clip_zscore", "remove_iqr"] = "none"
    outlier_factor: float = 1.5
    encoding_strategy: Literal["onehot", "ordinal"] = "onehot"
    max_onehot_cardinality: int = 20
    scaling_strategy: Literal["standard", "minmax", "robust", "none"] = "standard"
    extract_datetime_features: bool = True
    force_numeric_coercion: bool = True


MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "null", "none", "nil", "undefined",
    "?", "-", "--", "---", "missing", "inconnu", "vide", "#n/a", "#value!",
    "inf", "-inf", "infinity", "-infinity"
}


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for idx, col in enumerate(df.columns):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"colonne_{idx + 1}"
        name = re.sub(r"\s+", "_", name)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cleaned.append(name)
    df.columns = cleaned
    return df


def normalize_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise un DataFrame brut : noms de colonnes, espaces superflus, valeurs sentinelles et infinis."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = _standardize_column_names(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        mask_missing = s.str.lower().isin(MISSING_TOKENS)
        s = s.mask(mask_missing, pd.NA)
        df[col] = s.astype(object).where(s.notna(), np.nan)

    return df


def _try_coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Convertit une colonne texte contenant des nombres (virgules, devises, %) en numérique."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s
    non_null = s.dropna().astype(str)
    if non_null.empty:
        return s

    cleaned = (
        non_null.str.replace(r"[€$£%\s]", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    converted = pd.to_numeric(cleaned, errors="coerce")
    if float(converted.notna().mean()) >= 0.85:
        full_cleaned = (
            s.dropna()
            .astype(str)
            .str.replace(r"[€$£%\s]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[full_cleaned.index] = pd.to_numeric(full_cleaned, errors="coerce")
        return out
    return s


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    dt_cols: list[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_cols.append(col)
            continue
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50)
            if sample.empty:
                continue
            if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.80:
                dt_cols.append(col)
    return dt_cols


def load_dataset_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Charge automatiquement CSV (détection séparateur/encodage), Excel, JSON, Parquet ou TSV."""
    lower = filename.lower()
    buf = io.BytesIO(file_bytes)

    if lower.endswith((".xlsx", ".xls")):
        return normalize_raw_dataframe(pd.read_excel(buf))

    if lower.endswith(".parquet"):
        return normalize_raw_dataframe(pd.read_parquet(buf))

    if lower.endswith(".json"):
        raw_text = file_bytes.decode("utf-8", errors="replace").strip()
        try:
            df = pd.read_json(io.StringIO(raw_text))
        except ValueError:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        df = pd.DataFrame(v)
                        break
                else:
                    df = pd.json_normalize(parsed)
            else:
                df = pd.json_normalize(parsed)
        return normalize_raw_dataframe(df)

    decoded = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            decoded = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        decoded = file_bytes.decode("utf-8", errors="replace")

    sep = "\t" if lower.endswith(".tsv") else None
    try:
        df = pd.read_csv(io.StringIO(decoded), sep=sep, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(io.StringIO(decoded), sep=";", on_bad_lines="skip")

    return normalize_raw_dataframe(df)


def parse_pasted_data(raw_text: str) -> pd.DataFrame:
    """Parse des données collées depuis Excel, CSV, TSV ou JSON."""
    text = raw_text.strip()
    if not text:
        raise ValueError("Le texte saisi est vide.")
    if text.startswith(("{", "[")):
        return load_dataset_from_bytes(text.encode("utf-8"), "pasted.json")
    return load_dataset_from_bytes(text.encode("utf-8"), "pasted.csv")


def extract_from_url(url: str, timeout: int = 15) -> pd.DataFrame:
    """Extrait un jeu de données depuis une URL directe (GitHub raw, OpenML, API JSON/CSV, etc.)."""
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        raise ValueError("L'URL doit commencer par http:// ou https://")

    if "github.com" in clean_url and "/blob/" in clean_url:
        clean_url = clean_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    req = urllib.request.Request(
        clean_url,
        headers={"User-Agent": "Mozilla/5.0 (DataScienceStudio/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read()
        content_type = resp.headers.get("Content-Type", "").lower()

    if "json" in content_type or clean_url.lower().endswith(".json"):
        fname = "remote.json"
    elif "excel" in content_type or clean_url.lower().endswith((".xlsx", ".xls")):
        fname = "remote.xlsx"
    elif clean_url.lower().endswith(".parquet"):
        fname = "remote.parquet"
    else:
        fname = "remote.csv"

    return load_dataset_from_bytes(content, fname)


def generate_synthetic_dataset(
    preset: str = "iris",
    n_samples: int = 300,
    n_features: int = 5,
    noise_level: float = 0.1,
    inject_dirty_issues: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Génère ou charge un jeu de données prêt à l'emploi (avec option d'injection d'anomalies)."""
    rng = np.random.default_rng(random_state)

    if preset == "iris":
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        target_names = dict(enumerate(data.target_names))
        df["espece"] = df["target"].map(target_names)
        df = df.drop(columns=["target"])
    elif preset == "wine":
        df = load_wine(as_frame=True).frame.copy()
    elif preset == "breast_cancer":
        df = load_breast_cancer(as_frame=True).frame.copy()
    elif preset == "diabetes":
        df = load_diabetes(as_frame=True).frame.copy()
    elif preset == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=max(2, n_features),
            n_informative=max(2, min(n_features, int(n_features * 0.7) + 1)),
            n_redundant=0,
            random_state=random_state,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i + 1}" for i in range(X.shape[1])])
        df["categorie_client"] = rng.choice(["Standard", "Premium", "VIP"], size=n_samples)
        df["target"] = y
    elif preset == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=max(2, n_features),
            noise=noise_level * 25.0,
            random_state=random_state,
        )
        df = pd.DataFrame(X, columns=[f"variable_{i + 1}" for i in range(X.shape[1])])
        df["region"] = rng.choice(["Nord", "Sud", "Est", "Ouest"], size=n_samples)
        df["cible_continue"] = y
    elif preset == "clustering":
        X, _ = make_blobs(
            n_samples=n_samples,
            centers=4,
            n_features=max(2, n_features),
            cluster_std=1.2,
            random_state=random_state,
        )
        df = pd.DataFrame(X, columns=[f"axe_{i + 1}" for i in range(X.shape[1])])
    else:
        raise ValueError(f"Preset inconnu : {preset}")

    if inject_dirty_issues and len(df) >= 10:
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in df.columns[: min(4, len(df.columns))]:
            idx = rng.choice(df.index, size=max(1, int(0.10 * len(df))), replace=False)
            df.loc[idx, col] = np.nan
        if num_cols:
            out_idx = rng.choice(df.index, size=min(3, len(df)), replace=False)
            df.loc[out_idx, num_cols[0]] = df[num_cols[0]].max() * 25.0 + 999.0
        df["colonne_constante"] = "INCHANGE"
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    return normalize_raw_dataframe(df)


def audit_dataframe(df: pd.DataFrame) -> DataHealthReport:
    """Analyse un DataFrame pour détecter toutes les anomalies de qualité."""
    df_work = normalize_raw_dataframe(df)
    n_rows, n_cols = df_work.shape

    duplicate_rows = int(df_work.duplicated().sum())
    all_nan_cols = [c for c in df_work.columns if df_work[c].isna().all()]
    constant_cols = [
        c for c in df_work.columns
        if c not in all_nan_cols and df_work[c].nunique(dropna=True) <= 1
    ]

    mixed_type_cols: list[str] = []
    coerced_df = df_work.copy()
    for col in df_work.columns:
        if df_work[col].dtype == object:
            s_non_null = df_work[col].dropna().astype(str)
            if not s_non_null.empty:
                num_ratio = pd.to_numeric(
                    s_non_null.str.replace(",", ".", regex=False), errors="coerce"
                ).notna().mean()
                if 0.15 <= num_ratio < 0.99:
                    mixed_type_cols.append(col)
                coerced_df[col] = _try_coerce_numeric_series(df_work[col])

    datetime_candidates = _detect_datetime_columns(coerced_df)
    boolean_cols = coerced_df.select_dtypes(include=["bool"]).columns.tolist()
    numeric_cols = [
        c for c in coerced_df.select_dtypes(include=[np.number]).columns.tolist()
        if c not in boolean_cols
    ]
    categorical_cols = [
        c for c in coerced_df.columns
        if c not in numeric_cols and c not in boolean_cols and c not in datetime_candidates
    ]

    high_cardinality_cols = [
        c for c in categorical_cols if coerced_df[c].nunique(dropna=True) > 30
    ]
    id_like_cols = [
        c for c in coerced_df.columns
        if (
            re.search(r"(^id$|_id$|^uuid$|^index$|^code$)", c.lower())
            and coerced_df[c].nunique(dropna=True) >= 0.90 * max(1, n_rows)
        )
    ]

    missing_by_col = {c: int(df_work[c].isna().sum()) for c in df_work.columns}
    missing_pct_by_col = {
        c: round((missing_by_col[c] / max(1, n_rows)) * 100.0, 2) for c in df_work.columns
    }

    outliers_by_col: dict[str, int] = {}
    skewness_by_col: dict[str, float] = {}
    for col in numeric_cols:
        s = coerced_df[col].dropna()
        if len(s) >= 4:
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            outliers = int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()) if iqr > 0 else 0
            outliers_by_col[col] = outliers
            skew_val = float(s.skew())
            skewness_by_col[col] = 0.0 if np.isnan(skew_val) else round(skew_val, 3)
        else:
            outliers_by_col[col] = 0
            skewness_by_col[col] = 0.0

    warnings: list[str] = []
    if n_rows < 15:
        warnings.append(f"Jeu de données très petit ({n_rows} lignes) : risque élevé de surapprentissage.")
    if duplicate_rows > 0:
        warnings.append(f"{duplicate_rows} ligne(s) dupliquée(s) détectée(s).")
    if all_nan_cols:
        warnings.append(f"Colonnes 100% vides : {', '.join(all_nan_cols)}.")
    if constant_cols:
        warnings.append(f"Colonnes constantes (variance nulle) : {', '.join(constant_cols)}.")
    if mixed_type_cols:
        warnings.append(f"Colonnes à types mixtes (texte + nombres) : {', '.join(mixed_type_cols)}.")
    total_missing = sum(missing_by_col.values())
    if total_missing > 0:
        warnings.append(f"{total_missing} valeur(s) manquante(s) sur {sum(v > 0 for v in missing_by_col.values())} colonne(s).")
    total_outliers = sum(outliers_by_col.values())
    if total_outliers > 0:
        warnings.append(f"{total_outliers} valeur(s) aberrante(s) (critère IQR) détectée(s).")
    if high_cardinality_cols:
        warnings.append(f"Colonnes catégorielles à haute cardinalité (>30 modalités) : {', '.join(high_cardinality_cols)}.")
    if id_like_cols:
        warnings.append(f"Colonnes identifiants probables : {', '.join(id_like_cols)}.")

    return DataHealthReport(
        n_rows=n_rows,
        n_cols=n_cols,
        duplicate_rows=duplicate_rows,
        constant_cols=constant_cols,
        all_nan_cols=all_nan_cols,
        high_cardinality_cols=high_cardinality_cols,
        id_like_cols=id_like_cols,
        mixed_type_cols=mixed_type_cols,
        datetime_candidates=datetime_candidates,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        boolean_cols=boolean_cols,
        missing_by_col=missing_by_col,
        missing_pct_by_col=missing_pct_by_col,
        outliers_by_col=outliers_by_col,
        skewness_by_col=skewness_by_col,
        warnings=warnings,
    )


def clean_and_prepare_data(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> dict[str, Any]:
    """Pipeline complet de nettoyage, imputation, encodage et normalisation."""
    cfg = config or PreprocessingConfig()
    logs: list[str] = []

    if df is None or df.empty:
        raise ValueError("Le jeu de données fourni est vide.")

    df_clean = normalize_raw_dataframe(df)
    initial_shape = df_clean.shape

    if cfg.drop_duplicates:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        removed = before - len(df_clean)
        if removed > 0:
            logs.append(f"Doublons supprimés : {removed} ligne(s).")

    target_col = cfg.target_col if (cfg.target_col and cfg.target_col in df_clean.columns) else None
    y_series: pd.Series | None = None
    if target_col is not None:
        if cfg.drop_missing_target:
            before = len(df_clean)
            df_clean = df_clean[df_clean[target_col].notna()].reset_index(drop=True)
            removed = before - len(df_clean)
            if removed > 0:
                logs.append(f"Lignes avec cible '{target_col}' manquante supprimées : {removed}.")
        y_series = df_clean[target_col].copy()
        X_df = df_clean.drop(columns=[target_col]).copy()
    else:
        X_df = df_clean.copy()

    if cfg.selected_features:
        valid_feats = [c for c in cfg.selected_features if c in X_df.columns]
        if valid_feats:
            X_df = X_df[valid_feats].copy()
            logs.append(f"Sélection appliquée : {len(valid_feats)} variable(s) conservée(s).")

    if cfg.drop_all_nan_cols:
        nan_cols = [c for c in X_df.columns if X_df[c].isna().all()]
        if nan_cols:
            X_df = X_df.drop(columns=nan_cols)
            logs.append(f"Colonnes 100% vides supprimées : {', '.join(nan_cols)}.")

    if 0.0 < cfg.drop_high_missing_threshold < 1.0:
        high_nan = [
            c for c in X_df.columns
            if float(X_df[c].isna().mean()) > cfg.drop_high_missing_threshold
        ]
        if high_nan:
            X_df = X_df.drop(columns=high_nan)
            logs.append(f"Colonnes > {int(cfg.drop_high_missing_threshold * 100)}% manquantes supprimées : {', '.join(high_nan)}.")

    if cfg.drop_id_cols:
        id_cols = [
            c for c in X_df.columns
            if re.search(r"(^id$|_id$|^uuid$|^index$|^code$)", c.lower())
            and X_df[c].nunique(dropna=True) >= 0.90 * max(1, len(X_df))
        ]
        if id_cols:
            X_df = X_df.drop(columns=id_cols)
            logs.append(f"Colonnes identifiants supprimées : {', '.join(id_cols)}.")

    if cfg.force_numeric_coercion:
        for col in X_df.columns:
            if X_df[col].dtype == object:
                coerced = _try_coerce_numeric_series(X_df[col])
                if pd.api.types.is_numeric_dtype(coerced):
                    X_df[col] = coerced
                    logs.append(f"Conversion numérique réussie pour '{col}'.")

    if cfg.extract_datetime_features:
        dt_cols = _detect_datetime_columns(X_df)
        for col in dt_cols:
            parsed_dt = pd.to_datetime(X_df[col], errors="coerce", format="mixed")
            X_df[f"{col}_annee"] = parsed_dt.dt.year
            X_df[f"{col}_mois"] = parsed_dt.dt.month
            X_df[f"{col}_jour"] = parsed_dt.dt.day
            X_df[f"{col}_jour_semaine"] = parsed_dt.dt.dayofweek
            X_df = X_df.drop(columns=[col])
            logs.append(f"Colonne date '{col}' décomposée en année/mois/jour/jour_semaine.")

    for col in X_df.select_dtypes(include=["bool"]).columns:
        X_df[col] = X_df[col].astype(int)

    if cfg.drop_constant_cols:
        const_cols = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 1]
        if const_cols:
            X_df = X_df.drop(columns=const_cols)
            logs.append(f"Colonnes constantes supprimées : {', '.join(const_cols)}.")

    if X_df.shape[1] == 0:
        raise ValueError("Aucune colonne exploitable ne reste après le filtrage.")

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    if cfg.numeric_imputation == "drop_rows" or cfg.categorical_imputation == "drop_rows":
        keep_mask = pd.Series(True, index=X_df.index)
        if cfg.numeric_imputation == "drop_rows" and num_cols:
            keep_mask &= X_df[num_cols].notna().all(axis=1)
        if cfg.categorical_imputation == "drop_rows" and cat_cols:
            keep_mask &= X_df[cat_cols].notna().all(axis=1)
        dropped = int((~keep_mask).sum())
        if dropped > 0:
            X_df = X_df.loc[keep_mask].reset_index(drop=True)
            if y_series is not None:
                y_series = y_series.loc[keep_mask].reset_index(drop=True)
            logs.append(f"Lignes incomplètes supprimées : {dropped}.")

    if num_cols and X_df[num_cols].isna().any().any():
        if cfg.numeric_imputation == "knn" and len(X_df) >= 6:
            X_df[num_cols] = KNNImputer(n_neighbors=min(5, len(X_df) - 1)).fit_transform(X_df[num_cols])
            logs.append("Imputation numérique KNN appliquée.")
        elif cfg.numeric_imputation == "zero":
            X_df[num_cols] = X_df[num_cols].fillna(0.0)
            logs.append("Imputation numérique par 0 appliquée.")
        else:
            strategy = "mean" if cfg.numeric_imputation == "mean" else "median"
            X_df[num_cols] = SimpleImputer(strategy=strategy).fit_transform(X_df[num_cols])
            logs.append(f"Imputation numérique ({strategy}) appliquée.")

    if cat_cols and X_df[cat_cols].isna().any().any():
        if cfg.categorical_imputation == "constant_unknown":
            X_df[cat_cols] = X_df[cat_cols].fillna("Inconnu")
            logs.append("Imputation catégorielle par 'Inconnu' appliquée.")
        else:
            X_df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X_df[cat_cols].astype(str))
            logs.append("Imputation catégorielle (mode) appliquée.")

    if num_cols and cfg.outlier_strategy != "none":
        if cfg.outlier_strategy == "clip_iqr":
            for col in num_cols:
                q1, q3 = float(X_df[col].quantile(0.25)), float(X_df[col].quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    X_df[col] = X_df[col].clip(lower=q1 - cfg.outlier_factor * iqr, upper=q3 + cfg.outlier_factor * iqr)
            logs.append(f"Winsorisation IQR (facteur {cfg.outlier_factor}) appliquée.")
        elif cfg.outlier_strategy == "clip_zscore":
            for col in num_cols:
                mean, std = float(X_df[col].mean()), float(X_df[col].std())
                if std > 0:
                    z = max(2.0, cfg.outlier_factor * 2.0)
                    X_df[col] = X_df[col].clip(lower=mean - z * std, upper=mean + z * std)
            logs.append("Écrêtage Z-score appliqué.")
        elif cfg.outlier_strategy == "remove_iqr":
            mask = pd.Series(True, index=X_df.index)
            for col in num_cols:
                q1, q3 = float(X_df[col].quantile(0.25)), float(X_df[col].quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    mask &= (X_df[col] >= q1 - cfg.outlier_factor * iqr) & (X_df[col] <= q3 + cfg.outlier_factor * iqr)
            removed_out = int((~mask).sum())
            if removed_out > 0 and mask.sum() >= 5:
                X_df = X_df.loc[mask].reset_index(drop=True)
                if y_series is not None:
                    y_series = y_series.loc[mask].reset_index(drop=True)
                logs.append(f"Lignes aberrantes (IQR) supprimées : {removed_out}.")

    encoders: dict[str, Any] = {}
    if cat_cols:
        for col in cat_cols:
            X_df[col] = X_df[col].astype(str)

        if cfg.encoding_strategy == "onehot":
            for col in cat_cols:
                top_cats = X_df[col].value_counts().nlargest(cfg.max_onehot_cardinality).index
                if X_df[col].nunique() > cfg.max_onehot_cardinality:
                    X_df[col] = np.where(X_df[col].isin(top_cats), X_df[col], "Autre")
                    logs.append(f"Modalités rares regroupées sous 'Autre' pour '{col}'.")

            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
            encoded_arr = ohe.fit_transform(X_df[cat_cols])
            df_encoded = pd.DataFrame(encoded_arr, columns=ohe.get_feature_names_out(cat_cols).tolist(), index=X_df.index)
            X_df = pd.concat([X_df.drop(columns=cat_cols), df_encoded], axis=1)
            encoders["onehot"] = ohe
            logs.append(f"Encodage One-Hot appliqué sur {len(cat_cols)} variable(s).")
        else:
            ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_df[cat_cols] = ord_enc.fit_transform(X_df[cat_cols])
            encoders["ordinal"] = ord_enc
            logs.append(f"Encodage Ordinal appliqué sur {len(cat_cols)} variable(s).")

    post_const = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 1]
    if post_const and len(post_const) < X_df.shape[1]:
        X_df = X_df.drop(columns=post_const)

    scaler = None
    X_processed = X_df.copy()
    if cfg.scaling_strategy != "none" and X_processed.shape[1] > 0:
        scaler = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[cfg.scaling_strategy]()
        X_processed[X_processed.columns] = scaler.fit_transform(X_processed)
        logs.append(f"Normalisation ({cfg.scaling_strategy}) appliquée sur {X_processed.shape[1]} variable(s).")

    target_encoder: LabelEncoder | None = None
    target_classes: list[str] | None = None
    y_clean: pd.Series | None = None

    if y_series is not None:
        y_coerced = _try_coerce_numeric_series(y_series)
        if pd.api.types.is_bool_dtype(y_coerced):
            y_clean = y_coerced.astype(int)
            target_classes = ["False", "True"]
        elif pd.api.types.is_numeric_dtype(y_coerced):
            y_clean = y_coerced
        else:
            target_encoder = LabelEncoder()
            y_encoded_arr = target_encoder.fit_transform(y_coerced.astype(str))
            y_clean = pd.Series(y_encoded_arr, name=target_col, index=X_processed.index)
            target_classes = [str(c) for c in target_encoder.classes_]
            logs.append(f"Cible catégorielle '{target_col}' encodée ({len(target_classes)} classes).")

    df_clean_export = X_processed.copy()
    if y_clean is not None and target_col is not None:
        df_clean_export[target_col] = y_clean.values

    return {
        "X": X_processed,
        "X_unscaled": X_df,
        "y": y_clean,
        "df_clean": df_clean_export,
        "target_col": target_col,
        "target_encoder": target_encoder,
        "target_classes": target_classes,
        "scaler": scaler,
        "encoders": encoders,
        "initial_shape": initial_shape,
        "final_shape": df_clean_export.shape,
        "logs": logs,
    }