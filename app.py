from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from data.preprocessing import (
    PreprocessingConfig,
    audit_dataframe,
    clean_and_prepare_data,
    extract_from_url,
    generate_synthetic_dataset,
    load_dataset_from_bytes,
    parse_pasted_data,
)
from export.exporter import (
    create_run_folder,
    export_evaluation_report,
    export_figures,
    export_model,
    export_predictions_table,
)
from models.registry import (
    MODEL_CATALOG,
    UnifiedRunResult,
    execute_model,
    get_implementation_status,
)

# ---------------------------------------------------------------------------
# CONFIGURATION GLOBALE & DESIGN SYSTEM MODERNE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IDIA ML Studio — Plateforme Intelligente de Data Science",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
        padding: 1.6rem 2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.22);
    }
    .hero-banner h1 { margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em; }
    .hero-banner p { margin: 0.45rem 0 0 0; color: #dbeafe; font-size: 0.98rem; }
    .badge-pill {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(255,255,255,0.16);
        color: #fff;
        margin-right: 0.4rem;
        margin-top: 0.6rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# INITIALISATION DU SESSION STATE
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "raw_df": None,
        "dataset_name": "Aucun dataset chargé",
        "prepared_bundle": None,
        "run_history": [],
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _set_new_dataset(df: pd.DataFrame, name: str) -> None:
    if df is None or df.empty:
        st.error("Le jeu de données chargé est vide.")
        return
    st.session_state["raw_df"] = df
    st.session_state["dataset_name"] = name
    st.session_state["prepared_bundle"] = None
    st.session_state["last_result"] = None


# ---------------------------------------------------------------------------
# SIDEBAR : STATUT ÉQUIPE & ÉTAT DU PIPELINE
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Master IDIA — FST Tanger")
    st.caption("Programmation avancée pour la Data Science • 2026/2027")
    st.divider()

    st.markdown("#### 📊 État des Données")
    raw_df: pd.DataFrame | None = st.session_state["raw_df"]
    if raw_df is not None:
        st.success(f"**Source :** {st.session_state['dataset_name']}")
        c1, c2 = st.columns(2)
        c1.metric("Lignes", f"{raw_df.shape[0]:,}")
        c2.metric("Colonnes", raw_df.shape[1])
    else:
        st.info("Chargez un dataset dans l'onglet **1. Données**.")

    prep = st.session_state["prepared_bundle"]
    if prep is not None:
        st.success(f"✅ Données prêtes ({prep['final_shape'][0]} × {prep['final_shape'][1]})")
    elif raw_df is not None:
        st.warning("⚠️ Prétraitement non encore appliqué.")

    st.divider()
    st.markdown("#### 🧩 Intégration de l'Équipe")
    status_map = get_implementation_status()
    n_team = sum(1 for v in status_map.values() if v["implemented_in_team_file"])
    st.progress(n_team / len(status_map), text=f"{n_team}/{len(status_map)} modèles reliés aux fichiers d'équipe")

    with st.expander("Détails des modules (Zakariya, Wissal, Nour, Salmane)"):
        st.markdown("- **Zakariya (Tâche 1) :** `app.py` & `data/preprocessing.py` ✅")
        st.markdown("- **Salmane (Tâche 4) :** `unsupervised_nn.py`, `metrics_plots.py`, `exporter.py` ✅")
        for k, info in status_map.items():
            icon = "🟢" if info["implemented_in_team_file"] else "🟡"
            st.caption(f"{icon} **{MODEL_CATALOG[k].display_name}** : {info['source_label']}")


# ---------------------------------------------------------------------------
# EN-TÊTE HERO
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🧠 IDIA ML Studio — Plateforme Polyvalente d'Analyse & Machine Learning</h1>
        <p>Acquisition universelle • Diagnostic & Nettoyage automatique • Exploration visuelle • 11 Algorithmes ML & Deep Learning • Évaluation & Export</p>
        <span class="badge-pill">FST Tanger • Master IDIA</span>
        <span class="badge-pill">Auto-Healing Data Pipeline</span>
        <span class="badge-pill">Supervisé & Non Supervisé</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# NAVIGATION PAR ONGLETS
# ---------------------------------------------------------------------------
tab_data, tab_viz, tab_prep, tab_ml, tab_compare, tab_docs = st.tabs([
    "1️⃣ Acquisition & Diagnostic",
    "2️⃣ Exploration Visuelle",
    "3️⃣ Nettoyage & Préparation",
    "4️⃣ Entraînement & Évaluation ML",
    "5️⃣ Comparateur & Export",
    "6️⃣ Tutoriels & Architecture",
])


# ===========================================================================
# ONGLET 1 : ACQUISITION & DIAGNOSTIC QUALITÉ
# ===========================================================================
with tab_data:
    st.subheader("📥 Importer, Créer ou Extraire un Jeu de Données")
    src_mode = st.radio(
        "Choisissez la source de données :",
        [
            "📂 Importer un fichier (CSV, Excel, JSON, Parquet, TSV)",
            "🌐 Extraire depuis une URL (GitHub, OpenML, API)",
            "✍️ Créer / Coller manuellement des données",
            "🧪 Jeux de données intégrés & Générateur synthétique",
        ],
        horizontal=True,
    )

    if src_mode.startswith("📂"):
        uploaded = st.file_uploader(
            "Glissez-déposez votre fichier ici (détection automatique du séparateur et de l'encodage)",
            type=["csv", "xlsx", "xls", "json", "parquet", "tsv", "txt"],
        )
        if uploaded is not None:
            if st.button("Charger ce fichier", type="primary"):
                try:
                    df_loaded = load_dataset_from_bytes(uploaded.getvalue(), uploaded.name)
                    _set_new_dataset(df_loaded, uploaded.name)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Erreur lors de la lecture du fichier : {exc}")

    elif src_mode.startswith("🌐"):
        url_input = st.text_input(
            "URL directe vers un fichier CSV, JSON, Excel ou lien GitHub :",
            placeholder="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        )
        if st.button("Extraire les données depuis l'URL", type="primary"):
            try:
                with st.spinner("Téléchargement et analyse en cours..."):
                    df_url = extract_from_url(url_input)
                _set_new_dataset(df_url, url_input.split("/")[-1] or "Dataset URL")
                st.rerun()
            except Exception as exc:
                st.error(f"Impossible d'extraire depuis cette URL : {exc}")

    elif src_mode.startswith("✍️"):
        sub_create1, sub_create2 = st.tabs(["Coller du texte (CSV / TSV / JSON)", "Éditeur de tableau interactif"])
        with sub_create1:
            sample_csv = "age,salaire,experience,ville,achete\n25,3200,2,Tanger,Non\n34,5400,7,Rabat,Oui\n45,7800,15,Casablanca,Oui\n29,,4,Tanger,Non\n52,9100,22,Rabat,Oui\n23,2800,1,Fes,Non"
            pasted_text = st.text_area("Collez vos données brutes :", value=sample_csv, height=170)
            if st.button("Valider les données collées", type="primary"):
                try:
                    df_pasted = parse_pasted_data(pasted_text)
                    _set_new_dataset(df_pasted, "Données saisies manuellement")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Format non reconnu : {exc}")

        with sub_create2:
            base_edit = st.session_state["raw_df"] if st.session_state["raw_df"] is not None else pd.DataFrame(
                {"feature_1": [1.2, 3.4, 2.1, 5.6], "feature_2": [10.0, 20.5, 15.0, 30.2], "categorie": ["A", "B", "A", "B"], "cible": [0, 1, 0, 1]}
            )
            edited_df = st.data_editor(base_edit, num_rows="dynamic", use_container_width=True)
            if st.button("Enregistrer ce tableau comme dataset actif", type="primary"):
                _set_new_dataset(edited_df, "Tableau édité manuellement")
                st.rerun()

    else:
        col_p1, col_p2, col_p3, col_p4 = st.columns([2, 1, 1, 1.3])
        preset_choice = col_p1.selectbox(
            "Dataset de démonstration :",
            [
                ("iris", "🌸 Iris (Classification multiclasse)"),
                ("wine", "🍷 Wine Recognition (Classification)"),
                ("breast_cancer", "🩺 Breast Cancer Wisconsin (Classification binaire)"),
                ("diabetes", "📈 Diabetes (Régression)"),
                ("classification", "🎲 Synthétique — Classification"),
                ("regression", "🎲 Synthétique — Régression"),
                ("clustering", "🎲 Synthétique — Clustering K-Means"),
            ],
            format_func=lambda x: x[1],
        )
        n_samp = col_p2.number_input("Échantillons", min_value=30, max_value=5000, value=250, step=50)
        n_feat = col_p3.number_input("Variables", min_value=2, max_value=30, value=5)
        inject_dirty = col_p4.checkbox("Injecter des anomalies (NaN, outliers, doublons)", value=False)

        if st.button("Générer / Charger ce dataset", type="primary"):
            df_gen = generate_synthetic_dataset(
                preset=preset_choice[0],
                n_samples=int(n_samp),
                n_features=int(n_feat),
                inject_dirty_issues=inject_dirty,
            )
            _set_new_dataset(df_gen, f"Dataset ({preset_choice[0]})")
            st.rerun()

    # Diagnostic automatique si un dataset est chargé
    if st.session_state["raw_df"] is not None:
        df_curr: pd.DataFrame = st.session_state["raw_df"]
        st.divider()
        st.subheader("🩺 Audit Automatique & Santé des Données")
        report = audit_dataframe(df_curr)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Observations", report.n_rows)
        m2.metric("Variables", report.n_cols)
        m3.metric("Valeurs manquantes", sum(report.missing_by_col.values()))
        m4.metric("Lignes dupliquées", report.duplicate_rows)
        m5.metric("Outliers (IQR)", sum(report.outliers_by_col.values()))

        if report.warnings:
            for w in report.warnings:
                st.warning(f"⚠️ {w}")
        else:
            st.success("✅ Aucune anomalie structurelle majeure détectée sur ce jeu de données !")

        col_prev, col_meta = st.columns([1.4, 1])
        with col_prev:
            st.markdown("**Aperçu des données brutes :**")
            st.dataframe(df_curr.head(20), use_container_width=True)
        with col_meta:
            st.markdown("**Profil détaillé par colonne :**")
            meta_rows = []
            for c in df_curr.columns:
                meta_rows.append({
                    "Colonne": c,
                    "Type": str(df_curr[c].dtype),
                    "Manquants": f"{report.missing_by_col[c]} ({report.missing_pct_by_col[c]}%)",
                    "Uniques": int(df_curr[c].nunique(dropna=True)),
                    "Outliers": report.outliers_by_col.get(c, 0),
                })
            st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)


# ===========================================================================
# ONGLET 2 : EXPLORATION VISUELLE DES DONNÉES
# ===========================================================================
with tab_viz:
    st.subheader("📊 Représentation Graphique & Analyse Exploratoire")
    if st.session_state["raw_df"] is None:
        st.info("Veuillez d'abord charger un jeu de données dans l'onglet **1️⃣ Acquisition & Diagnostic**.")
    else:
        df_v: pd.DataFrame = (
            st.session_state["prepared_bundle"]["df_clean"]
            if st.session_state["prepared_bundle"] is not None
            and st.checkbox("Visualiser sur les données nettoyées (onglet 3)", value=False)
            else st.session_state["raw_df"]
        )

        num_cols = df_v.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df_v.columns if c not in num_cols]

        chart_type = st.selectbox(
            "Type de représentation graphique :",
            [
                "Distribution univariée (Histogramme + KDE / Boîte à moustaches)",
                "Relation bivariée (Nuage de points / Scatter Plot)",
                "Matrice de corrélation (Heatmap)",
                "Analyse des variables catégorielles (Diagramme en barres)",
                "Matrice de dispersion multi-variables (Pairplot)",
            ],
        )

        if chart_type.startswith("Distribution"):
            if not num_cols:
                st.warning("Aucune variable numérique disponible.")
            else:
                col_sel = st.selectbox("Variable numérique :", num_cols)
                hue_sel = st.selectbox("Colorer par groupe (optionnel) :", ["Aucun"] + cat_cols)
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
                sns.histplot(
                    data=df_v,
                    x=col_sel,
                    hue=None if hue_sel == "Aucun" else hue_sel,
                    kde=True,
                    ax=axes[0],
                    palette="viridis",
                )
                axes[0].set_title(f"Distribution de {col_sel}")
                sns.boxplot(
                    data=df_v,
                    x=None if hue_sel == "Aucun" else hue_sel,
                    y=col_sel,
                    ax=axes[1],
                    palette="Set2",
                )
                axes[1].set_title(f"Boîte à moustaches — {col_sel}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        elif chart_type.startswith("Relation"):
            if len(num_cols) < 2:
                st.warning("Au moins 2 variables numériques sont nécessaires.")
            else:
                c1, c2, c3 = st.columns(3)
                x_var = c1.selectbox("Axe X :", num_cols, index=0)
                y_var = c2.selectbox("Axe Y :", num_cols, index=min(1, len(num_cols) - 1))
                color_var = c3.selectbox("Couleur (optionnel) :", ["Aucun"] + df_v.columns.tolist())
                fig, ax = plt.subplots(figsize=(8.5, 4.8))
                sns.scatterplot(
                    data=df_v,
                    x=x_var,
                    y=y_var,
                    hue=None if color_var == "Aucun" else color_var,
                    alpha=0.82,
                    s=55,
                    ax=ax,
                )
                ax.set_title(f"{y_var} en fonction de {x_var}")
                ax.grid(True, linestyle="--", alpha=0.4)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        elif chart_type.startswith("Matrice de corrélation"):
            if len(num_cols) < 2:
                st.warning("Au moins 2 variables numériques sont nécessaires.")
            else:
                corr = df_v[num_cols].corr()
                fig, ax = plt.subplots(figsize=(9, 6))
                sns.heatmap(corr, annot=len(num_cols) <= 12, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
                ax.set_title("Matrice des corrélations (Pearson)")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        elif chart_type.startswith("Analyse des variables catégorielles"):
            if not cat_cols:
                st.info("Aucune variable catégorielle détectée dans ce dataset.")
            else:
                cat_sel = st.selectbox("Variable catégorielle :", cat_cols)
                top_counts = df_v[cat_sel].fillna("Manquant").value_counts().head(20)
                fig, ax = plt.subplots(figsize=(8.5, 4.2))
                sns.barplot(x=top_counts.values, y=top_counts.index.astype(str), palette="Blues_r", ax=ax)
                ax.set_title(f"Répartition des modalités — {cat_sel}")
                ax.set_xlabel("Effectif")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        else:
            if len(num_cols) < 2:
                st.warning("Au moins 2 variables numériques sont nécessaires.")
            else:
                chosen_cols = st.multiselect("Sélectionnez 2 à 5 variables :", num_cols, default=num_cols[: min(4, len(num_cols))])
                hue_p = st.selectbox("Variable de classe (optionnel) :", ["Aucun"] + cat_cols)
                if len(chosen_cols) >= 2:
                    plot_cols = chosen_cols + ([hue_p] if hue_p != "Aucun" else [])
                    g = sns.pairplot(df_v[plot_cols].dropna().head(400), hue=None if hue_p == "Aucun" else hue_p, corner=True)
                    st.pyplot(g.fig)
                    plt.close(g.fig)


# ===========================================================================
# ONGLET 3 : NETTOYAGE & PRÉPARATION DES DONNÉES
# ===========================================================================
with tab_prep:
    st.subheader("🧹 Pipeline de Nettoyage, Imputation & Normalisation")
    if st.session_state["raw_df"] is None:
        st.info("Chargez d'abord un jeu de données dans l'onglet **1️⃣ Acquisition & Diagnostic**.")
    else:
        df_raw: pd.DataFrame = st.session_state["raw_df"]
        all_cols = df_raw.columns.tolist()

        col_t1, col_t2 = st.columns([1.2, 1.8])
        target_choice = col_t1.selectbox(
            "🎯 Variable cible (y) — laisser 'Aucune' pour le clustering K-Means :",
            ["Aucune (Non supervisé)"] + all_cols,
            index=len(all_cols),
        )
        resolved_target = None if target_choice.startswith("Aucune") else target_choice
        candidate_features = [c for c in all_cols if c != resolved_target]
        selected_feats = col_t2.multiselect(
            "Variables explicatives (X) à conserver :",
            candidate_features,
            default=candidate_features,
        )

        with st.expander("⚙️ Paramètres avancés de nettoyage et transformation", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**1. Structure & Anomalies**")
                drop_dups = st.checkbox("Supprimer les doublons", value=True)
                drop_const = st.checkbox("Supprimer colonnes constantes", value=True)
                drop_nan_cols = st.checkbox("Supprimer colonnes 100% vides", value=True)
                drop_ids = st.checkbox("Supprimer colonnes ID détectées", value=True)
                coerce_num = st.checkbox("Convertir nombres texte (ex: 12,5 €)", value=True)
                extract_dt = st.checkbox("Extraire année/mois/jour des dates", value=True)

            with c2:
                st.markdown("**2. Valeurs Manquantes & Outliers**")
                num_imp = st.selectbox(
                    "Imputation numérique :",
                    ["median", "mean", "knn", "zero", "drop_rows"],
                    format_func=lambda x: {
                        "median": "Médiane (robuste)",
                        "mean": "Moyenne",
                        "knn": "KNN Imputer (plus proches voisins)",
                        "zero": "Remplacer par 0",
                        "drop_rows": "Supprimer les lignes incomplètes",
                    }[x],
                )
                cat_imp = st.selectbox(
                    "Imputation catégorielle :",
                    ["most_frequent", "constant_unknown", "drop_rows"],
                    format_func=lambda x: {
                        "most_frequent": "Valeur la plus fréquente (Mode)",
                        "constant_unknown": "Catégorie 'Inconnu'",
                        "drop_rows": "Supprimer les lignes incomplètes",
                    }[x],
                )
                out_strat = st.selectbox(
                    "Gestion des valeurs aberrantes :",
                    ["none", "clip_iqr", "clip_zscore", "remove_iqr"],
                    format_func=lambda x: {
                        "none": "Conserver telles quelles",
                        "clip_iqr": "Winsorisation (Écrêtage IQR)",
                        "clip_zscore": "Écrêtage Z-Score",
                        "remove_iqr": "Supprimer les lignes aberrantes (IQR)",
                    }[x],
                )

            with c3:
                st.markdown("**3. Encodage & Mise à l'échelle**")
                enc_strat = st.selectbox(
                    "Encodage catégoriel :",
                    ["onehot", "ordinal"],
                    format_func=lambda x: "One-Hot Encoding (recommandé)" if x == "onehot" else "Encodage Ordinal (0, 1, 2...)",
                )
                scale_strat = st.selectbox(
                    "Normalisation / Standardisation :",
                    ["standard", "minmax", "robust", "none"],
                    format_func=lambda x: {
                        "standard": "StandardScaler (Z-score : μ=0, σ=1)",
                        "minmax": "MinMaxScaler ([0, 1])",
                        "robust": "RobustScaler (Médiane & IQR)",
                        "none": "Aucune normalisation",
                    }[x],
                )

        if st.button("🚀 Exécuter le pipeline de préparation des données", type="primary", use_container_width=True):
            try:
                cfg = PreprocessingConfig(
                    drop_duplicates=drop_dups,
                    drop_constant_cols=drop_const,
                    drop_all_nan_cols=drop_nan_cols,
                    drop_id_cols=drop_ids,
                    selected_features=selected_feats,
                    target_col=resolved_target,
                    numeric_imputation=num_imp,  # type: ignore
                    categorical_imputation=cat_imp,  # type: ignore
                    outlier_strategy=out_strat,  # type: ignore
                    encoding_strategy=enc_strat,  # type: ignore
                    scaling_strategy=scale_strat,  # type: ignore
                    extract_datetime_features=extract_dt,
                    force_numeric_coercion=coerce_num,
                )
                bundle = clean_and_prepare_data(df_raw, cfg)
                st.session_state["prepared_bundle"] = bundle
                st.success("Pipeline exécuté avec succès ! Vos données sont prêtes pour le Machine Learning.")
            except Exception as exc:
                st.error(f"Erreur durant le prétraitement : {exc}")

        if st.session_state["prepared_bundle"] is not None:
            b = st.session_state["prepared_bundle"]
            st.divider()
            c_log, c_df = st.columns([1, 1.5])
            with c_log:
                st.markdown("**Journal des transformations appliquées :**")
                if b["logs"]:
                    for item in b["logs"]:
                        st.markdown(f"- ✅ {item}")
                else:
                    st.info("Aucune modification nécessaire : les données étaient déjà propres.")
                st.caption(f"Dimensions : {b['initial_shape']} ➡️ {b['final_shape']}")
            with c_df:
                st.markdown("**Jeu de données transformé prêt pour les modèles :**")
                st.dataframe(b["df_clean"].head(15), use_container_width=True)


# ===========================================================================
# ONGLET 4 : ENTRAÎNEMENT & ÉVALUATION DES MODÈLES ML
# ===========================================================================
with tab_ml:
    st.subheader("🤖 Modélisation, Validation & Visualisation des Résultats")

    if st.session_state["raw_df"] is None:
        st.info("Chargez d'abord un jeu de données dans l'onglet **1️⃣ Acquisition & Diagnostic**.")
    else:
        # Auto-préparation par défaut si l'utilisateur va directement sur l'onglet ML
        if st.session_state["prepared_bundle"] is None:
            st.info("💡 Prétraitement automatique par défaut activé (personnalisable dans l'onglet **3️⃣ Nettoyage & Préparation**).")
            default_target = st.session_state["raw_df"].columns[-1]
            st.session_state["prepared_bundle"] = clean_and_prepare_data(
                st.session_state["raw_df"],
                PreprocessingConfig(target_col=default_target),
            )

        bundle = st.session_state["prepared_bundle"]
        X_ready: pd.DataFrame = bundle["X"]
        y_ready: pd.Series | None = bundle["y"]

        # Détection intelligente du type de problème recommandé
        if y_ready is None:
            recommended_task = "clustering"
        elif bundle["target_classes"] is not None or y_ready.nunique() <= 15:
            recommended_task = "classification"
        else:
            recommended_task = "regression"

        col_m1, col_m2 = st.columns([1.1, 1.4])
        with col_m1:
            task_filter = st.selectbox(
                "Famille d'apprentissage :",
                ["classification", "regression", "clustering"],
                index=["classification", "regression", "clustering"].index(recommended_task),
                format_func=lambda x: {
                    "classification": "🎯 Classification Supervisée",
                    "regression": "📈 Régression Supervisée",
                    "clustering": "🔮 Clustering Non Supervisé (K-Means)",
                }[x],
            )

        available_models = [k for k, spec in MODEL_CATALOG.items() if spec.task_type == task_filter]
        with col_m2:
            selected_model_key = st.selectbox(
                "Algorithme de Machine Learning :",
                available_models,
                format_func=lambda k: MODEL_CATALOG[k].display_name,
            )

        spec = MODEL_CATALOG[selected_model_key]
        impl_info = get_implementation_status()[selected_model_key]

        st.info(f"**{spec.display_name}** — {spec.description}  \n*Statut module :* `{impl_info['source_label']}`")

        # Hyperparamètres dynamiques
        st.markdown("#### 🎛️ Réglage des Hyperparamètres & Validation")
        hp_values: dict[str, Any] = {}
        hp_cols = st.columns(max(1, len(spec.hyperparams_schema)))
        for idx, (hp_name, hp_cfg) in enumerate(spec.hyperparams_schema.items()):
            with hp_cols[idx % len(hp_cols)]:
                if hp_cfg["type"] == "bool":
                    hp_values[hp_name] = st.checkbox(hp_cfg["label"], value=hp_cfg["default"], key=f"hp_{selected_model_key}_{hp_name}")
                elif hp_cfg["type"] == "int":
                    hp_values[hp_name] = st.slider(
                        hp_cfg["label"],
                        min_value=int(hp_cfg["min"]),
                        max_value=int(hp_cfg["max"]),
                        value=int(hp_cfg["default"]),
                        key=f"hp_{selected_model_key}_{hp_name}",
                    )
                elif hp_cfg["type"] == "float":
                    hp_values[hp_name] = st.number_input(
                        hp_cfg["label"],
                        min_value=float(hp_cfg["min"]),
                        max_value=float(hp_cfg["max"]),
                        value=float(hp_cfg["default"]),
                        format="%.4f",
                        key=f"hp_{selected_model_key}_{hp_name}",
                    )
                elif hp_cfg["type"] == "select":
                    opts = hp_cfg["options"]
                    hp_values[hp_name] = st.selectbox(
                        hp_cfg["label"],
                        opts,
                        index=opts.index(hp_cfg["default"]),
                        key=f"hp_{selected_model_key}_{hp_name}",
                    )

        val_c1, val_c2, val_c3 = st.columns(3)
        test_ratio = val_c1.slider("Taille du jeu de test (%)", 10, 40, 20, step=5) / 100.0
        cv_k = val_c2.selectbox("Validation croisée (K-Fold)", [0, 3, 5, 10], format_func=lambda k: "Désactivée" if k == 0 else f"{k}-Fold Cross-Validation")
        seed = val_c3.number_input("Graine aléatoire (Random State)", min_value=0, max_value=9999, value=42)

        if st.button(f"⚡ Entraîner et Évaluer ({spec.display_name})", type="primary", use_container_width=True):
            try:
                with st.spinner("Entraînement du modèle et génération des graphiques..."):
                    run_res = execute_model(
                        model_key=selected_model_key,
                        X=X_ready,
                        y=y_ready,
                        params=hp_values,
                        test_size=test_ratio,
                        cv_folds=int(cv_k),
                        random_state=int(seed),
                        target_classes=bundle["target_classes"],
                    )
                st.session_state["last_result"] = run_res
                st.session_state["run_history"].append(run_res)
                st.success(f"Entraînement terminé via `{run_res.source_backend}` !")
            except Exception as exc:
                st.error(f"Impossible d'exécuter ce modèle sur la configuration actuelle : {exc}")

        # Affichage des résultats du dernier modèle exécuté
        last_res: UnifiedRunResult | None = st.session_state["last_result"]
        if last_res is not None:
            st.divider()
            st.markdown(f"### 📈 Résultats : {last_res.model_name}")

            metrics_dict = last_res.evaluation.metrics
            m_cols = st.columns(len(metrics_dict))
            for idx, (m_name, m_val) in enumerate(metrics_dict.items()):
                m_cols[idx].metric(m_name.upper(), f"{m_val:.4f}")

            if last_res.cv_scores:
                st.info(
                    f"🔁 **Validation croisée ({len(last_res.cv_scores)}-Fold) :** "
                    f"Moyenne = `{np.mean(last_res.cv_scores):.4f}` (± `{np.std(last_res.cv_scores):.4f}`) • Scores : {last_res.cv_scores}"
                )

            if last_res.figures:
                fig_cols = st.columns(len(last_res.figures))
                for idx, (fig_name, fig_obj) in enumerate(last_res.figures.items()):
                    with fig_cols[idx]:
                        st.pyplot(fig_obj)

            for tbl_title, tbl_df in last_res.extra_tables.items():
                st.markdown(f"**{tbl_title} :**")
                st.dataframe(tbl_df, use_container_width=True)

            st.markdown("**Aperçu des prédictions / affectations :**")
            st.dataframe(last_res.predictions_df.head(20), use_container_width=True)


# ===========================================================================
# ONGLET 5 : COMPARATEUR MULTI-MODÈLES & EXPORTATION
# ===========================================================================
with tab_compare:
    st.subheader("🏆 Comparateur de Performances & Exportation Complète")
    history: list[UnifiedRunResult] = st.session_state["run_history"]

    if not history:
        st.info("Entraînez au moins un modèle dans l'onglet **4️⃣ Entraînement & Évaluation ML** pour comparer et exporter.")
    else:
        comp_rows = []
        for idx, item in enumerate(history, start=1):
            comp_rows.append({
                "Run #": idx,
                "Modèle": item.model_name,
                "Tâche": item.task_type,
                "Backend": item.source_backend,
                **{k: round(v, 4) for k, v in item.evaluation.metrics.items()},
            })
        df_comp = pd.DataFrame(comp_rows)
        st.dataframe(df_comp, use_container_width=True)

        if st.button("🗑️ Réinitialiser l'historique des comparaisons"):
            st.session_state["run_history"] = []
            st.session_state["last_result"] = None
            st.rerun()

        st.divider()
        st.markdown("### 💾 Exporter un Modèle, ses Métriques et ses Graphiques")
        sel_idx = st.selectbox(
            "Sélectionnez l'exécution à exporter :",
            range(len(history)),
            index=len(history) - 1,
            format_func=lambda i: f"Run #{i + 1} — {history[i].model_name} ({history[i].task_type})",
        )
        chosen_run = history[sel_idx]

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            if st.button("📁 Sauvegarder dans le dossier local `outputs/`", type="primary"):
                run_dir = create_run_folder(base_dir="outputs", prefix=chosen_run.model_key)
                export_evaluation_report(chosen_run.evaluation, run_dir, filename_stem=f"{chosen_run.model_key}_metrics")
                export_predictions_table(chosen_run.predictions_df, run_dir, filename_stem=f"{chosen_run.model_key}_predictions")
                export_model(chosen_run.pipeline, run_dir, filename_stem=f"{chosen_run.model_key}_pipeline")
                export_figures(chosen_run.figures, run_dir, close_after=False)
                st.success(f"Tous les fichiers ont été exportés dans : `{run_dir}`")

        with c_exp2:
            # Génération d'une archive ZIP téléchargeable directement depuis le navigateur
            zip_buf = io.BytesIO()
            tmp_dir = Path("/tmp/idiastudio_export")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            rep_paths = export_evaluation_report(chosen_run.evaluation, tmp_dir, filename_stem="metrics")
            pred_paths = export_predictions_table(chosen_run.predictions_df, tmp_dir, filename_stem="predictions", export_excel=True)
            mod_path = export_model(chosen_run.pipeline, tmp_dir, filename_stem="model_pipeline")
            fig_paths = export_figures(chosen_run.figures, tmp_dir, close_after=False)

            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in [*rep_paths.values(), *pred_paths.values(), mod_path, *fig_paths.values()]:
                    zf.write(p, arcname=p.name)

            st.download_button(
                label="⬇️ Télécharger le pack complet (.ZIP)",
                data=zip_buf.getvalue(),
                file_name=f"export_{chosen_run.model_key}.zip",
                mime="application/zip",
                use_container_width=True,
            )


# ===========================================================================
# ONGLET 6 : DOCUMENTATION & TUTORIELS INTERACTIFS
# ===========================================================================
with tab_docs:
    st.subheader("📚 Guide d'Utilisation & Tutoriels des Algorithmes")
    st.markdown(
        """
        ### 🧭 Parcours recommandé en 5 étapes
        1. **Acquisition & Diagnostic :** Importez un fichier (`CSV`, `Excel`, `JSON`, `Parquet`), collez un tableau, extrayez depuis une URL ou générez un dataset. L'audit détecte automatiquement les valeurs manquantes, doublons, colonnes constantes et valeurs aberrantes.
        2. **Exploration Visuelle :** Inspectez les distributions, corrélations et relations entre variables avant toute modélisation.
        3. **Nettoyage & Préparation :** Choisissez la variable cible `y` (ou laissez *Aucune* pour K-Means), puis appliquez l'imputation, le traitement des outliers, l'encodage et la normalisation.
        4. **Entraînement & Évaluation :** Sélectionnez un algorithme, ajustez ses hyperparamètres, activez la validation croisée K-Fold et analysez les métriques et graphiques.
        5. **Comparateur & Export :** Comparez tous les modèles exécutés et téléchargez le modèle entraîné (`.joblib`), les prédictions (`.csv`/`.xlsx`), les métriques (`.json`) et les graphiques (`.png`).
        """
    )
    st.divider()
    st.markdown("### 🧠 Fiches Pédagogiques des 11 Algorithmes")
    for k, spec in MODEL_CATALOG.items():
        with st.expander(f"{spec.display_name} — Responsable : {spec.owner}"):
            st.markdown(spec.tutorial)
            st.caption(f"Module cible : `{spec.module_path}` • Classes reconnues : `{', '.join(spec.class_candidates)}`")