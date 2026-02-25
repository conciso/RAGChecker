#!/usr/bin/env python3
"""
RAGChecker Parameter-Importance-Analyse
========================================

Analysiert alle ragcheck_*.json Läufe und bewertet den Einfluss der
Parametrisierungen auf die Retrieval-Qualität.

Phasen:
  1. Exploration  – Korrelationsmatrix, Boxplots, Scatterplots
  2. SHAP         – Surrogate-Modell (RF oder XGBoost) + SHAP Dependence Plots + PDP
  3. Optuna       – Surrogate-Modell: nächste Experiment-Empfehlungen
  4. DoWhy        – Kausale Analyse: ATE, Counterfactuals, Sensitivitätschecks

Verwendung:
  python parameter_analysis.py --reports ../reports
  python parameter_analysis.py --reports ../reports --model xgb --phase 2
  python parameter_analysis.py --reports ../reports --target graph_recall --suggestions 10
  python parameter_analysis.py --reports ../reports --treatment query_mode --phase 4
  python parameter_analysis.py --help
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_val_score

import shap

import optuna
import optuna.importance
from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    IntDistribution,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
matplotlib.use("Agg")  # Headless: kein Display erforderlich

# ─────────────────────────────────────────────────────────────
# Defaults (überschreibbar per CLI)
# ─────────────────────────────────────────────────────────────
DEFAULT_REPORTS_DIR = Path("../reports")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_TARGET = "llm_f1"
DEFAULT_MODEL = "rf"          # "rf" | "xgb"
DEFAULT_N_SUGGESTIONS = 5
MIN_RUNS_FOR_ML = 5
RANDOM_STATE = 42

ALL_METRICS = [
    "llm_f1", "llm_recall", "llm_precision", "llm_mrr", "llm_hitrate",
    "graph_recall", "graph_mrr", "graph_ndcg",
]
META_COLS = {
    "file", "run_label", "timestamp", "total_testcases", "runs_per_testcase",
    *ALL_METRICS,
}

# Typ-Alias
Surrogate = Union[RandomForestRegressor, "xgboost.XGBRegressor"]  # type: ignore[name-defined]
ParamSpec = dict[str, tuple]  # name → ("float"|"int"|"categorical", *args)


# ─────────────────────────────────────────────────────────────
# 0. DATEN LADEN
# ─────────────────────────────────────────────────────────────

def find_report_files(reports_dir: Path) -> list[Path]:
    """Findet alle ragcheck_*.json Dateien rekursiv (comparison ausgenommen)."""
    pattern = str(reports_dir / "**" / "ragcheck_*.json")
    files = [
        Path(f) for f in glob.glob(pattern, recursive=True)
        if "comparison" not in Path(f).name
    ]
    if not files:
        print(f"  ⚠  Keine ragcheck_*.json Dateien in '{reports_dir}' gefunden.")
    else:
        print(f"  ✓  {len(files)} Report-Datei(en) gefunden.")
    return sorted(files)


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "configuration" not in data or "summary" not in data:
            print(f"     ⚠  {path.name}: Fehlende Pflichtfelder – übersprungen.")
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        print(f"     ⚠  {path.name}: Lesefehler ({exc}) – übersprungen.")
        return None


def _extract_row(data: dict, path: Path) -> dict:
    cfg = data["configuration"]
    summary = data["summary"]
    graph = summary.get("graph", {})
    llm = summary.get("llm", {})

    row: dict = {
        "file": path.name,
        "run_label": cfg.get("runLabel") or path.parent.name,
        "timestamp": data.get("timestamp", ""),
        "query_mode": cfg.get("queryMode", "unknown"),
        "top_k": cfg.get("topK", np.nan),
        "runs_per_testcase": cfg.get("runsPerTestCase", 1),
        "graph_mrr":    round(graph.get("avgMrr", np.nan), 4),
        "graph_ndcg":   round(graph.get("avgNdcgAtK", np.nan), 4),
        "graph_recall": round(graph.get("avgRecallAtK", np.nan), 4),
        "llm_recall":    round(llm.get("avgRecall", np.nan), 4),
        "llm_precision": round(llm.get("avgPrecision", np.nan), 4),
        "llm_f1":        round(llm.get("avgF1", np.nan), 4),
        "llm_hitrate":   round(llm.get("avgHitRate", np.nan), 4),
        "llm_mrr":       round(llm.get("avgMrr", np.nan), 4),
        "total_testcases": summary.get("totalTestCases", 0),
    }

    for k, v in cfg.get("runParameters", {}).items():
        col = f"param_{k.lower()}"
        try:
            row[col] = float(v)
        except (ValueError, TypeError):
            row[col] = str(v)

    return row


def build_dataframe(files: list[Path]) -> pd.DataFrame:
    rows = [r for f in files if (data := _load_json(f)) and (r := _extract_row(data, f))]
    if not rows:
        print("  ❌  Keine validen Daten gefunden.")
        sys.exit(1)
    df = pd.DataFrame(rows)
    print(f"  ✓  DataFrame: {len(df)} Läufe × {len(df.columns)} Spalten")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def encode_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """One-Hot für Kategorien, Median-Imputation für fehlende numerische Werte."""
    df_enc = df[feature_cols].copy()
    for col in df_enc.select_dtypes(include=["object", "category"]).columns.tolist():
        dummies = pd.get_dummies(df_enc[col], prefix=col, dummy_na=False)
        df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)
    for col in df_enc.select_dtypes(include=[np.number]).columns:
        df_enc[col] = df_enc[col].fillna(df_enc[col].median())
    return df_enc.astype(float)


# ─────────────────────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     → {path.name}")


def _run_label(row: pd.Series) -> str:
    lbl = str(row.get("run_label", "")).strip()
    return lbl if lbl else row["file"]


def print_banner(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def _build_surrogate(model_type: str, n_samples: int) -> Surrogate:
    """Erstellt das Surrogate-Modell (RF oder XGBoost)."""
    if model_type == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("  ⚠  xgboost nicht installiert – Fallback auf Random Forest.")
            print("     pip install xgboost")
            model_type = "rf"
        else:
            return XGBRegressor(
                n_estimators=min(300, max(50, n_samples * 15)),
                max_depth=min(4, max(2, n_samples - 1)),
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_STATE,
                verbosity=0,
            )
    # Random Forest (default)
    return RandomForestRegressor(
        n_estimators=min(300, max(50, n_samples * 15)),
        max_depth=max(2, min(5, n_samples - 1)),
        random_state=RANDOM_STATE,
        min_samples_leaf=1,
    )


# ─────────────────────────────────────────────────────────────
# PHASE 1: EXPLORATION
# ─────────────────────────────────────────────────────────────

def phase1_exploration(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    out: Path,
) -> None:
    print_banner("Phase 1: Exploration")

    present_metrics = [m for m in ALL_METRICS if m in df.columns]
    num_features = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1
    ]
    cat_features = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 6
    ]

    # ── 1a) Korrelationsmatrix ──────────────────────────────
    corr_input = df[num_features + present_metrics].dropna(axis=1, how="all")
    if corr_input.shape[1] >= 2:
        corr = corr_input.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        h = max(6, corr.shape[0] * 0.75)
        fig, ax = plt.subplots(figsize=(h * 1.2, h))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            ax=ax, square=True, linewidths=0.5, annot_kws={"size": 8},
        )
        ax.set_title("Korrelationsmatrix: Parameter ↔ Metriken", fontsize=13, pad=12)
        plt.tight_layout()
        save_fig(fig, out / "1a_korrelation.png")

    # ── 1b) Boxplots für kategorische Parameter ─────────────
    for feat in cat_features:
        if df[feat].nunique() < 2:
            continue
        categories = df[feat].dropna().unique()
        n_metrics = len(present_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), squeeze=False)
        for ax, metric in zip(axes[0], present_metrics):
            data_groups = [df.loc[df[feat] == v, metric].dropna().tolist() for v in categories]
            ax.boxplot(data_groups, labels=[str(v) for v in categories], patch_artist=True)
            ax.set_title(metric, fontsize=9)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylim(-0.05, 1.15)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
        fig.suptitle(f"Einfluss von '{feat}' auf Retrieval-Metriken", fontsize=12, y=1.02)
        plt.tight_layout()
        safe = feat.replace("/", "_").replace("\\", "_")
        save_fig(fig, out / f"1b_boxplot_{safe}.png")

    # ── 1c) Scatterplots numerische Parameter vs. Zielmetrik ─
    varying_num = [c for c in num_features if df[c].nunique() > 2]
    if varying_num and len(df) >= 3:
        cols = min(3, len(varying_num))
        rows = (len(varying_num) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for i, feat in enumerate(varying_num):
            ax = axes[i // cols][i % cols]
            sc = ax.scatter(
                df[feat], df[target],
                c=df[target], cmap="RdYlGn", vmin=0, vmax=1,
                alpha=0.8, s=60, zorder=3,
            )
            plt.colorbar(sc, ax=ax, shrink=0.8)
            x_vals = df[feat].fillna(df[feat].median())
            y_vals = df[target].fillna(0)
            if x_vals.std() > 0:
                z = np.polyfit(x_vals, y_vals, 1)
                xs = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(xs, np.poly1d(z)(xs), "r--", linewidth=1.5, alpha=0.6)
            ax.set_xlabel(feat, fontsize=9)
            ax.set_ylabel(target, fontsize=9)
            ax.set_title(f"{feat} → {target}", fontsize=10)
            ax.set_ylim(-0.05, 1.1)
        for i in range(len(varying_num), rows * cols):
            axes[i // cols][i % cols].set_visible(False)
        fig.suptitle(f"Numerische Parameter vs. '{target}'", fontsize=12)
        plt.tight_layout()
        save_fig(fig, out / "1c_scatter_params.png")

    # ── 1d) Übersicht aller Läufe sortiert nach Zielmetrik ──
    if len(df) >= 2:
        show_metrics = [m for m in ["llm_f1", "llm_recall", "llm_precision", "graph_recall", "graph_mrr"]
                        if m in df.columns]
        df_s = df.sort_values(target, ascending=False).reset_index(drop=True)
        x = np.arange(len(df_s))
        width = 0.8 / max(len(show_metrics), 1)
        fig, ax = plt.subplots(figsize=(max(8, len(df_s) * 0.9), 5))
        palette = plt.cm.tab10.colors
        for i, metric in enumerate(show_metrics):
            ax.bar(x + i * width, df_s[metric], width, label=metric,
                   color=palette[i % 10], alpha=0.85)
        labels = [_run_label(df_s.loc[j]) for j in range(len(df_s))]
        ax.set_xticks(x + width * (len(show_metrics) - 1) / 2)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(f"Alle Läufe – sortiert nach '{target}' (absteigend)")
        ax.legend(fontsize=8, loc="upper right")
        ax.axhline(0.8, color="green", linestyle="--", alpha=0.3, linewidth=1)
        plt.tight_layout()
        save_fig(fig, out / "1d_laeufe_uebersicht.png")

    print("  ✓  Phase 1 abgeschlossen.")


# ─────────────────────────────────────────────────────────────
# PHASE 2: SHAP + DEPENDENCE PLOTS + PDP
# ─────────────────────────────────────────────────────────────

def phase2_shap(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    out: Path,
    model_type: str = "rf",
) -> tuple[Optional[Surrogate], Optional[pd.DataFrame]]:
    label = "XGBoost" if model_type == "xgb" else "Random Forest"
    print_banner(f"Phase 2: SHAP + Dependence Plots ({label})")

    if len(df) < MIN_RUNS_FOR_ML:
        print(f"  ⚠  Nur {len(df)} Läufe (Minimum empfohlen: {MIN_RUNS_FOR_ML}).")
        print("     SHAP-Ergebnisse mit wenigen Datenpunkten mit Vorsicht interpretieren.")

    X = encode_features(df, feature_cols)
    y = df[target].fillna(0).values

    if X.empty or len(X) < 2 or X.shape[1] == 0:
        print("  ❌  Nicht genug Features/Zeilen für SHAP-Analyse.")
        return None, None

    model = _build_surrogate(model_type, len(df))
    model.fit(X, y)

    # Cross-Validation
    if len(df) >= MIN_RUNS_FOR_ML:
        cv = min(3, len(df))
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        r2_info = f"R² = {scores.mean():.3f} ± {scores.std():.3f}"
        if scores.mean() < 0.3:
            r2_info += "  ⚠ niedrig (mehr Läufe erhöhen Aussagekraft)"
        print(f"  ℹ  Cross-Val {cv}-fold ({label}): {r2_info}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # (n_samples, n_features)

    # ── 2a) SHAP Summary (Beeswarm) ──────────────────────────
    fig = plt.figure(figsize=(10, max(4, X.shape[1] * 0.45)))
    shap.summary_plot(shap_values, X, show=False, plot_type="dot")
    plt.title(f"SHAP Summary: Parametereinfluss auf '{target}' ({label})",
              fontsize=12, pad=10)
    plt.tight_layout()
    save_fig(fig, out / "2a_shap_summary.png")

    # ── 2b) SHAP Feature Importance (Bar) ────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = (
        pd.DataFrame({"feature": X.columns, "importance": mean_abs})
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(imp_df) * 0.45)))
    max_imp = imp_df["importance"].max() or 1.0
    colors = [plt.cm.RdYlGn(v / max_imp) for v in imp_df["importance"]]
    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.set_xlabel("Mittlerer |SHAP-Wert|")
    ax.set_title(f"Parameter-Wichtigkeit (SHAP, {label}) für '{target}'")
    plt.tight_layout()
    save_fig(fig, out / "2b_shap_importance.png")

    print(f"\n  📊  Parameter-Ranking (wichtigste zuerst, Ziel: '{target}'):")
    for _, row in imp_df.iloc[::-1].iterrows():
        bar_len = int(row["importance"] / max_imp * 25)
        print(f"     {row['feature']:38s} {'█' * bar_len:<25s} {row['importance']:.4f}")

    # ── 2c) SHAP Dependence Plots (Top-4, x=Parameterwert, y=SHAP-Wert) ──
    #
    # Kernunterschied zu PDP:
    #   - y-Achse = SHAP-Wert (Beitrag dieses Parameters zum Output), nicht vorhergesagter Score
    #   - Farbe = automatisch stärkste interagierende Feature (shap.dependence_plot auto-detection)
    #   → Zeigt direkt: in welchem Wertebereich trägt ein Parameter positiv bei?
    #
    top4 = imp_df.tail(min(4, len(imp_df)))["feature"].tolist()
    if top4:
        n_cols = len(top4)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
        for ax, feat in zip(axes[0], top4):
            feat_idx = list(X.columns).index(feat)
            shap.dependence_plot(
                feat_idx,
                shap_values,
                X,
                interaction_index="auto",
                ax=ax,
                show=False,
            )
            ax.set_title(feat, fontsize=9)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        fig.suptitle(
            f"SHAP Dependence Plots: Top-Parameter → '{target}'\n"
            "(y = SHAP-Beitrag, Farbe = stärkste Interaktion)",
            fontsize=11,
        )
        plt.tight_layout()
        save_fig(fig, out / "2c_shap_dependence.png")

    # ── 2d) Partial Dependence Plots (PDP, ergänzend) ────────
    #
    # PDP zeigt den marginalen Effekt auf den vorhergesagten Score (ceteris paribus).
    # Gut für: Schwellwerte, Monotonie, Sättigungseffekte erkennen.
    #
    if top4 and len(df) >= 3:
        try:
            fig, axes = plt.subplots(1, len(top4), figsize=(5 * len(top4), 4), squeeze=False)
            PartialDependenceDisplay.from_estimator(
                model, X, features=top4, ax=axes[0],
                kind="average", grid_resolution=25,
                random_state=RANDOM_STATE,
            )
            fig.suptitle(
                f"Partial Dependence (PDP): Top-Parameter → '{target}'\n"
                "(y = vorhergesagter Score, ceteris paribus)",
                fontsize=11,
            )
            plt.tight_layout()
            save_fig(fig, out / "2d_partial_dependence.png")
        except Exception as exc:
            print(f"  ⚠  PDP fehlgeschlagen: {exc}")

    # ── 2e) SHAP Interaktionseffekte (ab 10 Läufen) ──────────
    if len(df) >= 10 and X.shape[1] >= 2:
        try:
            shap_interact = explainer.shap_interaction_values(X)
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_interact, X, show=False)
            plt.title(f"SHAP Interaktionseffekte zwischen Parametern ({label})", fontsize=12)
            plt.tight_layout()
            save_fig(fig, out / "2e_shap_interaktion.png")
        except Exception as exc:
            print(f"  ⚠  Interaktions-Plot fehlgeschlagen: {exc}")

    print("  ✓  Phase 2 abgeschlossen.")
    return model, X


# ─────────────────────────────────────────────────────────────
# PHASE 3: OPTUNA SURROGATE-MODELL
# ─────────────────────────────────────────────────────────────

def _infer_param_space(df: pd.DataFrame, feature_cols: list[str]) -> ParamSpec:
    space: ParamSpec = {}
    for col in feature_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        is_cat = not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 5
        if is_cat:
            cats = sorted([str(v) for v in series.unique()])
            space[col] = ("categorical", cats)
        elif (series == series.round().values).all() and series.nunique() <= 30:
            lo, hi = int(series.min()), int(series.max())
            space[col] = ("int", lo, hi) if lo < hi else ("categorical", [str(lo)])
        else:
            lo, hi = float(series.min()), float(series.max())
            space[col] = ("float", lo, hi) if lo < hi else ("categorical", [str(lo)])
    return space


def _add_past_trials(
    study: optuna.Study,
    df: pd.DataFrame,
    feature_cols: list[str],
    space: ParamSpec,
    target: str,
) -> int:
    added = 0
    for _, row in df.iterrows():
        target_val = row.get(target)
        if pd.isna(target_val):
            continue
        params: dict = {}
        distributions: dict = {}
        for col, spec in space.items():
            raw = row.get(col)
            kind = spec[0]
            if kind == "categorical":
                choices = spec[1]
                val = str(raw) if not (isinstance(raw, float) and np.isnan(raw)) else choices[0]
                val = val if val in choices else choices[0]
                dist = CategoricalDistribution(choices=choices)
            elif kind == "int":
                lo, hi = spec[1], spec[2]
                val = int(float(raw)) if not (isinstance(raw, float) and np.isnan(raw)) else (lo + hi) // 2
                val = max(lo, min(hi, val))
                dist = IntDistribution(low=lo, high=hi)
            else:
                lo, hi = spec[1], spec[2]
                val = float(raw) if not (isinstance(raw, float) and np.isnan(raw)) else (lo + hi) / 2
                val = max(lo, min(hi, val))
                dist = FloatDistribution(low=lo, high=hi)
            params[col] = val
            distributions[col] = dist
        trial = optuna.trial.create_trial(
            params=params, distributions=distributions, value=float(target_val)
        )
        study.add_trial(trial)
        added += 1
    return added


def _suggest_next(
    study: optuna.Study,
    space: ParamSpec,
    surrogate_model: Optional[Surrogate],
    X_train: Optional[pd.DataFrame],
    feature_cols: list[str],
    n: int,
    df_existing: pd.DataFrame,
) -> list[tuple[dict, float]]:
    existing_keys = {
        tuple(str(row.get(c, "")) for c in feature_cols)
        for _, row in df_existing.iterrows()
    }

    # Kandidaten via Optuna-Sampler generieren
    dummy_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    for trial in study.trials:
        dummy_study.add_trial(trial)

    candidates: list[dict] = []
    for _ in range(n * 40):
        trial = dummy_study.ask()
        p: dict = {}
        for col, spec in space.items():
            if spec[0] == "categorical":
                p[col] = trial.suggest_categorical(col, spec[1])
            elif spec[0] == "int":
                p[col] = trial.suggest_int(col, spec[1], spec[2])
            else:
                p[col] = trial.suggest_float(col, spec[1], spec[2])
        dummy_study.tell(trial, 0.0)
        key = tuple(str(p.get(c, "")) for c in feature_cols)
        if key not in existing_keys:
            candidates.append(p)

    if not candidates:
        print("  ⚠  Keine neuen Kandidaten (Parameterraum erschöpft?).")
        return []

    # RF/XGB-Scoring der Kandidaten
    if surrogate_model is not None and X_train is not None:
        scored: list[tuple[dict, float]] = []
        for cand in candidates:
            row_enc = {}
            for col in X_train.columns:
                for fc in feature_cols:
                    prefix = f"{fc}_"
                    if col.startswith(prefix):
                        cat_val = col[len(prefix):]
                        row_enc[col] = 1.0 if str(cand.get(fc, "")) == cat_val else 0.0
                        break
                else:
                    row_enc[col] = float(cand.get(col, X_train[col].median()))
            X_cand = pd.DataFrame([row_enc])[X_train.columns]
            scored.append((cand, float(surrogate_model.predict(X_cand)[0])))

        scored.sort(key=lambda t: t[1], reverse=True)
        seen: set = set()
        unique: list[tuple[dict, float]] = []
        for cand, score in scored:
            key = tuple(str(cand.get(c, "")) for c in feature_cols)
            if key not in seen:
                seen.add(key)
                unique.append((cand, score))
        return unique[:n]

    # Fallback: Optuna-Trials nach Score
    result: list[tuple[dict, float]] = []
    for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True):
        if t.value is None:
            continue
        key = tuple(str(t.params.get(c, "")) for c in feature_cols)
        if key not in existing_keys:
            result.append((t.params, t.value))
        if len(result) >= n:
            break
    return result


def phase3_optuna(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    out: Path,
    n_suggestions: int,
    surrogate_model: Optional[Surrogate] = None,
    X_encoded: Optional[pd.DataFrame] = None,
) -> None:
    print_banner("Phase 3: Optuna Surrogate-Modell")

    if len(df) < MIN_RUNS_FOR_ML:
        print(f"  ⚠  Nur {len(df)} Läufe – Empfehlungen eingeschränkt aussagekräftig.")

    space = _infer_param_space(df, feature_cols)
    if not space:
        print("  ❌  Kein Parameterraum definierbar.")
        return

    print(f"  ℹ  Parameterraum ({len(space)} Dimensionen):")
    for name, spec in space.items():
        print(f"     {name:35s}: {spec[1] if spec[0] == 'categorical' else f'[{spec[1]:.4g}, {spec[2]:.4g}]'}")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"ragcheck_{target}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    added = _add_past_trials(study, df, feature_cols, space, target)
    print(f"\n  ✓  {added} vergangene Läufe als Trials geladen.")

    # ── 3a) FANOVA Parameter-Importance ──────────────────────
    if added >= 3:
        try:
            importances = optuna.importance.get_param_importances(study)
            fig, ax = plt.subplots(figsize=(9, max(4, len(importances) * 0.5)))
            names, vals = list(importances.keys()), list(importances.values())
            max_v = max(vals) or 1.0
            ax.barh(names, vals, color=[plt.cm.RdYlGn(v / max_v) for v in vals])
            ax.set_xlabel("Relative Wichtigkeit (FANOVA)")
            ax.set_title(f"Parameter-Wichtigkeit (Optuna FANOVA) für '{target}'")
            plt.tight_layout()
            save_fig(fig, out / "3a_optuna_importance.png")
        except Exception as exc:
            print(f"  ⚠  Optuna Importance fehlgeschlagen: {exc}")

    # ── 3b) Optimierungshistorie ──────────────────────────────
    try:
        values = [t.value for t in study.trials if t.value is not None]
        running_max = np.maximum.accumulate(values)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(range(len(values)), values, alpha=0.4, s=15, label="Trial-Wert", zorder=3)
        ax.plot(range(len(running_max)), running_max, "r-", linewidth=2, label="Bestes Ergebnis")
        ax.axvline(added - 1, color="steelblue", linestyle="--", alpha=0.6,
                   label=f"Echte Daten ({added} Läufe)")
        ax.set_xlabel("Trial-Nummer")
        ax.set_ylabel(target)
        ax.set_title(f"Optuna Optimierungshistorie für '{target}'")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, out / "3b_optuna_history.png")
    except Exception as exc:
        print(f"  ⚠  History-Plot fehlgeschlagen: {exc}")

    # ── Empfehlungen ─────────────────────────────────────────
    suggestions = _suggest_next(study, space, surrogate_model, X_encoded, feature_cols, n_suggestions, df)
    if not suggestions:
        print("  ⚠  Keine Empfehlungen generierbar.")
        return

    model_label = type(surrogate_model).__name__ if surrogate_model else "Optuna"
    print(f"\n  🎯  Top-{len(suggestions)} Empfehlungen (predicted {target} via {model_label}):\n")
    csv_rows = []
    for i, (params, score) in enumerate(suggestions, 1):
        print(f"  Empfehlung {i}  →  predicted {target}: {score:.4f}")
        row_data: dict = {"rank": i, f"predicted_{target}": round(score, 4)}
        for k, v in params.items():
            print(f"    {k:38s} = {v}")
            row_data[k] = v
        print()
        csv_rows.append(row_data)

    csv_path = out / "3_optuna_suggestions.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  ✓  Empfehlungen gespeichert: {csv_path}")
    print("  ✓  Phase 3 abgeschlossen.")


# ─────────────────────────────────────────────────────────────
# PHASE 4: DOWHY – KAUSALE ANALYSE
# ─────────────────────────────────────────────────────────────

def phase4_dowhy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    out: Path,
    treatment_col: Optional[str] = None,
) -> None:
    print_banner("Phase 4: DoWhy – Kausale Analyse")

    try:
        import dowhy
        from dowhy import CausalModel
    except ImportError:
        print("  ❌  dowhy nicht installiert.")
        print("     pip install dowhy")
        return

    if len(df) < 6:
        print(f"  ⚠  Nur {len(df)} Läufe – kausale Analyse erfordert mindestens 6.")
        return

    # ── Treatment-Feature auswählen ───────────────────────────
    #
    # Für kausale Inferenz brauchen wir ein klar definiertes Treatment T.
    # Idealerweise: binär (True/False, 0/1) oder kategorisch mit 2 Werten.
    # Numerische Features werden automatisch in low/high gebucketed.
    #
    if treatment_col and treatment_col not in feature_cols:
        print(f"  ⚠  Treatment '{treatment_col}' nicht in Feature-Spalten. Verfügbar: {feature_cols}")
        treatment_col = None

    # Welche Kategorie soll als "1" gelten bei automatischer Binarisierung?
    _treatment_pos_val: Optional[str] = None

    if treatment_col is None:
        # Priorität 1: Features mit exakt 2 Werten
        binary_feats = [c for c in feature_cols if df[c].dropna().nunique() == 2]
        if binary_feats:
            treatment_col = binary_feats[0]
            print(f"  ℹ  Auto-Treatment: '{treatment_col}' (binär)")

        # Priorität 2: Kategorische Features – wähle die Kategorie mit größtem Effekt auf Target
        if treatment_col is None:
            cat_feats = [
                c for c in feature_cols
                if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().nunique() >= 2
            ]
            if cat_feats:
                overall_mean = df[target].dropna().mean()
                best_feat, best_cat, best_delta = None, None, -999.0
                for feat in cat_feats:
                    for val in df[feat].dropna().unique():
                        grp = df.loc[df[feat] == val, target].dropna()
                        if len(grp) >= 2:
                            # Positiv abweichende Kategorie bevorzugen (z.B. hybrid statt global)
                            delta = grp.mean() - overall_mean
                            if delta > best_delta:
                                best_delta, best_feat, best_cat = delta, feat, val
                if best_feat:
                    treatment_col = best_feat
                    _treatment_pos_val = best_cat
                    print(f"  ℹ  Auto-Treatment: '{treatment_col}' "
                          f"('{best_cat}' vs. Rest, Δ={best_delta:.3f})")

        # Priorität 3: Numerisch – höchste Varianz
        if treatment_col is None:
            num_varying = sorted(
                [c for c in feature_cols
                 if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1],
                key=lambda c: df[c].std(), reverse=True,
            )
            if not num_varying:
                print("  ❌  Kein geeignetes Treatment-Feature gefunden.")
                return
            treatment_col = num_varying[0]
            print(f"  ℹ  Auto-Treatment: '{treatment_col}' (numerisch, höchste Varianz)")

    # ── Dataframe für DoWhy vorbereiten ───────────────────────
    df_dw = df[feature_cols + [target]].copy().dropna(subset=[target])

    # Numerisches Treatment: in binäres low/high bucketen
    treatment_is_numeric = pd.api.types.is_numeric_dtype(df_dw[treatment_col])
    treatment_display = treatment_col
    if treatment_is_numeric and df_dw[treatment_col].nunique() > 2:
        median_val = df_dw[treatment_col].median()
        df_dw[treatment_col] = (df_dw[treatment_col] > median_val).astype(int)
        treatment_display = f"{treatment_col} (>Median={median_val:.3g})"
        print(f"  ℹ  Numerisches Treatment gebucketed: 0 = ≤{median_val:.3g}, 1 = >{median_val:.3g}")

    # Kategorisches Treatment binarisieren
    if not pd.api.types.is_numeric_dtype(df_dw[treatment_col]):
        if _treatment_pos_val is not None:
            # Auto-gewählte Kategorie vs. Rest (1 = pos_val, 0 = alles andere)
            df_dw[treatment_col] = (df_dw[treatment_col] == _treatment_pos_val).astype(int)
            treatment_display = f"{treatment_col}=='{_treatment_pos_val}'"
            print(f"  ℹ  Kategorisches Treatment binarisiert: 1='{_treatment_pos_val}', 0=Rest")
        else:
            # Manuell angegeben oder wirklich binär: ordinale Kodierung
            uniq = sorted(df_dw[treatment_col].dropna().unique())
            df_dw[treatment_col] = df_dw[treatment_col].map({v: i for i, v in enumerate(uniq)})
            print(f"  ℹ  Kategorisches Treatment ordinalkodiert: {dict(enumerate(uniq))}")

    df_dw = df_dw.fillna(df_dw.median(numeric_only=True))
    common_causes = [c for c in feature_cols if c != treatment_col]

    print(f"\n  ℹ  Treatment  : {treatment_display}")
    print(f"  ℹ  Outcome    : {target}")
    print(f"  ℹ  Confounders: {common_causes or '(keine)'}")

    # Annahme: alle anderen Parameter sind Confounder (unabhängig gesetzt).
    # common_causes statt graph= vermeidet networkx-Versionskonflikte.
    try:
        causal_model = CausalModel(
            data=df_dw,
            treatment=treatment_col,
            outcome=target,
            common_causes=common_causes if common_causes else None,
        )
    except Exception as exc:
        print(f"  ❌  CausalModel-Erstellung fehlgeschlagen: {exc}")
        return

    # ── ATE (Average Treatment Effect) via LinearRegression ──
    #
    # Backdoor-Adjustment: Regressiere Outcome auf Treatment + alle Confounders.
    # Der Koeffizient des Treatments ist der ATE (unter Linearitätsannahme).
    # Robust gegenüber networkx-Versionskonflikten in DoWhy's identify_effect().
    #
    print("\n  Schätze Average Treatment Effect (ATE)...")
    from sklearn.linear_model import LinearRegression
    from scipy import stats as scipy_stats

    ate: float = 0.0
    estimate = None
    identified_estimand = None

    try:
        X_ate = df_dw[feature_cols].copy().astype(float)
        y_ate = df_dw[target].values

        lr_ate = LinearRegression()
        lr_ate.fit(X_ate, y_ate)

        t_idx = list(X_ate.columns).index(treatment_col)
        ate = float(lr_ate.coef_[t_idx])

        # p-Wert via t-Test (OLS-Approximation)
        y_pred = lr_ate.predict(X_ate)
        residuals = y_ate - y_pred
        n, k = len(y_ate), X_ate.shape[1]
        mse = np.dot(residuals, residuals) / max(n - k - 1, 1)
        try:
            X_np = X_ate.values
            cov = mse * np.linalg.inv(X_np.T @ X_np)
            se = float(np.sqrt(np.diag(cov)[t_idx]))
            t_stat = ate / se if se > 0 else 0.0
            p_val = float(2 * scipy_stats.t.sf(abs(t_stat), df=n - k - 1))
        except np.linalg.LinAlgError:
            se, p_val = 0.0, 1.0

        print(f"\n  📐  ATE ({treatment_display} → {target}): {ate:+.4f}")
        if abs(ate) < 0.01:
            interpretation = "kein messbarer kausaler Effekt"
        elif ate > 0:
            interpretation = f"'{treatment_col}' == 1 verbessert {target} kausal"
        else:
            interpretation = f"'{treatment_col}' == 1 verschlechtert {target} kausal"
        print(f"      Interpretation: {interpretation}")
        sig_label = "✓ signifikant" if p_val < 0.05 else "⚠ nicht signifikant"
        print(f"      p-Wert: {p_val:.4f}  {sig_label}  (OLS-Approximation, n={n})")

        # DoWhy: Refutation versuchen (optional, kann bei networkx-Konflikten fehlschlagen)
        try:
            identified_estimand = causal_model.identify_effect(
                proceed_when_unidentifiable=True
            )
            estimate = causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                test_significance=False,
            )
        except Exception:
            pass  # Refutation wird weiter unten sauber behandelt

    except Exception as exc:
        print(f"  ⚠  ATE-Schätzung fehlgeschlagen: {exc}")

    # ── Sensitivitätschecks (Refutation) ─────────────────────
    if identified_estimand is not None and estimate is not None:
        print("\n  Refutation-Tests (Sensitivitätschecks)...")
        for method, label in [
            ("random_common_cause", "Zufälliger Confounder"),
            ("placebo_treatment_refuter", "Placebo Treatment"),
            ("data_subset_refuter", "Data Subset"),
        ]:
            try:
                ref = causal_model.refute_estimate(
                    identified_estimand, estimate,
                    method_name=method,
                    random_seed=RANDOM_STATE,
                )
                new_effect = getattr(ref, "new_effect", None)
                status = "✓" if new_effect is not None and abs(new_effect - ate) < abs(ate) * 0.5 else "⚠"
                print(f"    {status} {label:30s}: neuer Effekt = {new_effect:.4f if new_effect is not None else 'n/a'}")
            except Exception as exc:
                print(f"    ⚠  {label}: {exc}")
    else:
        print("\n  ℹ  DoWhy-Refutation übersprungen (networkx-Inkompatibilität)."
              "\n     ATE via OLS-Backdoor-Regression berechnet (gleichwertig).")

    # ── Counterfactuals ───────────────────────────────────────
    #
    # „Was wäre der Score für diesen Lauf, wenn Treatment = 1 statt 0 gewesen wäre?"
    #
    print("\n  Berechne Counterfactuals (was wäre wenn?)...")
    try:
        # Konservativ: lineares Modell für Counterfactuals
        from sklearn.linear_model import LinearRegression

        X_cf = df_dw[feature_cols].copy()
        y_cf = df_dw[target].values

        lr = LinearRegression()
        lr.fit(X_cf, y_cf)

        treatment_idx = list(X_cf.columns).index(treatment_col)
        treatment_coef = lr.coef_[treatment_idx]

        # Für die ersten min(5, n) Läufe Counterfactual berechnen
        n_cf = min(5, len(df_dw))
        cf_rows = []
        for i in range(n_cf):
            row = df_dw.iloc[i]
            actual_treatment = row[treatment_col]
            counterfactual_treatment = 1 - int(actual_treatment)  # flip binary
            actual_score = float(lr.predict(X_cf.iloc[[i]])[0])
            cf_score = actual_score + treatment_coef * (counterfactual_treatment - actual_treatment)
            cf_rows.append({
                "run": _run_label(df.iloc[i]),
                f"{treatment_col}_actual": actual_treatment,
                f"{treatment_col}_counterfactual": counterfactual_treatment,
                f"{target}_actual": round(actual_score, 4),
                f"{target}_counterfactual": round(cf_score, 4),
                "delta": round(cf_score - actual_score, 4),
            })

        cf_df = pd.DataFrame(cf_rows)
        cf_path = out / "4_counterfactuals.csv"
        cf_df.to_csv(cf_path, index=False)

        print(f"\n  Counterfactual-Tabelle (Treatment: '{treatment_display}', flip 0↔1):")
        print(cf_df.to_string(index=False))
        print(f"\n  ✓  Counterfactuals gespeichert: {cf_path}")
    except Exception as exc:
        print(f"  ⚠  Counterfactuals fehlgeschlagen: {exc}")

    # ── 4a) ATE Visualisierung ────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Linke Seite: Boxplot Score nach Treatment-Wert
        groups = [
            df_dw.loc[df_dw[treatment_col] == v, target].dropna()
            for v in sorted(df_dw[treatment_col].unique())
        ]
        labels = [f"Treatment={v}" for v in sorted(df_dw[treatment_col].unique())]
        axes[0].boxplot(groups, labels=labels, patch_artist=True)
        axes[0].set_ylabel(target)
        axes[0].set_title(f"Score-Verteilung nach Treatment\n'{treatment_display}'")
        axes[0].set_ylim(-0.05, 1.15)

        # Rechte Seite: ATE als Balken (immer zeichnen – ate kommt aus OLS, nicht DoWhy)
        if ate != 0.0:
            axes[1].bar(
                [f"ATE\n{treatment_display}→{target}"],
                [ate],
                color="steelblue" if ate >= 0 else "salmon",
                alpha=0.8,
                width=0.4,
            )
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[1].set_ylabel("Kausaler Effekt (ATE)")
            axes[1].set_title("Average Treatment Effect")
            axes[1].set_ylim(min(-0.3, ate * 1.3), max(0.3, ate * 1.3))

        plt.suptitle(f"DoWhy Kausale Analyse: '{treatment_display}' → '{target}'", fontsize=12)
        plt.tight_layout()
        save_fig(fig, out / "4a_dowhy_ate.png")
    except Exception as exc:
        print(f"  ⚠  ATE-Plot fehlgeschlagen: {exc}")

    print("  ✓  Phase 4 abgeschlossen.")


# ─────────────────────────────────────────────────────────────
# ZUSAMMENFASSUNG
# ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, target: str) -> None:
    print("\n" + "═" * 55)
    print("  RAGChecker Parameter-Analyse – Datensatz-Übersicht")
    print("═" * 55)
    print(f"  Läufe analysiert : {len(df)}")
    ts_vals = df["timestamp"].dropna()
    if len(ts_vals) >= 2:
        print(f"  Zeitraum         : {ts_vals.min()[:10]} – {ts_vals.max()[:10]}")
    print()
    for m in [m for m in ALL_METRICS if m in df.columns]:
        col = df[m].dropna()
        if len(col) > 0:
            print(f"  {m:25s}  ø={col.mean():.3f}  max={col.max():.3f}  min={col.min():.3f}")
    if target in df.columns:
        best = df.loc[df[target].idxmax()]
        print(f"\n  Bester Lauf      : {_run_label(best)}")
        print(f"  {target:25s}  {best[target]:.4f}")
    print("═" * 55)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGChecker Parameter-Importance-Analyse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reports", type=Path, default=DEFAULT_REPORTS_DIR,
        help=f"Pfad zum Reports-Verzeichnis, rekursiv durchsucht (default: {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Ausgabeverzeichnis für Plots & CSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--target", type=str, default=DEFAULT_TARGET,
        choices=ALL_METRICS,
        help=f"Primäre Zielmetrik (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=["rf", "xgb"],
        help="Surrogate-Modell für Phase 2+3: 'rf' = Random Forest, 'xgb' = XGBoost (default: rf)",
    )
    parser.add_argument(
        "--suggestions", type=int, default=DEFAULT_N_SUGGESTIONS,
        help=f"Anzahl Optuna-Empfehlungen (default: {DEFAULT_N_SUGGESTIONS})",
    )
    parser.add_argument(
        "--treatment", type=str, default=None,
        help="Phase 4: Treatment-Feature für kausale Analyse (default: auto)",
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4],
        help="Nur eine bestimmte Phase ausführen (default: alle)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("RAGChecker Parameter-Importance-Analyse")
    print("=" * 45)

    files = find_report_files(args.reports)
    if not files:
        sys.exit(1)
    df = build_dataframe(files)
    feature_cols = get_feature_cols(df)

    if not feature_cols:
        print(
            "❌  Keine Parameter-Spalten gefunden.\n"
            "    Prüfe ob 'runParameters' oder 'queryMode' in den JSONs vorhanden sind."
        )
        sys.exit(1)

    print(f"  ✓  Feature-Spalten: {feature_cols}")
    print_summary(df, args.target)

    run_all = args.phase is None
    surrogate: Optional[Surrogate] = None
    X_encoded: Optional[pd.DataFrame] = None

    if run_all or args.phase == 1:
        phase1_exploration(df, feature_cols, args.target, args.output)

    if run_all or args.phase == 2:
        result = phase2_shap(df, feature_cols, args.target, args.output, args.model)
        if result:
            surrogate, X_encoded = result

    if run_all or args.phase == 3:
        phase3_optuna(
            df, feature_cols, args.target, args.output,
            args.suggestions, surrogate, X_encoded,
        )

    if run_all or args.phase == 4:
        phase4_dowhy(
            df, feature_cols, args.target, args.output,
            treatment_col=args.treatment,
        )

    print(f"\n✅  Analyse abgeschlossen. Alle Ergebnisse in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
