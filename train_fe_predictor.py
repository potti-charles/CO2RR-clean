"""Train XGBoost to predict Faradaic efficiency from Magpie + categorical features.

Usage:
  python train_fe_predictor.py
  python train_fe_predictor.py --input outputs/ml_dataset_v2.csv --output_dir outputs/ml_models
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

from build_ml_dataset_v2 import get_feature_columns

warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "fe_pct"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_feature_importance(importances, names, out_path, top_n=25):
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(idx)), importances[idx][::-1], align="center", color="#4C72B0")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top-{top_n} Feature Importances for FE% Prediction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_parity(y_true, y_pred, out_path, title=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.3, s=30)
    lo, hi = 0, 105
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    ax.fill_between([lo, hi], [lo - 10, hi - 10], [lo + 10, hi + 10],
                    alpha=0.1, color="red", label="±10%")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True FE (%)", fontsize=12)
    ax.set_ylabel("Predicted FE (%)", fontsize=12)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.set_title(f"{title}\nR²={r2:.3f}, MAE={mae:.1f}%", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_by_product(df, y_true, y_pred, out_path):
    """Parity plots split by product type."""
    products = df["product"].unique()
    n = len(products)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, prod in enumerate(sorted(products)):
        mask = df["product"] == prod
        ax = axes[i]
        ax.scatter(y_true[mask], y_pred[mask], alpha=0.5, s=20, edgecolors="k", linewidth=0.3)
        ax.plot([0, 100], [0, 100], "r--", linewidth=1)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        n_pts = mask.sum()
        if n_pts > 1:
            r2 = r2_score(y_true[mask], y_pred[mask])
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            ax.set_title(f"{prod} (n={n_pts})\nR²={r2:.2f} MAE={mae:.1f}", fontsize=9)
        else:
            ax.set_title(f"{prod} (n={n_pts})", fontsize=9)
        ax.set_xlabel("True FE%", fontsize=8)
        ax.set_ylabel("Pred FE%", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_by_category(df, y_true, y_pred, category_col, out_path, title=""):
    """Box plot of absolute errors by category."""
    abs_err = np.abs(y_true - y_pred)
    temp = pd.DataFrame({"category": df[category_col].values, "abs_error": abs_err})
    # Sort by median error
    order = temp.groupby("category")["abs_error"].median().sort_values().index
    temp["category"] = pd.Categorical(temp["category"], categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    temp.boxplot(column="abs_error", by="category", ax=ax, vert=True)
    ax.set_title(f"Absolute Error by {title}", fontsize=12)
    ax.set_xlabel(title, fontsize=10)
    ax.set_ylabel("|True - Pred| (%)", fontsize=10)
    plt.suptitle("")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train FE% predictor with XGBoost.")
    parser.add_argument("--input", default="outputs/ml_dataset_v2.csv")
    parser.add_argument("--output_dir", default="outputs/ml_models")
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    feature_cols = get_feature_columns(df)
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features")

    # Filter valid targets
    df = df.dropna(subset=[TARGET]).copy()
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    print(f"Training samples: {len(y)}, Features: {X.shape[1]}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f}, mean={y.mean():.1f}")

    # --- Model ---
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            learning_rate=args.lr,
            reg_alpha=1.0,
            reg_lambda=2.0,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=args.seed,
            verbosity=0,
            tree_method="hist",
        )
        model_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            learning_rate=args.lr,
            random_state=args.seed,
        )
        model_name = "GradientBoosting"

    print(f"Model: {model_name}")

    # --- Cross-validation (grouped by DOI to prevent leakage) ---
    if "doi" in df.columns:
        groups = df["doi"].values
        unique_groups = np.unique(groups)
        n_splits = min(5, len(unique_groups))
        print(f"Using GroupKFold ({n_splits}-fold, grouped by DOI, {len(unique_groups)} groups)")
        cv = GroupKFold(n_splits=n_splits)
        y_pred_cv = cross_val_predict(model, X, y, cv=cv, groups=groups)
    else:
        n_splits = 5
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        y_pred_cv = cross_val_predict(model, X, y, cv=cv)

    # CV metrics
    cv_r2 = r2_score(y, y_pred_cv)
    cv_mae = mean_absolute_error(y, y_pred_cv)
    cv_rmse = float(np.sqrt(mean_squared_error(y, y_pred_cv)))
    within_5 = float(np.mean(np.abs(y - y_pred_cv) <= 5)) * 100
    within_10 = float(np.mean(np.abs(y - y_pred_cv) <= 10)) * 100

    print(f"\n=== Cross-Validation Results (GroupKFold) ===")
    print(f"  R²:    {cv_r2:.3f}")
    print(f"  MAE:   {cv_mae:.2f}%")
    print(f"  RMSE:  {cv_rmse:.2f}%")
    print(f"  ±5%:   {within_5:.1f}%")
    print(f"  ±10%:  {within_10:.1f}%")

    # --- Train final model on all data ---
    model.fit(X, y)
    importances = model.feature_importances_

    # --- Save model ---
    model_path = output_dir / "fe_predictor.json"
    if hasattr(model, "save_model"):
        model.save_model(str(model_path))
    else:
        import pickle
        model_path = output_dir / "fe_predictor.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    print(f"\nSaved model → {model_path}")

    # --- Save feature names ---
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # --- Plots ---
    plot_feature_importance(importances, feature_cols,
                           report_dir / "fe_feature_importance.png")
    plot_parity(y, y_pred_cv,
                report_dir / "fe_parity_cv.png",
                title=f"{model_name} — {n_splits}-Fold GroupKFold CV")
    plot_by_product(df, y, y_pred_cv,
                    report_dir / "fe_parity_by_product.png")
    plot_error_by_category(df, y, y_pred_cv, "catalyst_type",
                           report_dir / "fe_error_by_catalyst_type.png",
                           title="Catalyst Type")
    plot_error_by_category(df, y, y_pred_cv, "product",
                           report_dir / "fe_error_by_product.png",
                           title="Product")
    print("Saved plots → reports/fe_*.png")

    # --- Per-product metrics ---
    print(f"\n=== Per-Product CV Metrics ===")
    prod_metrics = []
    for prod in sorted(df["product"].unique()):
        mask = df["product"] == prod
        n_pts = mask.sum()
        if n_pts < 5:
            continue
        r2 = r2_score(y[mask], y_pred_cv[mask])
        mae = mean_absolute_error(y[mask], y_pred_cv[mask])
        prod_metrics.append({"product": prod, "n": n_pts, "R2": r2, "MAE": mae})
        print(f"  {prod:10s}: n={n_pts:4d}, R²={r2:.3f}, MAE={mae:.1f}%")

    # --- Markdown report ---
    top_feats = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:20]
    report = [
        "# CO2RR Faradaic Efficiency Prediction Report",
        "",
        f"**Model**: {model_name}",
        f"**Dataset**: {len(df)} samples from {df['doi'].nunique()} papers",
        f"**Features**: {len(feature_cols)} (Magpie + element fractions + categorical)",
        f"**Validation**: {n_splits}-fold GroupKFold (grouped by DOI, no data leakage)",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| R² | {cv_r2:.3f} |",
        f"| MAE | {cv_mae:.2f}% |",
        f"| RMSE | {cv_rmse:.2f}% |",
        f"| Within ±5% | {within_5:.1f}% |",
        f"| Within ±10% | {within_10:.1f}% |",
        "",
        "## Per-Product Metrics",
        "",
        "| Product | N | R² | MAE (%) |",
        "|---------|---|-----|---------|",
    ]
    for pm in prod_metrics:
        report.append(f"| {pm['product']} | {pm['n']} | {pm['R2']:.3f} | {pm['MAE']:.1f} |")

    report += [
        "",
        "## Top-20 Feature Importances",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    for rank, (feat, imp) in enumerate(top_feats, 1):
        report.append(f"| {rank} | {feat} | {imp:.4f} |")

    report += [
        "",
        "## Plots",
        "",
        "![Parity Plot](fe_parity_cv.png)",
        "",
        "![Feature Importance](fe_feature_importance.png)",
        "",
        "![Per-Product Parity](fe_parity_by_product.png)",
        "",
        "![Error by Catalyst Type](fe_error_by_catalyst_type.png)",
        "",
        "![Error by Product](fe_error_by_product.png)",
    ]

    report_path = report_dir / "fe_prediction_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved report → {report_path}")


if __name__ == "__main__":
    main()
