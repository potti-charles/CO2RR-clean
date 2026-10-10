"""Train XGBoost (or RandomForest fallback) to predict Faradaic efficiency."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

MIN_ROWS_FOR_XGBOOST = 50
TARGET = "efficiency_pct"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature column names (elem_*, prod_*, potential_V)."""
    cols = [c for c in df.columns if c.startswith("elem_") or c.startswith("prod_")]
    if "potential_V" in df.columns:
        cols.append("potential_V")
    return sorted(cols)


def plot_feature_importance(importances: np.ndarray, names: list, out_path: Path) -> None:
    idx = np.argsort(importances)[::-1][:20]  # top-20
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(idx)), importances[idx][::-1], align="center")
    plt.yticks(range(len(idx)), [names[i] for i in idx][::-1])
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    lo = min(y_true.min(), y_pred.min()) - 5
    hi = max(y_true.max(), y_pred.max()) + 5
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("True FE (%)")
    plt.ylabel("Predicted FE (%)")
    plt.title("Parity Plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML model to predict Faradaic efficiency.")
    parser.add_argument("--input_csv", default="outputs/ml_dataset.csv")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    feature_cols = get_feature_cols(df)
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features")

    if TARGET not in df.columns:
        print(f"Target column '{TARGET}' not found. Exiting.")
        return

    df = df.dropna(subset=[TARGET]).copy()
    # Fill missing feature values with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    y = df[TARGET].values
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Choose model
    use_xgb = X.shape[0] >= MIN_ROWS_FOR_XGBOOST
    if use_xgb:
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                max_depth=4,
                n_estimators=100,
                learning_rate=0.1,
                reg_alpha=1.0,
                reg_lambda=2.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=args.seed,
                verbosity=0,
            )
            model_name = "XGBoost"
        except ImportError:
            print("xgboost not installed, falling back to RandomForest.")
            use_xgb = False

    if not use_xgb:
        model = RandomForestRegressor(
            n_estimators=100, max_depth=4, random_state=args.seed,
        )
        model_name = "RandomForest"

    print(f"Using {model_name} (n={X.shape[0]})")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    # Cross-validation
    n_splits = min(5, max(2, X.shape[0] // 5))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    print(f"Test MAE:  {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²:   {r2:.3f}")
    print(f"CV R² ({n_splits}-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importance
    importances = model.feature_importances_
    fi_path = report_dir / "feature_importance.png"
    plot_feature_importance(importances, feature_cols, fi_path)
    print(f"Saved feature importance plot → {fi_path}")

    # Parity plot
    parity_path = report_dir / "parity_plot.png"
    plot_parity(y_test, y_pred, parity_path)
    print(f"Saved parity plot → {parity_path}")

    # Save model
    model_path = output_dir / "xgb_model.json"
    if use_xgb:
        model.save_model(str(model_path))
    else:
        import pickle
        model_path = output_dir / "rf_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    print(f"Saved model → {model_path}")

    # Markdown report
    top_feats = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:10]
    report_lines = [
        "# ML Prediction Report",
        "",
        f"**Model**: {model_name}  ",
        f"**Samples**: {X.shape[0]} ({X_train.shape[0]} train / {X_test.shape[0]} test)  ",
        f"**Features**: {X.shape[1]}",
        "",
        "## Test Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| MAE    | {mae:.2f} |",
        f"| RMSE   | {rmse:.2f} |",
        f"| R²     | {r2:.3f} |",
        f"| CV R² ({n_splits}-fold) | {cv_scores.mean():.3f} ± {cv_scores.std():.3f} |",
        "",
        "## Top Feature Importances",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    for rank, (feat, imp) in enumerate(top_feats, 1):
        report_lines.append(f"| {rank} | {feat} | {imp:.4f} |")
    report_lines += [
        "",
        "## Plots",
        "",
        "![Feature Importance](feature_importance.png)",
        "",
        "![Parity Plot](parity_plot.png)",
        "",
    ]

    report_path = report_dir / "ml_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved report → {report_path}")


if __name__ == "__main__":
    main()
