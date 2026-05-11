import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = PROJECT_ROOT / "model-ready"
FIG_DIR = PROJECT_ROOT / "outputs" / "report-figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "target_median_sale_price_log_change_next_year"

BASELINE_FEATURE_COLUMNS = [
    "year",
    "annual_homes_sold",
    "annual_pending_sales",
    "annual_new_listings",
    "annual_inventory_mean",
    "annual_inventory_end",
    "annual_months_of_supply_est",
    "annual_median_sale_price",
    "annual_median_list_price",
    "annual_median_ppsf",
    "annual_median_list_ppsf",
    "annual_avg_sale_to_list",
    "annual_sold_above_list",
    "annual_off_market_in_two_weeks",
    "annual_median_dom",
    "log_total_population",
    "log_median_household_income",
    "median_household_income_missing",
    "age_0_17_share",
    "age_18_24_share",
    "age_25_34_share",
    "age_35_44_share",
    "age_45_64_share",
    "education_less_than_high_school_share",
    "education_some_college_or_associates_share",
    "education_bachelors_or_higher_share",
    "foreign_born_total_share",
    "native_born_other_state_us_share",
    "born_other_state_25plus_bachelors_or_higher_share",
    "foreign_born_25plus_bachelors_or_higher_share",
]

EXTENDED_FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + [
    "is_caaspp",
    "n_schools",
    "log1p_n_schools",
    "avg_pct_met_overall_z_assessment_year",
    "avg_pct_met_ela_z_assessment_year",
    "avg_pct_met_math_z_assessment_year",
    "median_pct_met_overall_z_assessment_year",
    "overall_score_range",
]

BASELINE_INT_COLUMNS = ["median_household_income_missing"]
EXTENDED_INT_COLUMNS = BASELINE_INT_COLUMNS + ["is_caaspp"]

PANEL_COLORS = {
    "Listings model panel": "#1f77b4",
    "Baseline panel": "#ff7f0e",
    "Extended panel": "#2ca02c",
}
MODEL_COLORS = {
    "Model 1": "#1f77b4",
    "Model 2": "#d62728",
}


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    listings = pd.read_csv(DATA_DIR / "listings_model_ready.csv")
    demographics = pd.read_csv(DATA_DIR / "zcta_demographics_features.csv")
    school = pd.read_csv(DATA_DIR / "school_quality_features.csv")

    for frame in (listings, demographics, school):
        frame["zcta5"] = frame["zcta5"].astype(str).str.zfill(5)

    return listings, demographics, school


def build_baseline_panel(listings: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    panel = listings.merge(
        demographics,
        on=["zcta5", "year"],
        how="inner",
        validate="one_to_one",
    )
    panel = panel[
        panel["baseline_model_eligible"] & panel["demographics_baseline_eligible"]
    ].copy()
    return panel.sort_values(["year", "zcta5"]).reset_index(drop=True)


def build_extended_panel(
    listings: pd.DataFrame,
    demographics: pd.DataFrame,
    school: pd.DataFrame,
) -> pd.DataFrame:
    panel = build_baseline_panel(listings, demographics).merge(
        school,
        on=["zcta5", "year"],
        how="inner",
        validate="one_to_one",
    )
    panel = panel[panel["school_quality_extended_eligible"]].copy()
    return panel.sort_values(["year", "zcta5"]).reset_index(drop=True)


def prepare_xy(
    panel: pd.DataFrame,
    feature_columns: list[str],
    int_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    X = panel[feature_columns].copy()
    for column in int_columns:
        X[column] = X[column].astype(int)
    y = panel[TARGET_COLUMN].astype(float)
    return X, y


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rows": float(len(y_true)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def temporal_evaluation(
    panel: pd.DataFrame,
    feature_columns: list[str],
    int_columns: list[str],
    train_year_max: int,
    valid_year_min: int,
    valid_year_max: int,
) -> dict[str, object]:
    X, y = prepare_xy(panel, feature_columns, int_columns)
    train_mask = panel["year"] <= train_year_max
    valid_mask = panel["year"].between(valid_year_min, valid_year_max)

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_valid = X.loc[valid_mask].copy()
    y_valid = y.loc[valid_mask].copy()

    model = build_model()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    return {
        "model": model,
        "train_metrics": evaluate_predictions(y_train, train_pred),
        "test_metrics": evaluate_predictions(y_valid, valid_pred),
        "y_test": y_valid,
        "pred_test": valid_pred,
        "panel_test": panel.loc[valid_mask, ["zcta5", "year"]].reset_index(drop=True),
    }


def grouped_evaluation(
    panel: pd.DataFrame,
    feature_columns: list[str],
    int_columns: list[str],
    random_state: int = 42,
) -> dict[str, object]:
    X, y = prepare_xy(panel, feature_columns, int_columns)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=panel["zcta5"]))

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    model = build_model()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "model": model,
        "train_metrics": evaluate_predictions(y_train, train_pred),
        "test_metrics": evaluate_predictions(y_test, test_pred),
        "y_test": y_test,
        "pred_test": test_pred,
        "panel_test": panel.iloc[test_idx][["zcta5", "year"]].reset_index(drop=True),
    }


def log_to_pct(values: pd.Series | np.ndarray) -> np.ndarray:
    return (np.exp(values) - 1.0) * 100.0


def metrics_row(
    model_name: str,
    sample_name: str,
    evaluation_name: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "model": model_name,
        "sample": sample_name,
        "evaluation": evaluation_name,
        "rows": int(metrics["rows"]),
        "rmse_log": metrics["rmse"],
        "mae_log": metrics["mae"],
        "r2": metrics["r2"],
        "rmse_pct_approx": metrics["rmse"] * 100.0,
        "mae_pct_approx": metrics["mae"] * 100.0,
    }


def make_coverage_figure(
    listings_panel: pd.DataFrame,
    baseline_panel: pd.DataFrame,
    extended_panel: pd.DataFrame,
) -> None:
    year_index = pd.Index(
        range(
            int(min(listings_panel["year"].min(), baseline_panel["year"].min(), extended_panel["year"].min())),
            int(max(listings_panel["year"].max(), baseline_panel["year"].max(), extended_panel["year"].max())) + 1,
        ),
        name="year",
    )

    coverage = pd.DataFrame(
        {
            "Listings model panel": listings_panel.groupby("year").size().reindex(year_index, fill_value=0),
            "Baseline panel": baseline_panel.groupby("year").size().reindex(year_index, fill_value=0),
            "Extended panel": extended_panel.groupby("year").size().reindex(year_index, fill_value=0),
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    for label in coverage.columns:
        ax.plot(
            coverage.index,
            coverage[label],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=label,
            color=PANEL_COLORS[label],
        )

    ax.set_title("Modeling Panel Coverage by Year")
    ax.set_xlabel("Feature year t")
    ax.set_ylabel("ZCTA-year rows")
    ax.set_xticks(list(coverage.index))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_panel_coverage_by_year.png", dpi=200)
    plt.close(fig)

    coverage.reset_index().to_csv(FIG_DIR / "01_panel_coverage_by_year.csv", index=False)


def make_target_distribution_figure(baseline_panel: pd.DataFrame) -> None:
    growth_pct = pd.Series(log_to_pct(baseline_panel[TARGET_COLUMN]), name="target_pct")
    mean_value = float(growth_pct.mean())
    median_value = float(growth_pct.median())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(growth_pct, bins=50, color="#4c78a8", alpha=0.85, edgecolor="white")
    ax.axvline(mean_value, color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {mean_value:.1f}%")
    ax.axvline(median_value, color="#2ca02c", linestyle="--", linewidth=2, label=f"Median: {median_value:.1f}%")
    ax.set_title("Distribution of Next-Year Median Sale Price Growth")
    ax.set_xlabel("Next-year median sale price growth (%)")
    ax.set_ylabel("ZCTA-year rows")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_target_distribution.png", dpi=200)
    plt.close(fig)


def make_growth_trend_figure() -> None:
    annual_listings = pd.read_csv(
        DATA_DIR / "listings_annual_all_residential.csv",
        usecols=["year", "full_year_coverage", "annual_homes_sold", "annual_median_sale_price"],
    )
    annual_listings = annual_listings[
        annual_listings["full_year_coverage"]
        & annual_listings["annual_median_sale_price"].notna()
        & (annual_listings["annual_homes_sold"].fillna(0) > 0)
    ].copy()

    statewide_rows = []
    for year, group in annual_listings.groupby("year", sort=True):
        weights = group["annual_homes_sold"].to_numpy(dtype=float)
        prices = group["annual_median_sale_price"].to_numpy(dtype=float)
        statewide_rows.append(
            {
                "year": int(year),
                "statewide_sale_price": float(np.average(prices, weights=weights)),
                "statewide_homes_sold": float(weights.sum()),
                "zcta_count": int(len(group)),
            }
        )

    statewide = pd.DataFrame(statewide_rows).sort_values("year").reset_index(drop=True)
    statewide["statewide_sale_price_k"] = statewide["statewide_sale_price"] / 1000.0
    statewide["yoy_price_change_pct"] = statewide["statewide_sale_price"].pct_change() * 100.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.15]},
    )
    covid_start = 2019.5
    covid_end = float(statewide["year"].max()) + 0.5

    for ax in axes:
        ax.axvspan(covid_start, covid_end, color="#fdd0a2", alpha=0.28)

    axes[0].plot(
        statewide["year"],
        statewide["statewide_sale_price_k"],
        marker="o",
        linewidth=2.7,
        color="#1f77b4",
        label="Weighted statewide annual sale price",
    )
    axes[0].set_title("California Housing Price Shift Before and After COVID")
    axes[0].set_ylabel("Annual sale price ($K)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(
        0.985,
        0.06,
        "Shaded region: COVID/post-COVID market",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#7f2704",
    )

    bar_colors = ["#9ecae1" if year < 2020 else "#ef8a62" for year in statewide["year"]]
    axes[1].bar(
        statewide["year"],
        statewide["yoy_price_change_pct"],
        color=bar_colors,
        width=0.7,
    )
    axes[1].axhline(0, color="black", linewidth=1, alpha=0.55)
    axes[1].set_xlabel("Calendar year")
    axes[1].set_ylabel("YoY price change (%)")
    axes[1].set_xticks(statewide["year"].tolist())
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_statewide_housing_price_shift.png", dpi=200)
    plt.close(fig)

    statewide.to_csv(FIG_DIR / "03_statewide_housing_price_shift.csv", index=False)


def add_metric_text(ax: plt.Axes, metrics: dict[str, float]) -> None:
    text = "\n".join(
        [
            f"R\u00b2 = {metrics['r2']:.3f}",
            f"MAE \u2248 {metrics['mae'] * 100:.1f}%",
            f"RMSE \u2248 {metrics['rmse'] * 100:.1f}%",
            f"Rows = {int(metrics['rows']):,}",
        ]
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )


def make_scatter_figure(
    temporal_results: dict[str, object],
    grouped_results: dict[str, object],
    titles: tuple[str, str],
    colors: tuple[str, str],
    filename: str,
    alias_filename: str | None = None,
) -> None:
    actual_temporal = log_to_pct(temporal_results["y_test"])
    pred_temporal = log_to_pct(temporal_results["pred_test"])
    actual_grouped = log_to_pct(grouped_results["y_test"])
    pred_grouped = log_to_pct(grouped_results["pred_test"])

    global_min = float(min(actual_temporal.min(), pred_temporal.min(), actual_grouped.min(), pred_grouped.min()))
    global_max = float(max(actual_temporal.max(), pred_temporal.max(), actual_grouped.max(), pred_grouped.max()))
    lims = [global_min, global_max]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    plots = [
        (
            axes[0],
            actual_temporal,
            pred_temporal,
            titles[0],
            temporal_results["test_metrics"],
            colors[0],
        ),
        (
            axes[1],
            actual_grouped,
            pred_grouped,
            titles[1],
            grouped_results["test_metrics"],
            colors[1],
        ),
    ]

    for ax, actual, predicted, title, metrics, color in plots:
        ax.scatter(actual, predicted, alpha=0.28, s=18, color=color)
        ax.plot(lims, lims, linestyle="--", color="black", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Actual next-year growth (%)")
        ax.grid(alpha=0.2)
        add_metric_text(ax, metrics)

    axes[0].set_ylabel("Predicted next-year growth (%)")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    if alias_filename is not None:
        fig.savefig(FIG_DIR / alias_filename, dpi=200)
    plt.close(fig)


def make_performance_comparison_figure(metrics_df: pd.DataFrame) -> None:
    order = ["Temporal validation", "Grouped-ZCTA test"]
    model_order = ["Model 1", "Model 2"]
    plot_df = metrics_df.copy()
    plot_df["evaluation"] = pd.Categorical(plot_df["evaluation"], categories=order, ordered=True)
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values(["evaluation", "model"])

    metrics_to_plot = [
        ("r2", "R\u00b2", False),
        ("mae_pct_approx", "MAE (approx. %)", True),
        ("rmse_pct_approx", "RMSE (approx. %)", True),
    ]

    x = np.arange(len(order))
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for ax, (metric_column, title, is_error_metric) in zip(axes, metrics_to_plot):
        for i, model_name in enumerate(model_order):
            subset = plot_df[plot_df["model"] == model_name].set_index("evaluation").reindex(order)
            offset = (-width / 2) if i == 0 else (width / 2)
            bars = ax.bar(
                x + offset,
                subset[metric_column].values,
                width=width,
                label=model_name,
                color=MODEL_COLORS[model_name],
                alpha=0.9,
            )
            for bar in bars:
                value = bar.get_height()
                label = f"{value:.2f}" if metric_column != "r2" else f"{value:.3f}"
                y_text = value + 0.01 if value >= 0 else value - 0.03
                va = "bottom" if value >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, y_text, label, ha="center", va=va, fontsize=9)

        if not is_error_metric:
            ax.axhline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=12)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Model Performance Comparison on the Matched School-Available Sample", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_model_performance_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    listings, demographics, school = load_frames()

    listings_panel = listings[listings["baseline_model_eligible"]].copy()
    baseline_panel = build_baseline_panel(listings, demographics)
    extended_panel = build_extended_panel(listings, demographics, school)

    baseline_temporal = temporal_evaluation(
        baseline_panel,
        BASELINE_FEATURE_COLUMNS,
        BASELINE_INT_COLUMNS,
        train_year_max=2020,
        valid_year_min=2021,
        valid_year_max=2023,
    )
    baseline_grouped = grouped_evaluation(
        baseline_panel,
        BASELINE_FEATURE_COLUMNS,
        BASELINE_INT_COLUMNS,
    )

    matched_keys = extended_panel[["zcta5", "year"]].copy()
    baseline_matched_panel = baseline_panel.merge(
        matched_keys,
        on=["zcta5", "year"],
        how="inner",
        validate="one_to_one",
    )

    baseline_matched_temporal = temporal_evaluation(
        baseline_matched_panel,
        BASELINE_FEATURE_COLUMNS,
        BASELINE_INT_COLUMNS,
        train_year_max=2019,
        valid_year_min=2021,
        valid_year_max=2023,
    )
    baseline_matched_grouped = grouped_evaluation(
        baseline_matched_panel,
        BASELINE_FEATURE_COLUMNS,
        BASELINE_INT_COLUMNS,
    )
    extended_temporal = temporal_evaluation(
        extended_panel,
        EXTENDED_FEATURE_COLUMNS,
        EXTENDED_INT_COLUMNS,
        train_year_max=2019,
        valid_year_min=2021,
        valid_year_max=2023,
    )
    extended_grouped = grouped_evaluation(
        extended_panel,
        EXTENDED_FEATURE_COLUMNS,
        EXTENDED_INT_COLUMNS,
    )

    metrics_rows = [
        metrics_row("Model 1", "Full baseline sample", "Temporal validation", baseline_temporal["test_metrics"]),
        metrics_row("Model 1", "Full baseline sample", "Grouped-ZCTA test", baseline_grouped["test_metrics"]),
        metrics_row("Model 1", "Matched school sample", "Temporal validation", baseline_matched_temporal["test_metrics"]),
        metrics_row("Model 1", "Matched school sample", "Grouped-ZCTA test", baseline_matched_grouped["test_metrics"]),
        metrics_row("Model 2", "Matched school sample", "Temporal validation", extended_temporal["test_metrics"]),
        metrics_row("Model 2", "Matched school sample", "Grouped-ZCTA test", extended_grouped["test_metrics"]),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(FIG_DIR / "model_comparison_metrics.csv", index=False)

    make_coverage_figure(listings_panel, baseline_panel, extended_panel)
    make_target_distribution_figure(baseline_panel)
    make_growth_trend_figure()
    make_scatter_figure(
        baseline_temporal,
        baseline_grouped,
        titles=("Model 1 Temporal Validation", "Model 1 Grouped-ZCTA Test"),
        colors=("#1f77b4", "#ff7f0e"),
        filename="04_model1_predicted_vs_actual.png",
        alias_filename="model1_scatter_validation_methods.png",
    )
    make_scatter_figure(
        extended_temporal,
        extended_grouped,
        titles=("Model 2 Temporal Validation", "Model 2 Grouped-ZCTA Test"),
        colors=("#d62728", "#9467bd"),
        filename="05_model2_predicted_vs_actual.png",
        alias_filename="model2_scatter_validation_methods.png",
    )
    make_performance_comparison_figure(
        metrics_df[metrics_df["sample"] == "Matched school sample"].copy()
    )


if __name__ == "__main__":
    main()
