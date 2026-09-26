from pathlib import Path

import numpy as np
import pandas as pd


# 1. Load financial features and validate the analysis input
"""
Load the feature table produced by Credit_Risk_Feature_Engineering.py.

Check required columns, firm-year identifiers, annual coverage,
and infinite ratio values before analysis.

Retain missing financial values for explicit treatment in later steps.
"""

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"

credit_analysis = pd.read_csv(
    DATA_DIR / "credit_features.csv",
    dtype={
        "corp_code": "string",
        "stock_code": "string",
    },
)

base_cols = [
    "corp_code",
    "stock_code",
    "corp_name_master",
    "industry_krx",
    "bsns_year",
]

ratio_cols = [
    "liabilities_to_assets",
    "equity_ratio",
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "roa",
    "roe",
    "operating_margin",
    "finance_cost_coverage",
    "cfo_to_assets",
    "cfo_to_liabilities",
]

amount_cols = [
    "current_assets",
    "current_liabilities",
    "inventory",
    "net_income",
    "total_equity",
    "operating_income",
    "revenue",
    "finance_costs",
    "cfo",
    "total_liabilities",
]

# CFO is also used directly when constructing weak signals.
required_cols = base_cols + ratio_cols + amount_cols
missing_cols = [
    col for col in required_cols
    if col not in credit_analysis.columns
]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

if credit_analysis.empty:
    raise ValueError("The analysis input is empty.")

firm_year_keys = ["corp_code", "bsns_year"]

if credit_analysis[firm_year_keys].isna().any().any():
    raise ValueError("Missing firm-year identifiers.")

duplicates = credit_analysis.duplicated(
    firm_year_keys,
    keep=False,
)

if duplicates.any():
    print(
        credit_analysis.loc[duplicates, base_cols]
        .sort_values(firm_year_keys)
        .to_string(index=False)
    )
    raise ValueError("Duplicate firm-year observations.")

# Validate the fiscal-year coverage specified in preprocessing.
expected_years = set(range(2019, 2026))
observed_years = set(credit_analysis["bsns_year"].unique())

if observed_years != expected_years:
    raise ValueError(
        f"Unexpected fiscal-year coverage: {sorted(observed_years)}"
    )

year_counts = (
    credit_analysis.groupby("corp_code")["bsns_year"].nunique()
)

if year_counts.ne(len(expected_years)).any():
    print(year_counts[year_counts.ne(len(expected_years))])
    raise ValueError("Some firms do not have all required fiscal years.")

# Missing values are allowed; infinite ratios are not.
inf_count = np.isinf(credit_analysis[ratio_cols]).sum()

if inf_count.gt(0).any():
    print(inf_count[inf_count.gt(0)])
    raise ValueError("Infinite financial ratio values found.")

missing_summary = pd.DataFrame({
    "missing_count": credit_analysis[ratio_cols].isna().sum(),
    "missing_pct": credit_analysis[ratio_cols].isna().mean() * 100,
}).sort_values("missing_count", ascending=False)

n_firms = credit_analysis["corp_code"].nunique()

print("Dataset shape:", credit_analysis.shape)
print(f"Number of firms: {n_firms}")
print(f"Fiscal years: {sorted(int(year) for year in observed_years)}")
print(f"Expected observations: {n_firms * len(expected_years)}")
print(f"Actual observations: {len(credit_analysis)}")
print("\nMissing value summary:")
print(missing_summary.round(2))




# 2. Summarize financial ratio distributions
"""
Inspect central tendency and distribution tails.
Missing values are excluded separately for each ratio.
"""

ratio_summary = (
    credit_analysis[ratio_cols]
    .describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    .T
)

ratio_summary = ratio_summary[
    [
        "count", "mean", "std", "min",
        "1%", "5%", "25%", "50%", "75%", "95%", "99%",
        "max",
    ]
].round(4)

print(ratio_summary.to_string())




# 3. Inspect extreme ratios and their underlying amounts
"""
Inspect the five lowest and highest available firm-year observations
for each selected ratio across the full sample period.

Include numerator and denominator amounts to support interpretation.
For quick_ratio, the numerator is current assets less inventory.

Extreme values are retained without automatic adjustment or removal.
Lower-tail observations are diagnostic candidates, not distress labels.

Firm-level counts measure appearances across ratios and years.
They are not counts of distinct years or independent risk signals.
"""

# Map each ratio to its numerator and denominator.
ratio_components = {
    "current_ratio": (
        "current_assets",
        "current_liabilities",
    ),
    "quick_ratio": (
        "current_assets_less_inventory",
        "current_liabilities",
    ),
    "roe": (
        "net_income",
        "total_equity",
    ),
    "operating_margin": (
        "operating_income",
        "revenue",
    ),
    "finance_cost_coverage": (
        "operating_income",
        "finance_costs",
    ),
    "cfo_to_liabilities": (
        "cfo",
        "total_liabilities",
    ),
}

# Add the derived quick-ratio numerator to an inspection copy.
inspection_data = credit_analysis.copy()

inspection_data["current_assets_less_inventory"] = (
    inspection_data["current_assets"]
    - inspection_data["inventory"]
)

tail_size = 5
extreme_list = []

for ratio, (numerator, denominator) in ratio_components.items():
    temp = (
        inspection_data[
            base_cols + [ratio, numerator, denominator]
        ]
        .dropna(subset=[ratio])
        .rename(columns={
            ratio: "ratio_value",
            numerator: "numerator_amount",
            denominator: "denominator_amount",
        })
        .copy()
    )

    temp["ratio_name"] = ratio
    temp["numerator_name"] = numerator
    temp["denominator_name"] = denominator

    lowest = temp.nsmallest(tail_size, "ratio_value").copy()
    lowest["tail"] = "lowest"

    highest = temp.nlargest(tail_size, "ratio_value").copy()
    highest["tail"] = "highest"

    extreme_list.extend([lowest, highest])

extreme_values_df = pd.concat(extreme_list, ignore_index=True)

extreme_values_df = extreme_values_df[
    ["ratio_name", "tail"]
    + base_cols
    + [
        "ratio_value",
        "numerator_name",
        "numerator_amount",
        "denominator_name",
        "denominator_amount",
    ]
]

print("\nExtreme observations and underlying amounts:")
print(extreme_values_df.to_string(index=False))


# Count appearances in either tail.
firm_keys = [
    "corp_code",
    "stock_code",
    "corp_name_master",
    "industry_krx",
]

extreme_firm_count = (
    extreme_values_df
    .groupby(firm_keys, dropna=False)
    .size()
    .reset_index(name="extreme_count")
    .sort_values("extreme_count", ascending=False)
    .reset_index(drop=True)
)

print("\nFirm appearances in either tail:")
print(extreme_firm_count.to_string(index=False))


# All six selected ratios use the lower tail for this diagnostic screen.
bad_extreme_df = extreme_values_df[
    extreme_values_df["tail"].eq("lowest")
].copy()

bad_extreme_firm_count = (
    bad_extreme_df
    .groupby(firm_keys, dropna=False)
    .size()
    .reset_index(name="bad_extreme_count")
    .sort_values("bad_extreme_count", ascending=False)
    .reset_index(drop=True)
)

print("\nFirm appearances in the lower tail:")
print(bad_extreme_firm_count.to_string(index=False))




# 4. Year-by-year median ratio analysis
"""
Summarize annual median ratios for the fixed firm sample.

Medians describe the center of the observed distribution and do not
capture vulnerability concentrated in a subset of firms.

Missing values are excluded separately for each ratio.
Report valid observation counts alongside medians to show coverage.
"""

yearly_ratio_median = (
    credit_analysis
    .groupby("bsns_year")[ratio_cols]
    .median()
    .reset_index()
)

yearly_ratio_count = (
    credit_analysis
    .groupby("bsns_year")[ratio_cols]
    .count()
    .reset_index()
)

print("\nYearly median ratios:")
print(yearly_ratio_median.round(4).to_string(index=False))

print("\nValid observations by year and ratio:")
print(yearly_ratio_count.to_string(index=False))




# 5. Yearly weak-signal rate analysis
"""
Apply project-defined screening thresholds, not default labels.
Preserve missing inputs as unknown signals.
Calculate annual rates among non-missing signals and report their counts.
"""

weak_signal_data = credit_analysis.copy()

signal_inputs = credit_analysis[
    [
        "current_ratio",
        "quick_ratio",
        "liabilities_to_assets",
        "roa",
        "operating_margin",
        "finance_cost_coverage",
        "cfo",
    ]
].astype("Float64")

weak_signal_data["weak_liquidity"] = (
    signal_inputs["current_ratio"] < 1
)
weak_signal_data["weak_quick_liquidity"] = (
    signal_inputs["quick_ratio"] < 0.5
)
weak_signal_data["high_leverage"] = (
    signal_inputs["liabilities_to_assets"] > 0.75
)
weak_signal_data["negative_roa"] = (
    signal_inputs["roa"] < 0
)
weak_signal_data["negative_operating_margin"] = (
    signal_inputs["operating_margin"] < 0
)
weak_signal_data["weak_finance_cost_coverage"] = (
    signal_inputs["finance_cost_coverage"] < 1
)
weak_signal_data["negative_cfo"] = (
    signal_inputs["cfo"] < 0
)

weak_signal_cols = [
    "weak_liquidity",
    "weak_quick_liquidity",
    "high_leverage",
    "negative_roa",
    "negative_operating_margin",
    "weak_finance_cost_coverage",
    "negative_cfo",
]

yearly_weak_signal_rate = (
    weak_signal_data
    .groupby("bsns_year")[weak_signal_cols]
    .mean()
    .reset_index()
)

yearly_weak_signal_count = (
    weak_signal_data
    .groupby("bsns_year")[weak_signal_cols]
    .count()
    .reset_index()
)

yearly_weak_signal_rate_pct = yearly_weak_signal_rate.copy()
yearly_weak_signal_rate_pct[weak_signal_cols] = (
    yearly_weak_signal_rate_pct[weak_signal_cols] * 100
)

print("\nYearly weak-signal rates (%):")
print(yearly_weak_signal_rate_pct.round(2).to_string(index=False))

print("\nValid observations by year and signal:")
print(yearly_weak_signal_count.to_string(index=False))




# 6. Distress flag construction and risk bucket assignment
"""
Group related signals into five screening dimensions.
For OR conditions, one True is sufficient; False with missing stays unknown.

Assign scores and buckets only when all five dimensions are assessable.
Buckets are internal screening categories, not credit ratings.
"""

credit_flags = weak_signal_data.copy()

credit_flags["flag_weak_liquidity"] = (
    credit_flags["weak_liquidity"]
    | credit_flags["weak_quick_liquidity"]
)

credit_flags["flag_high_leverage"] = credit_flags["high_leverage"]

credit_flags["flag_weak_profitability"] = (
    credit_flags["negative_roa"]
    | credit_flags["negative_operating_margin"]
)

credit_flags["flag_weak_coverage"] = (
    credit_flags["weak_finance_cost_coverage"]
)

credit_flags["flag_negative_cashflow"] = credit_flags["negative_cfo"]

distress_flag_cols = [
    "flag_weak_liquidity",
    "flag_high_leverage",
    "flag_weak_profitability",
    "flag_weak_coverage",
    "flag_negative_cashflow",
]

credit_flags["available_dimension_count"] = (
    credit_flags[distress_flag_cols].notna().sum(axis=1)
)

# Require all five dimensions for a comparable total score.
credit_flags["distress_flag_count"] = (
    credit_flags[distress_flag_cols]
    .sum(axis=1, min_count=len(distress_flag_cols))
    .astype("Int64")
)


def assign_risk_bucket(flag_count):
    if pd.isna(flag_count):
        return "Insufficient data"
    if flag_count >= 3:
        return "High"
    if flag_count == 2:
        return "Moderate"
    if flag_count == 1:
        return "Watch"
    return "Low"


credit_flags["risk_bucket"] = (
    credit_flags["distress_flag_count"].apply(assign_risk_bucket)
)

bucket_order = [
    "Low",
    "Watch",
    "Moderate",
    "High",
    "Insufficient data",
]

# Include all firms and show categories with zero observations.
yearly_bucket_counts = (
    credit_flags
    .groupby(["bsns_year", "risk_bucket"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=bucket_order, fill_value=0)
)

yearly_risk_bucket_distribution = (
    yearly_bucket_counts
    .rename_axis(columns="risk_bucket")
    .stack()
    .rename("count")
    .reset_index()
)

yearly_risk_bucket_distribution["pct"] = (
    yearly_risk_bucket_distribution["count"]
    / yearly_risk_bucket_distribution
      .groupby("bsns_year")["count"]
      .transform("sum")
    * 100
)

print("\nYearly risk bucket distribution (% of all firms):")
print(
    yearly_risk_bucket_distribution
    .round({"pct": 2})
    .to_string(index=False)
)




# 7. Firm-level vulnerable candidate screening
"""
Summarize vulnerability using assessable years only.
Report assessment coverage and weak-year rates alongside counts.

Rank by High years, Moderate-or-High years, then average score.
Unequal assessment coverage limits comparability.
Candidates require firm-specific review; this is not a credit rating.
"""

firm_risk_summary = (
    credit_flags
    .groupby(firm_keys, dropna=False)
    .agg(
        total_years=("bsns_year", "nunique"),
        assessed_years=("distress_flag_count", "count"),
        avg_distress_flag_count=("distress_flag_count", "mean"),
        max_distress_flag_count=("distress_flag_count", "max"),
        high_risk_years=(
            "risk_bucket",
            lambda x: x.eq("High").sum(),
        ),
        moderate_or_high_years=(
            "risk_bucket",
            lambda x: x.isin(["Moderate", "High"]).sum(),
        ),
        watch_or_above_years=(
            "risk_bucket",
            lambda x: x.isin(["Watch", "Moderate", "High"]).sum(),
        ),
    )
    .reset_index()
)

firm_risk_summary["insufficient_data_years"] = (
    firm_risk_summary["total_years"]
    - firm_risk_summary["assessed_years"]
)

assessed_years = firm_risk_summary["assessed_years"].where(
    firm_risk_summary["assessed_years"] > 0
)

firm_risk_summary["high_risk_year_pct"] = (
    firm_risk_summary["high_risk_years"] / assessed_years * 100
)

firm_risk_summary["moderate_or_high_year_pct"] = (
    firm_risk_summary["moderate_or_high_years"] / assessed_years * 100
)

# Keep firms without assessable years separate from the candidate ranking.
unassessed_firms = firm_risk_summary[
    firm_risk_summary["assessed_years"].eq(0)
].copy()

firm_risk_ranking = (
    firm_risk_summary[
        firm_risk_summary["assessed_years"].gt(0)
    ]
    .sort_values(
        [
            "high_risk_years",
            "moderate_or_high_years",
            "avg_distress_flag_count",
            "corp_code",
        ],
        ascending=[False, False, False, True],
    )
    .reset_index(drop=True)
)

print("\nFirm-level screening summary:")
print(firm_risk_ranking.head(20).round(2).to_string(index=False))

print("\nFirms without assessable years:")
print(unassessed_firms.to_string(index=False))

# Select up to 10 candidates with at least one assessable year rated Watch or above.
top_vulnerable_firms = (
    firm_risk_ranking.loc[
        firm_risk_ranking["watch_or_above_years"].gt(0),
        "corp_code",
    ]
    .head(10)
    .tolist()
)

trend_cols = (
    firm_keys
    + ["bsns_year"]
    + distress_flag_cols
    + [
        "available_dimension_count",
        "distress_flag_count",
        "risk_bucket",
        "current_ratio",
        "quick_ratio",
        "liabilities_to_assets",
        "roa",
        "operating_margin",
        "finance_cost_coverage",
        "cfo",
        "cfo_to_assets",
    ]
)

firm_flag_trends = (
    credit_flags.loc[
        credit_flags["corp_code"].isin(top_vulnerable_firms),
        trend_cols,
    ]
    .sort_values(["corp_code", "bsns_year"])
    .copy()
)

print("\nAnnual profiles of selected candidates:")
print(firm_flag_trends.to_string(index=False))





# 8. Save analysis outputs
"""
Save full analysis results for follow-up analysis and visualization.
Annual weak-signal rates are saved as percentages with valid counts.
"""

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Combine annual signal rates and their assessment denominators.
yearly_weak_signal_rates_long = yearly_weak_signal_rate_pct.melt(
    id_vars="bsns_year",
    value_vars=weak_signal_cols,
    var_name="signal",
    value_name="weak_signal_rate_pct",
)

yearly_weak_signal_counts_long = yearly_weak_signal_count.melt(
    id_vars="bsns_year",
    value_vars=weak_signal_cols,
    var_name="signal",
    value_name="valid_count",
)

yearly_weak_signals = (
    yearly_weak_signal_rates_long
    .merge(
        yearly_weak_signal_counts_long,
        on=["bsns_year", "signal"],
        how="left",
        validate="one_to_one",
    )
    .sort_values(["bsns_year", "signal"])
    .reset_index(drop=True)
)

# Convert the ratio-name index into a column before saving.
ratio_summary_output = (
    ratio_summary
    .rename_axis("ratio_name")
    .reset_index()
)

output_tables = {
    "credit_flags.csv": credit_flags,
    "credit_firm_risk_summary.csv": firm_risk_summary,
    "credit_ratio_summary.csv": ratio_summary_output,
    "credit_extreme_values.csv": extreme_values_df,
    "credit_yearly_ratio_median.csv": yearly_ratio_median,
    "credit_yearly_ratio_count.csv": yearly_ratio_count,
    "credit_yearly_weak_signals.csv": yearly_weak_signals,
    "credit_yearly_risk_buckets.csv": yearly_risk_bucket_distribution,
}

for filename, table in output_tables.items():
    output_path = RESULTS_DIR / filename

    table.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved: {filename} ({len(table):,} rows)")

print(f"\nResults directory: {RESULTS_DIR}")