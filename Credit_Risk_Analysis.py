import pandas as pd
import numpy as np

"""
1. Load data and minimum validation
2. Ratio summary and interpretation guide
3. Extreme value inspection
4. Year-by-year median ratio analysis
5. Yearly weak-signal rate analysis
6. Distress flag construction and risk bucket assignment
7. Firm-level vulnerable candidate screening
"""




# 1. Load data and minimum validation

"""
Load credit_features.csv and perform minimum input validation.

Detailed preprocessing, account mapping, ratio construction, and balance-sheet
consistency checks were already performed in the previous feature engineering
script. Therefore, this block only verifies that the analysis input satisfies
the expected structure.

Checks:
    - number of firms and fiscal years
    - expected firm-year observations
    - duplicate firm-year observations
    - required ratio columns
    - infinite ratio values
    - missing-value profile

The purpose is not to repeat the full preprocessing validation, but to confirm
that the loaded credit_features.csv is the intended analysis-ready dataset.
"""
credit_analysis = pd.read_csv(r"C:\Users\minec\credit_features.csv")

# Basic panel structure
n_firms = credit_analysis["corp_code"].nunique()
n_years = credit_analysis["bsns_year"].nunique()
years = sorted(credit_analysis["bsns_year"].astype(int).unique().tolist())

# Firm-year uniqueness check
duplicate_count = credit_analysis.duplicated(["corp_code", "bsns_year"]).sum()

# Missing ratio columns
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
    "cfo_to_liabilities"
]

missing_cols = [col for col in ratio_cols if col not in credit_analysis.columns]

# Infinite value check
inf_count = np.isinf(credit_analysis[ratio_cols]).sum()

# Missing value summary
missing_summary = pd.DataFrame({"missing_count": credit_analysis[ratio_cols].isna().sum(),
                                "missing_pct": credit_analysis[ratio_cols].isna().mean()
                                }).sort_values("missing_count", ascending=False)


print("Dataset shape:", credit_analysis.shape)
print(f"Number of firms: {n_firms}")
print(f"Number of years: {n_years}")
print(f"Expected observations: {n_firms * n_years}")
print(f"Actual observations: {len(credit_analysis)}")
print(f"Duplicate firm-year observations: {duplicate_count}")
print("Missing ratio columns:", missing_cols)
print(f"Infinite values in ratio columns: {inf_count}")
print("\nMissing value summary:")
print(missing_summary)

"""
Input validation result:

The dataset contains 581 firm-year observations, corresponding to
83 firms observed over 7 fiscal years from 2019 to 2025.

No duplicate firm-year observations were found, all required ratio columns
are available, and no infinite ratio values are present.

Remaining missing values are concentrated in finance_cost_coverage,
CFO-based ratios, ROA/ROE, and selected liquidity ratios. These missing
patterns are consistent with the underlying financial statement item coverage
from the feature engineering step.

Therefore, the dataset is considered suitable for first-stage descriptive
credit risk analysis.
"""




# 2. Ratio summary and interpretation guide

"""
Compute summary statistics for the core credit-risk ratios.

The ratio interpretation guide below provides broad sanity-check ranges for non-financial firms. 
These ranges are not official rating rules and should not be used mechanically. 
They are used only to support first-stage screening and economic interpretation.

Important interpretation principles:
    
    - Industry characteristics matter.
    - Some ratios are highly sensitive to small denominators.
    - Extreme values are not automatically data errors.
    - Weak extreme values may represent credit-risk signals.
    - Favorable extreme values may reflect denominator effects.


1. liabilities_to_assets = total_liabilities / total_assets

    General interpretation:
        < 0.40  : conservative leverage
        0.40-0.60 : moderate leverage
        0.60-0.75 : relatively leveraged
        > 0.75  : high leverage; should be reviewed carefully
        > 0.85  : potentially weak equity buffer, depending on industry

        Capital-intensive industries can naturally have higher leverage.
        This ratio should be interpreted together with equity_ratio and profitability.

2. equity_ratio = total_equity / total_assets

    General interpretation:
        > 0.50  : strong equity buffer
        0.30-0.50 : moderate equity buffer
        0.15-0.30 : relatively thin equity buffer
        < 0.15  : weak capital structure
        < 0.10  : potentially vulnerable solvency position

        This ratio is mechanically related to liabilities_to_assets.
        
        If the balance sheet is consistent, liabilities_to_assets + equity_ratio
        should be close to 1.

3. current_ratio = current_assets / current_liabilities

    General interpretation:
        < 1.0   : potential short-term liquidity pressure
        1.0-2.0 : generally acceptable
        2.0-3.0 : comfortable liquidity
        > 3.0   : very high liquidity, but not necessarily better

        This ratio can become very large when current liabilities are small.
        
        Extremely high values are not automatically good; they may reflect
        inefficient asset use, excess cash.

4. quick_ratio = (current_assets - inventory) / current_liabilities

    General interpretation:
        < 0.5   : weak quick liquidity
        0.5-1.0 : moderate liquidity
        > 1.0   : generally comfortable
        > 3.0   : unusually high; should be reviewed

        Useful for manufacturing and inventory-heavy firms, but interpretation
        depends heavily on business model and working capital structure.

5. cash_ratio = cash_and_cash_equivalents / current_liabilities

    General interpretation:
        < 0.10  : low immediate liquidity
        0.10-0.30 : modest cash buffer
        0.30-0.70 : relatively comfortable
        > 1.00  : very strong cash coverage, but may indicate excess cash

        A low cash_ratio is not always a problem if the firm has stable cash flow
        and strong access to financing. A high cash_ratio is not always optimal
        because excess cash can reduce capital efficiency.

6. roa = net_income / total_assets

    General interpretation:
        < 0.00  : loss-making; negative credit signal
        0.00-0.02 : weak profitability
        0.02-0.05 : moderate profitability
        > 0.05  : strong profitability
        > 0.15  : unusually high; check industry or one-off effects

        ROA is generally more stable than ROE because total assets are usually
        less volatile than equity. It is one of the cleaner profitability ratios
        for cross-firm comparison.

7. roe = net_income / total_equity

    General interpretation:
        < 0.00  : loss-making; negative credit signal
        0.00-0.05 : weak return on equity
        0.05-0.15 : normal to decent profitability
        > 0.15  : strong profitability
        > 0.30  : potentially strong, but check denominator effects
        extremely negative or positive values: often caused by small equity base (usually in US stock markets)

        ROE can become extreme when equity is small. Therefore, ROE should not be interpreted alone. 
        It should be reviewed together with equity_ratio, liabilities_to_assets, and ROA.

8. operating_margin = operating_income / revenue

    General interpretation:
        < 0.00  : operating loss; negative credit signal
        0.00-0.03 : weak operating profitability
        0.03-0.10 : moderate operating profitability
        > 0.10  : strong operating profitability
        > 0.30  : very high; check industry structure or one-off effects

        Software, platform, and asset-light firms may have high margins, while 
        manufacturing, retail, and commodity-related firms may naturally have lower margins.
        
        Industry-level comparison is essential.

9. finance_cost_coverage = operating_income / finance_costs

    General interpretation:
        < 0.00  : operating loss; cannot cover finance costs from operations
        0.00-1.00 : weak coverage; operating income is insufficient
        1.00-2.00 : thin coverage
        2.00-5.00 : acceptable to moderate coverage
        > 5.00  : comfortable coverage

        This ratio is highly sensitive to small finance_costs values.

10. cfo_to_assets = cfo / total_assets

    General interpretation:
        < 0.00  : negative operating cash flow; negative credit signal
        0.00-0.03 : weak cash generation
        0.03-0.08 : moderate cash generation
        > 0.08  : strong cash generation
        > 0.20  : unusually high; check one-off working capital effects

        CFO can fluctuate due to working capital changes, so one-year negative CFO
        is not always a structural problem. Persistent negative CFO is much more serious.

11. cfo_to_liabilities = cfo / total_liabilities

    General interpretation:
        < 0.00  : negative operating cash flow; weak debt-servicing signal
        0.00-0.10 : weak cash-flow coverage
        0.10-0.25 : moderate coverage
        > 0.25  : strong cash-flow coverage
        > 1.00  : unusually high; check whether liabilities are very small or CFO is temporary

        This ratio is affected by both CFO volatility and the size of liabilities.
        It should be interpreted together with leverage and profitability ratios.
"""

ratio_summary = credit_analysis[ratio_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T

ratio_summary = ratio_summary[["count", "mean", "std", "min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"]].round(4)




# 3. Extreme value inspection

"""
Inspect extreme observations for selected sensitive ratios.

This block identifies the lowest and highest firm-year observations for ratios
that are particularly sensitive to small denominators or business-cycle effects.

Purpose:
    - identify firm-year observations that drive extreme ratio values
    - distinguish possible denominator effects from genuine weak signals
    - avoid mechanically removing extreme values
    - support later weak-signal and scorecard construction

Extreme values are not treated as errors by default.
"""

extreme_check_cols = [
    "current_ratio",
    "quick_ratio",
    "roe",
    "operating_margin",
    "finance_cost_coverage",
    "cfo_to_liabilities"
]

base_cols = [
    "corp_code",
    "stock_code",
    "corp_name_master",
    "industry_krx",
    "bsns_year"
]

extreme_list = []

for col in extreme_check_cols:
    temp = credit_analysis[base_cols + [col]].dropna(subset=[col]).copy()

    lowest = temp.nsmallest(5, col).copy()
    lowest["ratio_name"] = col
    lowest["tail"] = "lowest"
    lowest = lowest.rename(columns={col: "ratio_value"})

    highest = temp.nlargest(5, col).copy()
    highest["ratio_name"] = col
    highest["tail"] = "highest"
    highest = highest.rename(columns={col: "ratio_value"})

    extreme_list.append(lowest)
    extreme_list.append(highest)

extreme_values_df = pd.concat(extreme_list, ignore_index=True)

# Reorder columns
extreme_values_df = extreme_values_df[
    [
        "ratio_name",
        "tail",
        "corp_code",
        "stock_code",
        "corp_name_master",
        "industry_krx",
        "bsns_year",
        "ratio_value"
    ]
]

print(extreme_values_df)


extreme_firm_count = (extreme_values_df.groupby(["corp_name_master", "industry_krx"]).size()
                      .reset_index(name="extreme_count")
                      .sort_values("extreme_count", ascending=False))

print(extreme_firm_count)

"""
extreme_firm_count counts both favorable and unfavorable extremes.

bad_extreme_firm_count focuses only on unfavorable tails and is used as a
preliminary diagnostic screen. These counts are not final risk scores because
they depend on the selected ratios and tail size.
"""


bad_extreme_df = extreme_values_df[
(
        (extreme_values_df["ratio_name"].isin([
            "roe",
            "operating_margin",
            "finance_cost_coverage",
            "cfo_to_liabilities"
        ])) &
        (extreme_values_df["tail"] == "lowest")
    )
    |
    (
        (extreme_values_df["ratio_name"].isin([
            "current_ratio",
            "quick_ratio"
        ])) &
        (extreme_values_df["tail"] == "lowest")
    )
].copy()

bad_extreme_firm_count = (
    bad_extreme_df
    .groupby(["corp_name_master", "industry_krx"])
    .size()
    .reset_index(name="bad_extreme_count")
    .sort_values("bad_extreme_count", ascending=False)
)

print(bad_extreme_firm_count)

"""
extreme_firm_count counts both favorable and unfavorable extremes.
bad_extreme_firm_count focuses only on unfavorable tails and is used as a
preliminary diagnostic screen. These counts are not final risk scores because
they depend on the selected ratios and tail size.
"""




# 4. Year-by-year median ratio analysis

"""
Analyze yearly median ratios to understand the typical credit profile of the
sample over time.

Median values are used because accounting ratios can be skewed by extreme
observations. This block describes the central tendency of the large-cap
non-financial KOSPI universe, not the tail-risk profile.
"""

yearly_ratio_median = (
    credit_analysis
    .groupby("bsns_year")[ratio_cols]
    .median()
    .reset_index()
)

"""
Interpretation summary:
    
    The yearly median profile suggests that the sample is broadly stable.
    Leverage and equity buffer show modest improvement, profitability improves
    after 2020, and liquidity remains generally acceptable. However, median
    ratios may hide weak firm-year observations, so weak-signal rates and
    firm-level flags are analyzed next.
    
    Although a current ratio of 2.0 is often cited as a traditional benchmark, 
    the empirical distribution of large KOSPI non-financial firms suggests that a lower level, 
    around 1.3–1.4, may still represent normal liquidity for large, mature firms.
"""




# 5. Yearly weak-signal rate analysis

"""
Construct individual weak-signal indicators and summarize their yearly rates.

These indicators are preliminary diagnostic signals, not final distress labels.
They measure how often firms fall into potentially vulnerable financial
conditions within each fiscal year.

Weak-signal rules:
    - current_ratio < 1
    - quick_ratio < 0.5
    - liabilities_to_assets > 0.75
    - roa < 0
    - operating_margin < 0
    - finance_cost_coverage < 1
    - cfo < 0

Design choices:
    
    - Missing ratio values are not automatically classified as weak signals.
      Missingness may reflect financial statement item coverage limitations
      rather than actual financial weakness.
      
    - cash_ratio is reviewed in the ratio summary but is not directly used as
      a distress flag. Cash holdings vary substantially by business model and
      financial policy, so liquidity distress is captured through current_ratio
      and quick_ratio.
      
    - Because the sample consists of large non-financial KOSPI firms, the
      objective is relative vulnerability screening, not default prediction.
"""

weak_signal_data = credit_analysis.copy()

# Construct individual weak-signal indicators based on broad credit-risk thresholds.
weak_signal_data["weak_liquidity"] = weak_signal_data["current_ratio"] < 1
weak_signal_data["weak_quick_liquidity"] = weak_signal_data["quick_ratio"] < 0.5
weak_signal_data["high_leverage"] = weak_signal_data["liabilities_to_assets"] > 0.75
weak_signal_data["negative_roa"] = weak_signal_data["roa"] < 0
weak_signal_data["negative_operating_margin"] = weak_signal_data["operating_margin"] < 0
weak_signal_data["weak_finance_cost_coverage"] = weak_signal_data["finance_cost_coverage"] < 1
weak_signal_data["negative_cfo"] = weak_signal_data["cfo"] < 0

weak_signal_cols = [
    "weak_liquidity",
    "weak_quick_liquidity",
    "high_leverage",
    "negative_roa",
    "negative_operating_margin",
    "weak_finance_cost_coverage",
    "negative_cfo"
    ]

yearly_weak_signal_rate = (weak_signal_data.groupby("bsns_year")[weak_signal_cols].mean().reset_index())

yearly_weak_signal_rate_pct = yearly_weak_signal_rate.copy()
yearly_weak_signal_rate_pct[weak_signal_cols] = (yearly_weak_signal_rate_pct[weak_signal_cols] * 100).round(4)

"""
Yearly weak-signal rate interpretation:

The weak-signal rate table shows the proportion of firms that fall into
potentially vulnerable financial conditions in each fiscal year.

Key observations:
    
    - Weak finance cost coverage is the most persistent signal. Around 16-24%
      of firms have finance_cost_coverage below 1 in each year. This suggests
      that even though the median coverage ratio is generally above 3x, a
      meaningful minority of firms face weak operating-income coverage of
      finance costs.

    - Weak liquidity based on current_ratio below 1 appears in roughly 14-24%
      of firms each year. However, weak quick liquidity is much less frequent,
      suggesting that severe quick-liquidity stress is more limited.

    - High leverage is present in a smaller subset of firms, generally below
      11% of the sample. This is consistent with the yearly median analysis,
      which does not indicate broad-based leverage deterioration.

    - Negative ROA and negative operating margin are present but not dominant.
      Profitability weakness is concentrated in a minority of firms rather than
      across the overall sample.

    - Negative CFO signals increase notably in 2022, indicating that cash-flow
      weakness was more pronounced in that year. Because CFO can be affected by
      working-capital movements, persistent negative CFO should be examined at
      the firm level.

Conclusion:
    
    The overall sample does not show broad-based credit deterioration, but a
    persistent vulnerable subset exists, especially in finance cost coverage and
    liquidity. These weak-signal indicators provide a natural foundation for
    later distress flag construction.
    
    
Most large non-financial KOSPI firms show relatively stable median credit profiles.
However, weak-signal analysis identifies a persistent vulnerable subset,
especially in liquidity, finance cost coverage, and cash-flow related metrics.


Because the sample consists of large non-financial KOSPI firms, 
the overall median profile is relatively stable. 
Therefore, the project focuses not on predicting default events, 
but on identifying relative vulnerability and weak financial signals 
within a generally healthy large-cap universe.
"""




# 6. Distress flag construction and risk bucket assignment

"""
Convert individual weak signals into dimension-level distress flags.

The purpose is to avoid double counting similar ratios and to build an
interpretable rule-based screening scorecard.

Credit-risk dimensions:
    1. Liquidity       : weak_liquidity or weak_quick_liquidity
    2. Leverage        : high_leverage
    3. Profitability   : negative_roa or negative_operating_margin
    4. Coverage        : weak_finance_cost_coverage
    5. Cash flow       : negative_cfo

The distress_flag_count ranges from 0 to 5 and measures how many credit-risk
dimensions are weak in a given firm-year.

Risk bucket definition:
    - Low      : 0 weak dimensions
    - Watch    : 1 weak dimension
    - Moderate : 2 weak dimensions
    - High     : 3 or more weak dimensions

The bucket is an internal screening classification, not an official credit
rating or default probability estimate.
"""

credit_flags = weak_signal_data.copy()


credit_flags["flag_weak_liquidity"] = (
    credit_flags["weak_liquidity"] |
    credit_flags["weak_quick_liquidity"]
)

credit_flags["flag_high_leverage"] = (
    credit_flags["high_leverage"]
)

credit_flags["flag_weak_profitability"] = (
    credit_flags["negative_roa"] |
    credit_flags["negative_operating_margin"]
)

credit_flags["flag_weak_coverage"] = (
    credit_flags["weak_finance_cost_coverage"]
)

credit_flags["flag_negative_cashflow"] = (
    credit_flags["negative_cfo"]
)

distress_flag_cols = [
    "flag_weak_liquidity",
    "flag_high_leverage",
    "flag_weak_profitability",
    "flag_weak_coverage",
    "flag_negative_cashflow"
]

credit_flags["distress_flag_count"] = credit_flags[distress_flag_cols].sum(axis=1)


def assign_risk_bucket(flag_count):
    if flag_count >= 3:
        return "High"
    elif flag_count == 2:
        return "Moderate"
    elif flag_count == 1:
        return "Watch"
    else:
        return "Low"

credit_flags["risk_bucket"] = credit_flags["distress_flag_count"].apply(assign_risk_bucket)


yearly_risk_bucket_distribution = (
    credit_flags
    .groupby(["bsns_year", "risk_bucket"])
    .size()
    .reset_index(name="count")
)

yearly_risk_bucket_distribution["pct"] = (
    yearly_risk_bucket_distribution
    .groupby("bsns_year")["count"]
    .transform(lambda x: x / x.sum() * 100)
)

yearly_risk_bucket_distribution["pct"] = yearly_risk_bucket_distribution["pct"].round(2)

print("\nYearly risk bucket distribution:")
print(yearly_risk_bucket_distribution)





# 7. Firm-level vulnerable candidate screening

"""
Summarize firm-level vulnerability across the 2019-2025 panel.

The objective is to identify firms that repeatedly show weak signals across
multiple credit-risk dimensions.

Ranking criteria:
    - number of High-risk years
    - number of Moderate-or-High years
    - average distress flag count

This table is used to select firms for qualitative follow-up review.
It should not be interpreted as a final credit rating ranking.
"""

firm_risk_summary = (
    credit_flags
    .groupby(["corp_code", "stock_code", "corp_name_master", "industry_krx"])
    .agg(
        avg_distress_flag_count=("distress_flag_count", "mean"),
        max_distress_flag_count=("distress_flag_count", "max"),
        high_risk_years=("risk_bucket", lambda x: (x == "High").sum()),
        moderate_or_high_years=("risk_bucket", lambda x: x.isin(["Moderate", "High"]).sum()),
        watch_or_above_years=("risk_bucket", lambda x: x.isin(["Watch", "Moderate", "High"]).sum())
    )
    .reset_index()
    .sort_values(
        ["high_risk_years", "moderate_or_high_years", "avg_distress_flag_count"],
        ascending=False
    )
)

print("\nFirm-level risk summary:")
print(firm_risk_summary.head(20))

# Top vulnerable firms by scorecard
top_vulnerable_firms = firm_risk_summary.head(10)["corp_name_master"].tolist()

firm_flag_trends = (
    credit_flags[
        credit_flags["corp_name_master"].isin(top_vulnerable_firms)
    ][
        [
            "corp_name_master",
            "industry_krx",
            "bsns_year",
            "flag_weak_liquidity",
            "flag_high_leverage",
            "flag_weak_profitability",
            "flag_weak_coverage",
            "flag_negative_cashflow",
            "distress_flag_count",
            "risk_bucket",
            "current_ratio",
            "liabilities_to_assets",
            "roa",
            "operating_margin",
            "finance_cost_coverage",
            "cfo_to_assets"
        ]
    ]
    .sort_values(["corp_name_master", "bsns_year"])
)

print(firm_flag_trends)

"""
Interpretation note:

The scorecard helps classify different types of financial vulnerability:
    - persistent financial weakness
    - public-sector or policy-support overlay cases
    - cyclical or capital-intensive business models
    - past weakness followed by improvement
    - investment-heavy or growth-related cash-flow pressure

High-ranked firms require firm-specific review of business model, industry cycle,
cash-flow sustainability, and qualitative support factors.
"""














