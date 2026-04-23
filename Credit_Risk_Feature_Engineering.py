"""
Credit Risk Feature Engineering - Financial Statement Mapping

Purpose:
    Transform DART raw financial statement line items into a standardized
    firm-year panel for credit risk analysis.

Input:
    - credit_raw_fs_slim.csv
    - This file is produced by Credit_Risk_Preprocessing.py

Main steps:
    1. Load the raw slim financial statement data.
    2. Keep only core financial statements (BS, IS, CIS, CF) and only IFRS/DART account IDs.
    3. Examine account coverage across firms and years.
    4. Map selected raw accounts into standardized analysis variables.
    5. Resolve IS/CIS overlap with a statement-priority rule.
    6. Remove duplicate firm-year-standard-account observations.
    7. Pivot the long-format data into a one-row-per-firm-year panel.
    8. Compute core credit-risk ratios.
    9. Validate duplicates, missingness, ratio behavior, and balance-sheet consistency.
    10. Save the final feature table for downstream credit risk analysis.

Core design choices:
    - Preserve the original raw data.
    - Exclude SCE in the first-stage model because it contains repeated capital-component-level lines 
      and is less efficient for building a comparable core credit-risk panel.
    - Use only BS / IS / CIS / CF in the first-stage ratio framework.
    - Standardize accounts using sj_div + account_id.
    - Use a priority rule because IS and CIS can coexist.
    - Map ifrs-full_FinanceCosts to finance_costs rather than interest_expense.

Note:
    - finance_costs is used as a proxy for interest-related burden.
    - Validation blocks are intentionally retained so that the preprocessing logic 
      remains transparent, reusable, and easy to audit later.
"""

import pandas as pd

#Load raw slim data
credit_raw_fs_slim = pd.read_csv(r"C:\Users\minec\credit_raw_fs_slim.csv")


core_fs = credit_raw_fs_slim[(credit_raw_fs_slim["sj_div"].isin(["BS", "IS", "CIS", "CF"])) &
    (credit_raw_fs_slim["account_id"].str.startswith("ifrs-full_", na=False) |
     credit_raw_fs_slim["account_id"].str.startswith("dart_", na=False))].copy()




# Account presence table
"""
Logic:
    - Drop duplicate occurrences within the same firm-year first,
      so the counts reflect presence rather than repeated raw rows.

Interpretation:
    - firm_year_count: number of firm-year observations in which the account appears
    - firm_count: number of firms in which the account appears
    - year_count: number of years in which the account appears
"""

account_presence_core = (core_fs.drop_duplicates(["corp_code", "bsns_year", "sj_div", "account_id", "account_nm"])
                         .groupby(["sj_div", "account_id", "account_nm"], dropna=False)
                         .agg(firm_year_count=("corp_code", "size"), firm_count=("corp_code", "nunique"), year_count=("bsns_year", "nunique"))
                         .reset_index().sort_values(["firm_count", "year_count", "firm_year_count"], ascending=[False, False, False])
                         )




# Screen candidate accounts
"""
This block is used because a direct interest-expense account was not clearly identified 
from the account_presence_core table. Therefore, finance-related candidate accounts are screened
using account_nm keywords to identify a practical proxy.
"""

interest_candidates = account_presence_core[
    account_presence_core["account_nm"].str.contains("이자|금융비용|재무비용", na=False)
].copy()

interest_candidates = interest_candidates.sort_values(
    ["firm_year_count", "firm_count", "year_count"],
    ascending=[False, False, False]
)




# Manual mapping table for merging into core_fs
"""
Core variable set:
    - 7 BS accounts
    - 4 IS/CIS accounts
    - 1 CF account

Note on finance_costs:
    - ifrs-full_FinanceCosts may include broader finance-related costs beyond pure interest expense.
      Therefore, the standardized name is finance_costs.

Note on IS/CIS overlap:
    - Some firms report key income statement items in IS, others in CIS, so both are mapped.
"""

account_mapping = pd.DataFrame({
    "sj_div": [
        "BS", "BS", "BS", "BS", "BS", "BS", "BS",
        "CIS", "IS",
        "CIS", "IS",
        "CIS", "IS",
        "CIS", "IS",
        "CF"
    ],
    "account_id": [
        "ifrs-full_Assets",
        "ifrs-full_Liabilities",
        "ifrs-full_Equity",
        "ifrs-full_CurrentAssets",
        "ifrs-full_CurrentLiabilities",
        "ifrs-full_CashAndCashEquivalents",
        "ifrs-full_Inventories",

        "ifrs-full_Revenue", "ifrs-full_Revenue",
        "dart_OperatingIncomeLoss", "dart_OperatingIncomeLoss",
        "ifrs-full_ProfitLoss", "ifrs-full_ProfitLoss",
        "ifrs-full_FinanceCosts", "ifrs-full_FinanceCosts",

        "ifrs-full_CashFlowsFromUsedInOperatingActivities"
    ],
    "std_account": [
        "total_assets",
        "total_liabilities",
        "total_equity",
        "current_assets",
        "current_liabilities",
        "cash_and_cash_equivalents",
        "inventory",

        "revenue", "revenue",
        "operating_income", "operating_income",
        "net_income", "net_income",
        "finance_costs", "finance_costs",

        "cfo"
    ]
})

# Merge mapping into core_fs
fs_standardized_long = core_fs.merge(
    account_mapping,
    on=["sj_div", "account_id"],
    how="inner"
)




# Apply statement priority
"""
Objective:
    
    - Resolve IS/CIS overlap when the same standardized account appears in both statements.

Current rule:
    - BS, CF, and CIS are preferred
    - IS is secondary

if net_income exists in both CIS and IS for the same firm-year, the CIS value is kept.

Interpretation:
    - If both IS and CIS exist and diff is exactly zero, the overlap does not introduce distortion.
    - If overlap count is zero, the priority rule is unlikely to distort values,
      because the account is usually sourced from only one statement.
    - If diff is materially non-zero, statement-specific mapping may need revision.
"""

statement_priority = {"BS": 1, "CF": 1, "CIS": 1, "IS": 2}
fs_standardized_long["statement_priority"] = fs_standardized_long["sj_div"].map(statement_priority)


fs_standardized_long_before_dedupe = fs_standardized_long.copy()

def compare_is_cis(fs_long_before_dedupe: pd.DataFrame, std_account_name: str):
    
    temp = fs_long_before_dedupe[fs_long_before_dedupe["std_account"] == std_account_name].copy()

    temp = temp[temp["sj_div"].isin(["IS", "CIS"])]

    check = (temp.pivot_table(
            index=["corp_code", "bsns_year"],
            columns="sj_div",
            values="thstrm_amount",
            aggfunc="first").reset_index())

    print(f"\n=== {std_account_name} ===")

    if {"IS", "CIS"}.issubset(check.columns): 
        check["diff"] = check["CIS"] - check["IS"]
        print(check["diff"].describe())
    else:
        print("IS and CIS do not both exist in the pivoted result.")

    return check

check_net_income = compare_is_cis(fs_standardized_long_before_dedupe, "net_income")
check_revenue = compare_is_cis(fs_standardized_long_before_dedupe, "revenue")
check_operating_income = compare_is_cis(fs_standardized_long_before_dedupe, "operating_income")
check_finance_costs = compare_is_cis(fs_standardized_long_before_dedupe, "finance_costs")

"""
Result interpretation:
    - net_income showed overlapping IS and CIS observations, and the values were identical.
    - revenue, operating_income, and finance_costs showed no overlapping firm-year observations between IS and CIS.
"""




# Keep only one observation per firm-year-standard account based on the priority rule above.
fs_standardized_long = (fs_standardized_long
                        .sort_values(["corp_code", "bsns_year", "std_account", "statement_priority"])
                        .drop_duplicates(subset=["corp_code", "bsns_year", "std_account"], keep="first")
                        .copy())




# Validation - duplicate check after priority dedupe
dup_check = (fs_standardized_long
             .groupby(["corp_code", "bsns_year", "std_account"])
             .size()
             .reset_index(name="n"))

# empty result = no duplicate firm-year-standard account remains
print(dup_check[dup_check["n"] > 1])




# Pivot to wide format. Build a one-row-per-firm-year panel
"""
fs_standardized_long is a long-format table:
    
    each row represents one standardized account for one firm-year.

credit_fs_wide is a wide-format table:
    
    each row represents one firm-year, and each standardized account 
    becomes a separate column.

Therefore:
    
    the row count decreases because multiple account rows for the same firm-year 
    are collapsed into a single row, the column count increases because 
    std_account values are expanded into separate variables.
"""
credit_fs_wide = (fs_standardized_long
                  .pivot(index=["corp_code", "stock_code", "corp_name_master", "industry_krx", "bsns_year"],
                         columns="std_account",
                         values="thstrm_amount").reset_index())

# Missing Values by Column
print(credit_fs_wide.isna().sum().sort_values(ascending=False))

"""
std_account
finance_costs                44
cfo                          15
net_income                   10
inventory                     7
cash_and_cash_equivalents     7
revenue                       5
total_equity                  2
operating_income              1
corp_code                     0
total_assets                  0
current_liabilities           0
stock_code                    0
current_assets                0
bsns_year                     0
industry_krx                  0
corp_name_master              0
total_liabilities             0
dtype: int64


A small number of missing values remain, mainly in finance_costs, cfo, and net_income.
BS coverage is very strong, and the missing counts in the core panel are low enough
for a first-stage credit-risk feature set. Therefore, the overall wide panel is 
considered sufficiently usable for analysis.
"""




# Calculate core financial ratios
credit_fs_wide["liabilities_to_assets"] = (
    credit_fs_wide["total_liabilities"] / credit_fs_wide["total_assets"]
)

credit_fs_wide["equity_ratio"] = (
    credit_fs_wide["total_equity"] / credit_fs_wide["total_assets"]
)

credit_fs_wide["current_ratio"] = (
    credit_fs_wide["current_assets"] / credit_fs_wide["current_liabilities"]
)

credit_fs_wide["quick_ratio"] = (
    (credit_fs_wide["current_assets"] - credit_fs_wide["inventory"]) /
    credit_fs_wide["current_liabilities"]
)

credit_fs_wide["cash_ratio"] = (
    credit_fs_wide["cash_and_cash_equivalents"] / credit_fs_wide["current_liabilities"]
)

credit_fs_wide["roa"] = (
    credit_fs_wide["net_income"] / credit_fs_wide["total_assets"]
)

credit_fs_wide["roe"] = (
    credit_fs_wide["net_income"] / credit_fs_wide["total_equity"]
)

credit_fs_wide["operating_margin"] = (
    credit_fs_wide["operating_income"] / credit_fs_wide["revenue"]
)

credit_fs_wide["finance_cost_coverage"] = (
    credit_fs_wide["operating_income"] / credit_fs_wide["finance_costs"]
)

credit_fs_wide["cfo_to_assets"] = (
    credit_fs_wide["cfo"] / credit_fs_wide["total_assets"]
)

credit_fs_wide["cfo_to_liabilities"] = (
    credit_fs_wide["cfo"] / credit_fs_wide["total_liabilities"]
)

"""
The number of ratios does not need to match the number of standardized accounts.
A single account can be reused across multiple ratios, and some accounts serve mainly 
as denominators or supporting inputs. Therefore, 12 standardized accounts can reasonably produce 11 core ratios.
"""


# Validation - ratio summary statistics
"""
Interpretation guide:
    
    - high current_ratio / quick_ratio / cash_ratio can still be valid
    - roe and finance_cost_coverage can become extreme when denominators are small
    - negative values may naturally arise from losses, negative CFO, or negative operating income
"""

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

ratio_summary = credit_fs_wide[ratio_cols].describe().T

ratio_summary = ratio_summary[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]

print(ratio_summary)

"""
Most ratio distributions appear economically plausible. Some ratios, 
especially finance_cost_coverage, can show extreme values because the denominator may be very small.
At this stage, the summary is used as a first-pass screening tool rather than a final outlier treatment step.
"""




# Balance sheet consistency check (absolute difference)
credit_fs_wide["bs_check"] = (credit_fs_wide["total_assets"] 
                              - (credit_fs_wide["total_liabilities"] 
                              + credit_fs_wide["total_equity"]))

print(credit_fs_wide["bs_check"].describe())

"""
The absolute BS difference is essentially zero for almost all observations. Only one visible 
non-zero case appears, but it should be interpreted together with the relative BS check below.
"""




# Basic denominator sanity checks
print((credit_fs_wide["total_equity"] <= 0).sum())
print((credit_fs_wide["finance_costs"] <= 0).sum())

"""
No observations have non-positive total_equity, so ROE interpretation is relatively clean.
Three observations have non-positive finance_costs, so finance_cost_coverage 
should be interpreted with caution for those cases.
"""




# Optional safety check for finance_cost_coverage
"""
finance_costs receives additional attention because:
    
    1. it is the denominator of finance_cost_coverage,
    2. small denominator values can create extreme ratio values,
    3. it is a proxy variable rather than a pure interest-expense measure,
    4. its coverage is weaker than the core BS accounts.

In the current sample, three observations have negative finance_costs.
All remaining observed values are either positive or NaN.
Therefore, finance_cost_coverage is set to NaN when finance_costs <= 0
to avoid economically misleading coverage values.
"""

credit_fs_wide.loc[credit_fs_wide["finance_costs"] <= 0, "finance_cost_coverage"] = pd.NA




# Save outputs
credit_fs_wide.to_csv(
    r"C:\Users\minec\credit_features.csv",
    index=False,
    encoding="utf-8-sig"
)




# This file will serve as the input for Credit_Risk_Analysis.py.







