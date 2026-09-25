"""
Credit Risk Feature Engineering

Purpose:
- Transform annual consolidated financial statement accounts into a
  standardized firm-year panel for financial vulnerability screening.
- Calculate core financial ratios and report data-quality diagnostics.

Input:
- data/credit_raw_fs_slim.csv, produced by Credit_Risk_Preprocessing.py.

Output:
- data/credit_features.csv, containing firm identifiers, 12 financial
  variables, 11 financial ratios, and balance sheet consistency measures.

Account selection and mapping:
- Retain BS, IS, CIS, and CF rows with IFRS/DART account IDs.
- Exclude SCE because the selected financial variables do not require it.
- Summarize account presence and inspect interest- and finance-related
  candidates before applying the manual mapping.
- Map accounts using the combination of sj_div and account_id.
- Account presence does not guarantee a non-missing financial amount.

Duplicate handling:
- Stop if multiple rows map to the same variable within a firm-year
  and statement.
- Compare available IS and CIS amounts and report discrepancies.
- Select one observation per firm-year and variable, preferring
  non-missing amounts, then CIS over IS.

Ratio definitions and missing values:
- Use current-period amounts (thstrm_amount).
- ROA and ROE use year-end assets and equity.
- Quick ratio is calculated as (current assets - inventory)
  / current liabilities.
- Finance-cost coverage uses operating income / finance costs.
  Finance costs may include items beyond pure interest expense.
- Calculate ratios only when denominators are positive.
- Preserve missing values and negative numerators.
- Keep the underlying financial amounts unchanged.

Validation:
- Report account coverage, IS/CIS overlap, and panel missingness.
- Summarize ratio distributions after denominator treatment.
- Check assets = liabilities + equity using both signed monetary
  differences and absolute differences relative to total assets.
- Report missing, zero, and negative denominators separately.

Scope:
- Prepare financial features for downstream screening.
- Credit scoring and firm-level interpretation are performed in
  Credit_Risk_Analysis.py.
"""

from pathlib import Path
import pandas as pd


# 1. Set project paths and load financial statement data
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"

credit_raw_fs_slim = pd.read_csv(
    DATA_DIR / "credit_raw_fs_slim.csv",
    dtype={
        "corp_code": "string",
        "stock_code": "string",
    },
)

print(credit_raw_fs_slim.shape)
print(credit_raw_fs_slim.dtypes)




# 2. Select core financial statements and IFRS/DART account IDs
core_fs = credit_raw_fs_slim[
    credit_raw_fs_slim["sj_div"].isin(["BS", "IS", "CIS", "CF"])
    & credit_raw_fs_slim["account_id"].str.startswith(
        ("ifrs-full_", "dart_"),
        na=False,
    )
].copy()

print(f"Rows before filtering: {len(credit_raw_fs_slim):,}")
print(f"Rows after filtering: {len(core_fs):,}")
print(core_fs["sj_div"].value_counts())




# 3. Summarize account presence
"""
Count each statement-account combination once per firm-year.
Collect distinct account names for inspection.

Interpretation:
    - firm_year_count: number of distinct firm-year observations
      in which the statement-account combination appears
    - firm_count: number of distinct firms
    - year_count: number of distinct fiscal years

Account presence does not guarantee a non-missing amount.
"""

account_keys = ["sj_div", "account_id"]

# Count distinct firm-year occurrences.
account_presence_core = (
    core_fs
    .drop_duplicates(["corp_code", "bsns_year"] + account_keys)
    .groupby(account_keys, dropna=False)
    .agg(
        firm_year_count=("corp_code", "size"),
        firm_count=("corp_code", "nunique"),
        year_count=("bsns_year", "nunique"),
    )
    .reset_index()
)

# Collect the different names used for each statement-account combination.
account_names = (
    core_fs
    .groupby(account_keys, dropna=False)
    .agg(
        account_names=(
            "account_nm",
            lambda names: " | ".join(sorted(names.dropna().unique())),
        )
    )
    .reset_index()
)

account_presence_core = (
    account_presence_core
    .merge(
        account_names,
        on=account_keys,
        how="left",
        validate="one_to_one",
    )
    .sort_values(
        ["firm_year_count", "firm_count", "year_count"],
        ascending=False,
    )
    .reset_index(drop=True)
)

print(account_presence_core.head(5).to_string(index=False))




# 4. Screen broadly for interest and finance-related accounts
"""
Identify candidate accounts for interest expense and finance costs.
Review their statement context and meaning before mapping.
"""

interest_candidates = account_presence_core[
    account_presence_core["sj_div"].isin(["CF", "CIS", "IS"])
    & (
        account_presence_core["account_names"].str.contains(
            "이자|금융비용|금융원가|재무비용",
            na=False,
        )
        | account_presence_core["account_id"].str.contains(
            "Interest|FinanceCosts",
            case=False,
            na=False,
        )
    )
].copy()

print(interest_candidates.to_string(index=False))




# 5. Map financial statement accounts to analysis variables
"""
Core variable set:
    - 7 BS variables
    - 4 IS/CIS variables
    - 1 CF variable

Mapping uses the combination of sj_div and account_id.

Finance costs:
    - Use ifrs-full_FinanceCosts as a broader financing-cost measure.
    - Do not treat finance_costs as pure interest expense.

IS/CIS overlap:
    - Map both statements to the same analysis variables.
    - Resolve overlapping observations in the next step.
"""

account_mapping = pd.DataFrame(
    [
        # Balance sheet
        ("BS", "ifrs-full_Assets", "total_assets"),
        ("BS", "ifrs-full_Liabilities", "total_liabilities"),
        ("BS", "ifrs-full_Equity", "total_equity"),
        ("BS", "ifrs-full_CurrentAssets", "current_assets"),
        ("BS", "ifrs-full_CurrentLiabilities", "current_liabilities"),
        ("BS", "ifrs-full_CashAndCashEquivalents", "cash_and_cash_equivalents"),
        ("BS", "ifrs-full_Inventories", "inventory"),

        # Income statement / comprehensive income statement
        ("CIS", "ifrs-full_Revenue", "revenue"),
        ("IS", "ifrs-full_Revenue", "revenue"),
        ("CIS", "dart_OperatingIncomeLoss", "operating_income"),
        ("IS", "dart_OperatingIncomeLoss", "operating_income"),
        ("CIS", "ifrs-full_ProfitLoss", "net_income"),
        ("IS", "ifrs-full_ProfitLoss", "net_income"),
        ("CIS", "ifrs-full_FinanceCosts", "finance_costs"),
        ("IS", "ifrs-full_FinanceCosts", "finance_costs"),

        # Cash flow statement
        ("CF", "ifrs-full_CashFlowsFromUsedInOperatingActivities", "cfo"),
    ],
    columns=["sj_div", "account_id", "std_account"],
)

fs_standardized_long = core_fs.merge(
    account_mapping,
    on=["sj_div", "account_id"],
    how="inner",
    validate="many_to_one",
)

print(fs_standardized_long["std_account"].value_counts())




# 6. Inspect duplicates and compare IS/CIS amounts
"""
Check for duplicate observations within the same statement before
comparing IS and CIS.

Compare amounts only when both statements have non-missing values.
Actual observation selection is performed in the next step.
"""

statement_priority = {"BS": 1, "CF": 1, "CIS": 1, "IS": 2}

fs_standardized_long["statement_priority"] = (
    fs_standardized_long["sj_div"].map(statement_priority)
)

fs_standardized_long_before_dedupe = fs_standardized_long.copy()


# Check duplicates within the same statement.
statement_keys = ["corp_code", "bsns_year", "std_account", "sj_div"]

within_statement_duplicates = fs_standardized_long_before_dedupe[
    fs_standardized_long_before_dedupe.duplicated(
        statement_keys,
        keep=False,
    )
].sort_values(statement_keys)

if not within_statement_duplicates.empty:
    print(
        within_statement_duplicates[
            statement_keys + ["account_id", "account_nm", "thstrm_amount"]
        ].to_string(index=False)
    )
    raise ValueError(
        "Review duplicate accounts within the same statement before comparison."
    )


def compare_is_cis(
    fs_long_before_dedupe: pd.DataFrame,
    std_account_name: str,
) -> pd.DataFrame:

    temp = fs_long_before_dedupe[
        fs_long_before_dedupe["std_account"].eq(std_account_name)
        & fs_long_before_dedupe["sj_div"].isin(["IS", "CIS"])
    ].copy()

    check = (
        temp.pivot(
            index=["corp_code", "bsns_year"],
            columns="sj_div",
            values="thstrm_amount",
        )
        .reindex(columns=["IS", "CIS"])
        .reset_index()
    )

    both_available = check[["IS", "CIS"]].notna().all(axis=1)
    check["diff"] = check["CIS"] - check["IS"]

    different_amounts = both_available & check["diff"].ne(0)

    print(f"\n=== {std_account_name} ===")
    print(f"Firm-years with both amounts: {both_available.sum()}")
    print(f"Firm-years with different amounts: {different_amounts.sum()}")

    if both_available.any():
        print(check.loc[both_available, "diff"].describe())

    if different_amounts.any():
        print(check.loc[different_amounts].to_string(index=False))

    return check


check_net_income = compare_is_cis(
    fs_standardized_long_before_dedupe, "net_income"
)
check_revenue = compare_is_cis(
    fs_standardized_long_before_dedupe, "revenue"
)
check_operating_income = compare_is_cis(
    fs_standardized_long_before_dedupe, "operating_income"
)
check_finance_costs = compare_is_cis(
    fs_standardized_long_before_dedupe, "finance_costs"
)




# 7. Select one observation per firm-year and analysis variable
"""
Prefer non-missing current-period amounts.
When availability is equal, apply statement priority.
Keep one row even when all candidate amounts are missing.
"""

fs_standardized_long = (
    fs_standardized_long_before_dedupe
    .assign(
        amount_missing=lambda df: df["thstrm_amount"].isna()
    )
    .sort_values(
        [
            "corp_code",
            "bsns_year",
            "std_account",
            "amount_missing",
            "statement_priority",
        ]
    )
    .drop_duplicates(
        subset=["corp_code", "bsns_year", "std_account"],
        keep="first",
    )
    .drop(columns=["amount_missing"])
    .copy()
)

print(fs_standardized_long["std_account"].value_counts())




# 8. Build the firm-year financial statement panel
credit_fs_wide = (
    fs_standardized_long
    .pivot(
        index=[
            "corp_code",
            "stock_code",
            "corp_name_master",
            "industry_krx",
            "bsns_year",
        ],
        columns="std_account",
        values="thstrm_amount",
    )
    .reset_index()
)

credit_fs_wide.columns.name = None

print(f"Panel shape: {credit_fs_wide.shape}")
print(credit_fs_wide.isna().sum().sort_values(ascending=False))




# 9. Calculate core financial ratios
"""
Calculate ratios only when denominators are positive.
Preserve negative numerators and missing values.

Definitions:
    - ROA and ROE use year-end assets and equity.
    - Quick ratio uses current assets less inventory.
    - Finance-cost coverage uses operating income / finance costs,
      not a pure interest coverage ratio.
"""

# Keep the original financial amounts unchanged.
assets = credit_fs_wide["total_assets"].where(
    credit_fs_wide["total_assets"] > 0
)
liabilities = credit_fs_wide["total_liabilities"].where(
    credit_fs_wide["total_liabilities"] > 0
)
equity = credit_fs_wide["total_equity"].where(
    credit_fs_wide["total_equity"] > 0
)
current_liabilities = credit_fs_wide["current_liabilities"].where(
    credit_fs_wide["current_liabilities"] > 0
)
revenue = credit_fs_wide["revenue"].where(
    credit_fs_wide["revenue"] > 0
)
finance_costs = credit_fs_wide["finance_costs"].where(
    credit_fs_wide["finance_costs"] > 0
)

credit_fs_wide["liabilities_to_assets"] = (
    credit_fs_wide["total_liabilities"] / assets
)

credit_fs_wide["equity_ratio"] = (
    credit_fs_wide["total_equity"] / assets
)

credit_fs_wide["current_ratio"] = (
    credit_fs_wide["current_assets"] / current_liabilities
)

credit_fs_wide["quick_ratio"] = (
    (credit_fs_wide["current_assets"] - credit_fs_wide["inventory"])
    / current_liabilities
)

credit_fs_wide["cash_ratio"] = (
    credit_fs_wide["cash_and_cash_equivalents"] / current_liabilities
)

credit_fs_wide["roa"] = credit_fs_wide["net_income"] / assets

credit_fs_wide["roe"] = credit_fs_wide["net_income"] / equity

credit_fs_wide["operating_margin"] = (
    credit_fs_wide["operating_income"] / revenue
)

credit_fs_wide["finance_cost_coverage"] = (
    credit_fs_wide["operating_income"] / finance_costs
)

credit_fs_wide["cfo_to_assets"] = credit_fs_wide["cfo"] / assets

credit_fs_wide["cfo_to_liabilities"] = credit_fs_wide["cfo"] / liabilities




# 10. Summarize financial ratios
"""
Inspect ratio distributions and missingness after denominator treatment.
Extreme ratios are not automatically treated as errors.
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
    "cfo_to_liabilities",
]

ratio_summary = credit_fs_wide[ratio_cols].describe().T
ratio_summary["missing_count"] = (
    credit_fs_wide[ratio_cols].isna().sum()
)

print(ratio_summary)




# 11. Check balance sheet consistency
"""
Check the accounting identity: assets = liabilities + equity.

bs_check:
    Signed difference in reported monetary units.

bs_check_relative:
    Absolute difference divided by positive total assets.

Missing inputs remain missing and are not treated as successful checks.
"""

credit_fs_wide["bs_check"] = (
    credit_fs_wide["total_assets"]
    - credit_fs_wide["total_liabilities"]
    - credit_fs_wide["total_equity"]
)

credit_fs_wide["bs_check_relative"] = (
    credit_fs_wide["bs_check"].abs()
    / credit_fs_wide["total_assets"].where(
        credit_fs_wide["total_assets"] > 0
    )
)

print("\nBalance sheet consistency:")
print(
    credit_fs_wide[
        ["bs_check", "bs_check_relative"]
    ].describe()
)

print(
    "Firm-years without a relative BS check:",
    credit_fs_wide["bs_check_relative"].isna().sum(),
)





# 12. Summarize denominator conditions
"""
Report missing, zero, and negative denominators separately.
Non-positive denominators were excluded from ratio calculations above.
"""

denominator_cols = [
    "total_assets",
    "total_liabilities",
    "total_equity",
    "current_liabilities",
    "revenue",
    "finance_costs",
]

denominator_check = pd.DataFrame({
    "missing_count": credit_fs_wide[denominator_cols].isna().sum(),
    "zero_count": credit_fs_wide[denominator_cols].eq(0).sum(),
    "negative_count": credit_fs_wide[denominator_cols].lt(0).sum(),
})

print("\nDenominator conditions:")
print(denominator_check)




# 13. Save the financial features for downstream analysis
output_path = DATA_DIR / "credit_features.csv"

credit_fs_wide.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nSaved to: {output_path}")