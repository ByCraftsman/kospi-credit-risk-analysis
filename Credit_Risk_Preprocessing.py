"""
Credit Risk Preprocessing

Purpose:
- Construct a fixed KOSPI universe from a KRX industry-classification snapshot.
- Collect annual consolidated financial statements from OpenDART.
- Build input datasets for financial statement mapping and credit risk screening.

Sample construction:
- Universe date: 2026-01-02
- Retain stock codes ending in "0" as the common-stock screening rule.
- Exclude stock names ending in "리츠".
- Exclude the following KRX industry categories:기타금융, 보험, 은행, 증권, 부동산
- Select the top 100 remaining stocks by market capitalization.

Financial statement coverage:
- Fiscal years: 2019–2025
- Annual reports only
- Consolidated financial statements (CFS)

Final sample:
- Retain firms with financial statement data for every required fiscal year.
- Determine the final firm count after collection and coverage validation.
- Complete annual coverage does not imply complete coverage of every account.

Scope:
- Retrospective financial vulnerability screening of a fixed universe.
- The industry exclusions also remove non-financial holding companies classified under 기타금융.
"""


import io
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import keyring
import pandas as pd
import requests


# 1. Set project paths
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)




# 2. Construct the KOSPI universe
krx_kospi = pd.read_excel(
    DATA_DIR / "KOSPI 2026-01-02.xlsx",
    dtype={"종목코드": "string"},
)

krx_kospi = krx_kospi[krx_kospi["종목코드"].str[-1:] == "0"]
krx_kospi = krx_kospi[
    ~krx_kospi["종목명"].str.endswith("리츠", na=False)
]

exclude_industries = ["기타금융", "보험", "은행", "증권", "부동산"]
krx_kospi = krx_kospi[
    ~krx_kospi["업종명"].isin(exclude_industries)
]

krx_kospi = krx_kospi.sort_values("시가총액", ascending=False).head(100)
krx_kospi = krx_kospi.reset_index(drop=True)
krx_kospi = krx_kospi[["종목코드", "업종명", "시가총액"]]




# 3. Load the API key and retrieve DART corporation codes
dart_api_key = keyring.get_password("DART_API", "bycraftsman")


def get_corp_code_table(api_key: str) -> pd.DataFrame:

    if not api_key:
        raise ValueError("DART API key not found in keyring.")

    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": api_key}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        file_name = zf.namelist()[0]
        xml_bytes = zf.read(file_name)

    root = ET.fromstring(xml_bytes)

    rows = []

    for item in root.findall("list"):
        rows.append({
            "corp_code": item.findtext("corp_code"),
            "corp_name": item.findtext("corp_name"),
            "stock_code": item.findtext("stock_code"),
            "modify_date": item.findtext("modify_date"),
        })

    corp_df = pd.DataFrame(rows)

    corp_df["stock_code"] = (corp_df["stock_code"].astype("string").str.strip())

    corp_df = corp_df[corp_df["stock_code"].notna() & corp_df["stock_code"].ne("")].copy()

    return corp_df

corp_df = get_corp_code_table(dart_api_key)




# 4. Match the selected firms to DART corporation codes
firm_master = krx_kospi.merge(
    corp_df[["stock_code", "corp_name", "corp_code"]],
    left_on="종목코드",
    right_on="stock_code",
    how="left",
    validate="one_to_one",
)
firm_master = firm_master.drop(columns=["stock_code"])


unmatched = firm_master[firm_master["corp_code"].isna()]

if not unmatched.empty:
    print(unmatched)
    raise ValueError("Some selected stocks could not be matched to DART.")




# 5. Collect annual consolidated financial statements
def fetch_full_fs(api_key: str, corp_code: str, year: int,
                  reprt_code: str = "11011", fs_div: str = "CFS") -> pd.DataFrame:
    
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "fs_div": fs_div
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    status = data.get("status")

    # 조회된 자료가 없는 경우: 기록을 남기고 다음 요청 진행
    if status == "013":
        return pd.DataFrame([{
            "corp_code": corp_code,
            "bsns_year": year,
            "status": status,
            "message": data.get("message"),
            "fs_div_requested": fs_div,
        }])

    # 그 외 API 오류: 수집 중단
    if status != "000":
        raise RuntimeError(
            f"DART API error: corp_code={corp_code}, year={year}, "
            f"status={status}, message={data.get('message')}"
        )

    # 정상 응답을 DataFrame으로 변환
    df = pd.DataFrame(data.get("list", []))

    if df.empty:
        raise ValueError(
            f"Empty financial statement data despite status 000: "
            f"corp_code={corp_code}, year={year}"
        )

    df["corp_code"] = corp_code
    df["bsns_year"] = year
    df["fs_div_requested"] = fs_div
    df["status"] = status

    return df


years = range(2019, 2026)
raw_results = []

for _, row in firm_master.iterrows():
    corp_code = row["corp_code"]
    stock_code = row["종목코드"]
    corp_name = row["corp_name"]
    industry = row["업종명"]
    market_cap = row["시가총액"]

    for year in years:
        print(f"Fetching: {corp_name} ({stock_code}), {year}")

        df_year = fetch_full_fs(
            dart_api_key,
            corp_code,
            year,
            reprt_code="11011",
            fs_div="CFS",
        )

        df_year["stock_code"] = stock_code
        df_year["corp_name_master"] = corp_name
        df_year["industry_krx"] = industry
        df_year["market_cap_krx"] = market_cap

        raw_results.append(df_year)

        time.sleep(0.5)

raw_fs = (pd.concat(raw_results, ignore_index=True) if raw_results else pd.DataFrame())




# 6. Validate annual coverage and select the final sample
if "message" not in raw_fs.columns:
    raw_fs["message"] = pd.NA

missing_fs = raw_fs.loc[raw_fs["status"].eq("013"),
    [
        "corp_name_master",
        "stock_code",
        "corp_code",
        "bsns_year",
        "status",
        "message",
    ],
].copy()

print(missing_fs)
print(f"Missing firm-year observations: {len(missing_fs)}")
print(f"Firms with missing annual data: {missing_fs['corp_code'].nunique()}")

# 한 연도 이상 자료가 없는 기업의 코드
exclude_codes = missing_fs["corp_code"].unique()

# 7개년 자료를 모두 확보한 기업 목록
firm_master_final = firm_master[
    ~firm_master["corp_code"].isin(exclude_codes)
].copy()

# 해당 기업들의 정상 재무제표만 선택
raw_fs_final = raw_fs[
    raw_fs["status"].eq("000")
    & ~raw_fs["corp_code"].isin(exclude_codes)
].copy()

print(f"Final number of firms: {len(firm_master_final)}")

year_counts = raw_fs_final.groupby("corp_code")["bsns_year"].nunique()

print(year_counts.value_counts())


column_names = raw_fs_final.columns.tolist()
column_count = len(column_names)

print(f"열 이름: {column_names}")
print(f"열 개수: {column_count}개")





# 7. Convert financial amounts to numeric values
amount_cols = ["thstrm_amount", "frmtrm_amount", "bfefrmtrm_amount"]

for col in amount_cols:
    cleaned = (
        raw_fs_final[col]
        .astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace("", pd.NA)
    )

    converted = pd.to_numeric(cleaned, errors="coerce")

    # 값이 있었지만 숫자로 변환되지 않은 경우 확인
    failed = cleaned.notna() & converted.isna()

    if failed.any():
        print(f"Unparsed values in {col}:")
        print(cleaned[failed].value_counts())
        raise ValueError(f"Check unparsed financial amounts in {col}.")

    raw_fs_final[col] = converted
    
    


# 8. Build the slim financial statement dataset
"""
Keep both account_id and account_nm:
    
    - account_id: because it is based on XBRL standard account IDs.
      XBRL (eXtensible Business Reporting Language) is an international standard
      for digital business and financial reporting.

    - account_nm is retained for human-readable validation.
""" 

raw_fs_slim = raw_fs_final[
    [
        "bsns_year",
        "corp_code",
        "stock_code",
        "corp_name_master",
        "industry_krx",
        "sj_div",
        "account_id",
        "account_nm",
        "thstrm_amount",
        "frmtrm_amount"
    ]
].copy()

print(raw_fs_slim.dtypes)
print(raw_fs_slim[["thstrm_amount", "frmtrm_amount"]].isna().sum())
print(f"Final number of firms: {firm_master_final['corp_code'].nunique()}")




# 9. Save the final datasets
raw_fs.to_csv(
    DATA_DIR / "credit_raw_fs.csv",
    index=False,
    encoding="utf-8-sig",
)

firm_master_final.to_csv(
    DATA_DIR / "credit_firm_master_final.csv",
    index=False,
    encoding="utf-8-sig",
)

raw_fs_slim.to_csv(
    DATA_DIR / "credit_raw_fs_slim.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"Saved to: {DATA_DIR}")