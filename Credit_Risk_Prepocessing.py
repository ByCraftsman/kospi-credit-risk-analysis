"""
Credit Risk Preprocessing

Sample construction:
- KOSPI-listed common stocks only
- Non-financial firms only
- Initial universe: top 100 firms by market capitalization
- Universe fixed as of 2026-01-02

Financial statement coverage:
- Fiscal years: 2019–2025
- Annual reports only
- Consolidated financial statements (CFS)

Final analysis sample:
- 83 firms with complete annual financial statement coverage over 2019–2025
"""




#2026-01-02 한국거래소 업종분류 엑셀 파일
import pandas as pd
krx_kospi = pd.read_excel(r"C:\Users\minec\OneDrive\바탕 화면\KOSPI 2026-01-02.xlsx")
krx_kospi['종목코드'] = krx_kospi['종목코드'].astype(str).str.zfill(6)
krx_kospi = krx_kospi[krx_kospi['종목코드'].str[-1:] == '0']
krx_kospi = krx_kospi[~krx_kospi['종목명'].str.endswith('리츠', na=False)]
exclude_industries = ['기타금융', '보험', '은행', '증권', '부동산']
krx_kospi = krx_kospi[~krx_kospi['업종명'].isin(exclude_industries)]
krx_kospi = krx_kospi.sort_values('시가총액', ascending=False).head(100)
krx_kospi = krx_kospi.reset_index(drop=True)
krx_kospi = krx_kospi[['종목코드', '업종명', '시가총액']]




#DART 재무제표 2019~2025 다운로드 및 전처리
import keyring 
import requests
import zipfile
import io
import xml.etree.ElementTree as ET

dart_api_key = keyring.get_password("DART_API", "bycraftsman")


#Download and parse corpCode.xml
def get_corp_code_table(api_key: str) -> pd.DataFrame:
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
            "modify_date": item.findtext("modify_date")
        })

    corp_df = pd.DataFrame(rows)
    corp_df['stock_code'] = corp_df['stock_code'].astype(str).str.zfill(6)

    # Keep listed firms only
    corp_df = corp_df[corp_df['stock_code'].notna()]
    corp_df = corp_df[corp_df['stock_code'] != 'None']
    
    if dart_api_key is None:
        raise ValueError("DART API key not found in keyring.")

    return corp_df

corp_df = get_corp_code_table(dart_api_key)




firm_master = krx_kospi.merge(
    corp_df[['stock_code', 'corp_name', 'corp_code']],
    left_on='종목코드',
    right_on='stock_code',
    how='left'
)

firm_master = firm_master.drop(columns=['stock_code'])
firm_master[['corp_name', 'corp_code']].isna().sum()





#선정된 시가총액 상위 100개 기업 재무제표 (2019~2025년, 7년치)
import time
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

    if status != "000":
        return pd.DataFrame([{
            "corp_code": corp_code,
            "bsns_year": year,
            "status": status,
            "message": data.get("message"),
            "fs_div_requested": fs_div
        }])

    df = pd.DataFrame(data.get("list", []))
    if not df.empty:
        df["corp_code"] = corp_code
        df["bsns_year"] = year
        df["fs_div_requested"] = fs_div

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
        df_year = fetch_full_fs(dart_api_key, corp_code, year, reprt_code="11011", fs_div="CFS")

        if not df_year.empty:
            df_year["stock_code"] = stock_code
            df_year["corp_name_master"] = corp_name
            df_year["industry_krx"] = industry
            df_year["market_cap_krx"] = market_cap

            raw_results.append(df_year)

        time.sleep(0.5)


raw_fs = pd.concat(raw_results, ignore_index=True) if raw_results else pd.DataFrame()




#재무제표 데이터 전처리
raw_fs[raw_fs["status"].notna()][["corp_name_master", "stock_code", "bsns_year", "status", "message"]]
raw_fs_error = raw_fs[raw_fs["sj_div"].isna()].copy()
raw_fs_error = raw_fs_error[["corp_name_master", "stock_code", "corp_code"]].drop_duplicates()
exclude_codes = raw_fs_error["corp_code"].unique()

firm_master_final = firm_master[~firm_master["corp_code"].isin(exclude_codes)].copy()
raw_fs_valid = raw_fs[raw_fs["sj_div"].notna()].copy()
raw_fs_final = raw_fs_valid[~raw_fs_valid["corp_code"].isin(exclude_codes)].copy()


#불필요한 열 제거
raw_fs_final = raw_fs_final.drop(
    columns=[
        "rcept_no",
        "reprt_code",
        "status",
        "message",
        "currency",
        "fs_div_requested",
        "thstrm_nm",
        "frmtrm_nm",
        "bfefrmtrm_nm",
        "thstrm_add_amount"
    ],
    errors="ignore"
)

# 열 이름 목록 확인
column_names = raw_fs_final.columns.tolist()
column_count = len(raw_fs_final.columns)

print(f"열 이름: {column_names}")
print(f"열 개수: {column_count}개")


#데이터 타입 처리
print(raw_fs_final.dtypes)
amount_cols = ["thstrm_amount", "frmtrm_amount", "bfefrmtrm_amount"]

for col in amount_cols:
    raw_fs_final[col] = (
        raw_fs_final[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("None", pd.NA)
    )
    raw_fs_final[col] = pd.to_numeric(raw_fs_final[col], errors="coerce")
        

#분석용 테이블 
raw_fs_slim = raw_fs_final[
    [
        "bsns_year",
        "corp_code",
        "stock_code",
        "corp_name_master",
        "industry_krx",
        "sj_div",
        "account_nm",
        "thstrm_amount",
        "frmtrm_amount"
    ]
].copy()

print(raw_fs_slim.dtypes)
print(raw_fs_slim[["thstrm_amount", "frmtrm_amount"]].isna().sum())
print(f"Final number of firms: {firm_master_final['corp_code'].nunique()}")




#데이터 저장하기
firm_master_final.to_csv("credit_firm_master_final.csv", index=False, encoding="utf-8-sig")
raw_fs_final.to_csv("credit_raw_fs_final.csv", index=False, encoding="utf-8-sig")
raw_fs_slim.to_csv("credit_raw_fs_slim.csv", index=False, encoding="utf-8-sig")

#저장 경로 확인
import os
print(os.getcwd())
