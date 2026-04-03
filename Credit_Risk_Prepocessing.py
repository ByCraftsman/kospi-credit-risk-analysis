"""
KOSPI common stocks only
Non-financial firms only
Top 100 by market capitalization
Universe fixed as of January 2026

Fiscal years 2020 ~ 2025

Annual reports only
"""

import pandas as pd

"""
The sample was restricted to KOSPI-listed common stocks as of 2026-01-02. 
REITs and financial-sector firms were excluded to improve comparability of accounting-based credit risk indicators. 
The final non-financial universe was defined as the top 100 firms by market capitalization.
"""

#2026-01-02 한국거래소 업종분류 엑셀 파일
krx_kospi = pd.read_excel(r"C:\Users\minec\OneDrive\바탕 화면\KOSPI 2026-01-02.xlsx")
krx_kospi['종목코드'] = krx_kospi['종목코드'].astype(str).str.zfill(6)
krx_kospi = krx_kospi[krx_kospi['종목코드'].str[-1:] == '0']
krx_kospi = krx_kospi[~krx_kospi['종목명'].str.endswith('리츠', na=False)]
exclude_industries = ['기타금융', '보험', '은행', '증권', '부동산']
krx_kospi = krx_kospi[~krx_kospi['업종명'].isin(exclude_industries)]
krx_kospi = krx_kospi.sort_values('시가총액', ascending=False).head(100)
krx_kospi = krx_kospi.reset_index(drop=True)
krx_kospi = krx_kospi[['종목코드', '종목명', '업종명', '시가총액']]


#DART 재무제표 2020~2025 다운로드 및 전처리
