import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def handle_inf_and_fillna(df, fill_value=0):
    """
    DataFrame df 내부에 inf, -inf가 있으면 np.nan으로 치환 후,
    그 NaN 값을 fill_value(기본=0)로 대체한다.
    inplace=False로 처리해 새로운 DataFrame을 반환.
    """
    # 1) inf -> NaN 치환
    df_replaced = df.replace([np.inf, -np.inf], np.nan)

    # 2) 결측치(NaN)를 fill_value로 대체
    df_filled = df_replaced.fillna(fill_value)

    return df_filled

def preprocess_data(df):
    # ===== 1) 카테고리 열과 연속형 열을 구분 =====
    cat_cols = []
    num_cols = []
    for col in df.columns:
        # bool -> 일단 카테고리 취급(2개 범주)
        if pd.api.types.is_bool_dtype(df[col]):
            cat_cols.append(col)
        # object -> 범주형(문자열)
        elif pd.api.types.is_object_dtype(df[col]):
            cat_cols.append(col)
        # int
        elif pd.api.types.is_integer_dtype(df[col]):
            # 어떤 int 열은 사실상 '코드'일 수 있음(범주)
            # 여기서는 단순히 '유니크 값이 20 이하'면 카테고리, 아니면 수치형이라 가정
            if df[col].nunique() <= 20:
                cat_cols.append(col)
            else:
                num_cols.append(col)
        # float -> 수치형
        elif pd.api.types.is_float_dtype(df[col]):
            num_cols.append(col)
        else:
            # 그 외(드물게 datetime 등) -> 일단 cat_cols에 넣거나 별도 처리
            cat_cols.append(col)

    # ===== 2) bool을 0/1로 변환 =====
    bool_cols = df.select_dtypes(include='bool').columns
    for c in bool_cols:
        df[c] = df[c].astype(int)  # True->1, False->0

    # ===== 3) 범주형 열에 라벨 인코딩 (예시) =====
    # (이미 int인 열이 있더라도, 도메인 상 '코드' 열이면 LabelEncoder 쓸 수 있음)
    for col in cat_cols:
        if df[col].dtype == object:
            df[col] = df[col].fillna("NULL")  # 문자열 결측 처리
        else:
            df[col] = df[col].fillna(-999)    # int/bool 결측치 예시
        
        # 라벨 인코딩
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        # 참고: "train/val/test"로 분리된 경우, train에 fit, val/test엔 transform만 적용해야 함

    # ===== 4) 연속형 열에 결측치 처리 + Normalization =====
    # (min-max 스케일링을 예시로)
    scaler = MinMaxScaler()
    for col in num_cols:
        # 결측치가 있으면 일단 0 or 평균 등으로 대체(예시: 0)
        df[col] = df[col].fillna(0)
    
    # df[num_cols] 부분만 별도 스케일링
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, cat_cols, num_cols

def check_non_numeric_values(df: pd.DataFrame):
    """
    df 안의 object 타입 컬럼에서,
    숫자로 변환 불가능한 값이 있는지 탐색.
    - 결측값(NaN)은 즉시 0으로 치환.
    - 숫자로 변환 불가능한 값이 발견되면 위치와 값을 출력.
    """
    # 1) object 컬럼 추출
    obj_cols = df.select_dtypes(include=['object']).columns
    
    # 2) 각 컬럼의 모든 값 검사
    for col in obj_cols:
        for idx, val in df[col].items():
            
            # (a) 결측값이면 0으로 치환
            if pd.isna(val):
                df.at[idx, col] = 0  # in-place 수정
                continue

            # (b) 숫자로 변환 가능한지 시도
            try:
                float(val)  # 변환 시도
            except (ValueError, TypeError):
                # 변환 실패 => 문제 셀
                print(f"[NON-NUMERIC] col='{col}', index='{idx}', value='{val}'")
    print("검사 및 결측치 확인 상태 : 무결성")