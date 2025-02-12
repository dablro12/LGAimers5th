import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import pandas as pd 
# def vis_class_check(class_array:np.array):
    
    
def vis_numeric_corr_matrix(df:pd.DataFrame):
    # 숫자형 칼럼만 선택
    numeric_df = df.select_dtypes(include=['number'])

    # '임신 성공 여부' 칼럼을 첫 번째로 이동
    if '임신 성공 여부' in numeric_df.columns:
        columns = ['임신 성공 여부'] + [col for col in numeric_df.columns if col != '임신 성공 여부']
        numeric_df = numeric_df[columns]
        
    # 숫자형 칼럼의 상관계수 계산
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(40, 40))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

    plt.title("Correlation Matrix of Numerical Features")
    plt.show()
    
import numpy as np
import pandas as pd

def find_inf_nan_columns(df: pd.DataFrame):
    """
    입력받은 df에서
    1) ±무한대(np.inf, -np.inf)가 들어있는 컬럼 목록
    2) NaN이 들어있는 컬럼 목록
    을 각각 반환합니다.
    """
    inf_cols = df.columns[df.replace([np.inf, -np.inf], np.nan).isna().any()].tolist()
    nan_cols = df.columns[df.isna().any()].tolist()
    
    return {
        "inf_col" : inf_cols, 
        "nan_col" : nan_cols
    }


def vis_prob_barchart(prob_array :np.array):
    plt.figure(figsize=(8, 4))
    plt.hist(prob_array, bins=50)  # bins 개수는 상황에 맞게 조절
    plt.title('Histogram of Probability')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.show()

# 사용 예시
# inf_cols, nan_cols = find_inf_nan_columns(df)
# print("Inf(±∞) 포함 컬럼:", inf_cols)
# print("NaN 포함 컬럼:", nan_cols)
