from dotenv import load_dotenv
import os
load_dotenv()


import pandas as pd 
from sklearn.preprocessing import LabelEncoder

def get_feature(df, validation: bool):
    """
    Get the feature columns from the dataframe
    """
    columns_str = os.getenv("USE_COLUMNS")
    
    if columns_str:
        columns_list = [col.strip() for col in columns_str.split(",")]
    
    # 임신 성공 여부 제거
    if validation:
        columns_list.remove('임신 성공 여부')

    return df[columns_list]

def preprocess(train:pd.DataFrame, validation = False):
    #%% SETTING
    le = LabelEncoder()

    #%% 시술 시기 코드
    train['시술 시기 코드'] = le.fit_transform(train['시술 시기 코드'])
    
    #%% 시술 당시 나이
    train['시술 당시 나이'] = train['시술 당시 나이'].map({
    '만18-34세': 0,
    '만35-37세': 1,
    '만38-39세': 2,
    '만40-42세': 3,
    '만43-44세': 4,
    '만45-50세': 5,
    '알 수 없음' : 999
    })
    
    #%% 임신 시도 또는 마지막 임신 경과 연수
    train['임신 시도 또는 마지막 임신 경과 연수'] = train['임신 시도 또는 마지막 임신 경과 연수'].fillna(999)
    
    #%% 시술 유형
    train['시술 유형'] = train['시술 유형'].fillna(0)
    train['시술 유형'] = train['시술 유형'].map({'IVF': 1, 'DI': 2})
    
    #%%
    train['특정 시술 유형'] = train['특정 시술 유형'].fillna('0')
    train['특정 시술 유형'] = le.fit_transform(train['특정 시술 유형'])
    
    #%% 배란 유도 유형
    # 배란 유도 유형에서 세트로타이드, 생식선 자극 호르몬 제거  -> 소수 데이터 sampling
    train = train[train['배란 유도 유형'] != '세트로타이드 (억제제)']
    train = train[train['배란 유도 유형'] != '생식선 자극 호르몬']

    # 기록되지 않은 시행 -> 1 / 알 수 없음 -> 0
    train['배란 유도 유형'] = train['배란 유도 유형'].map({
        '기록되지 않은 시행' : 1,
        '알 수 없음' : 0
    })

    #%% 단일 배아 이식 여부
    train['단일 배아 이식 여부'] = train['단일 배아 이식 여부'].fillna(0)
    
    #%% 착상 전 유전 검사 사용 여부
    train['착상 전 유전 검사 사용 여부'] = train['착상 전 유전 검사 사용 여부'].fillna(0)
    
    #%% 착상 전 유전 진단 사용 여부
    train['착상 전 유전 진단 사용 여부'] = train['착상 전 유전 진단 사용 여부'].fillna(999) # -> 
    
    #%% 배아 생성 주요 이유
    train['배아 생성 주요 이유'] = train['배아 생성 주요 이유'].fillna('0')

    # train['배아 생성 주요 이유'] = train['배아 생성 주요 이유'].str.split(',')
    # train = train.explode('배아 생성 주요 이유') # 여러 행 분리 
    # train['배아 생성 주요 이유'] = train['배아 생성 주요 이유'].str.strip() # 좌우 공백 제거

    train['배아 생성 주요 이유'] = le.fit_transform(train['배아 생성 주요 이유'])
    
    #%% 총 시술 횟수
    train['총 시술 횟수'] = train['총 시술 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })

    #%% 클리닉 내 총 시술 횟수
    train['클리닉 내 총 시술 횟수'] = train['클리닉 내 총 시술 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })
    #%% IVF 시술 횟수
    train['IVF 시술 횟수'] = train['IVF 시술 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })

    #%% DI 시술 횟수
    train['DI 시술 횟수'] = train['DI 시술 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })

    #%% 총 임신 횟수
    train['총 임신 횟수'] = train['총 임신 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })

    #%% IVF 임신 횟수 
    train['IVF 임신 횟수'] = train['IVF 임신 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })
    #%% DI 임신 횟수
    train['DI 임신 횟수'] = train['DI 임신 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })
    #%% 총 출산 횟수
    train['총 출산 횟수'] = train['총 출산 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })
    
    #%% IVF 출산 횟수
    train['IVF 출산 횟수'] = train['IVF 출산 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })
    
    #%% DI 출산 횟수
    train['DI 출산 횟수'] = train['DI 출산 횟수'].map({
        '0회' : 0,
        '1회' : 1,
        '2회' : 2,
        '3회' : 3,
        '4회' : 4,
        '5회' : 5,
        '6회 이상' : 6,
    })
    
    #%% 총 생성 배아 수
    train['총 생성 배아 수'] = train['총 생성 배아 수'].fillna(999)

    #%% 미세주입된 난자 수 
    train['미세주입된 난자 수'] = train['미세주입된 난자 수'].fillna(999)
    
    #%% 미세주입에서 생성된 배아 수 
    train['미세주입에서 생성된 배아 수'] = train['미세주입에서 생성된 배아 수'].fillna(999)

    #%% 이식된 배아 수
    train['이식된 배아 수'] = train['이식된 배아 수'].fillna(999)
    
    #%% 미세주입 배아 이식 수 
    train['미세주입 배아 이식 수'] = train['미세주입 배아 이식 수'].fillna(999)

    #%% 혼합된 난자 수
    train['혼합된 난자 수'] = train['혼합된 난자 수'].fillna(999)

    #%% 파트너 정자와 혼합된 난자 수
    train['파트너 정자와 혼합된 난자 수'] = train['파트너 정자와 혼합된 난자 수'].fillna(999)
    
    #%% 기증자 정자와 혼합된 난자 수 -> 6291개는 제거하는게 좋을 것 같음
    if not validation:
        train = train[train['기증자 정자와 혼합된 난자 수'].notnull()]
        
    #%% 난자 기증자 나이
    train['난자 기증자 나이'] = train['난자 기증자 나이'].map({
        '알 수 없음' : 999,
        '만20세 이하' : 0,
        '만21-25세' : 1,
        '만26-30세' : 2,
        '만31-35세' : 3,
    })
    
    #%% 정자 기증자 나이
    train['정자 기증자 나이'] = train['정자 기증자 나이'].map({
        '알 수 없음' : 999,
        '만20세 이하' : 0,
        '만21-25세' : 1,
        '만26-30세' : 2,
        '만31-35세' : 3,
        '만36-40세' : 4,
        '만41-45세' : 5,
    })
    #%% 동결 배아 사용 여부
    train['동결 배아 사용 여부'] = train['동결 배아 사용 여부'].fillna(999)
    
    #%% 신선 배아 사용 여부
    train['신선 배아 사용 여부'] = train['신선 배아 사용 여부'].fillna(999)

    #%% 난자 해동 경과일
    train['난자 해동 경과일'] = train['난자 해동 경과일'].fillna(999)
    
    #%% 난자 혼합 경과일
    train['난자 혼합 경과일'] = train['난자 혼합 경과일'].fillna(999)
    
    #%% 배아 이식 경과일 
    train['배아 이식 경과일'] = train['배아 이식 경과일'].fillna(999)

    #%% 배아 해동 경과일 
    train['배아 해동 경과일'] = train['배아 해동 경과일'].fillna(999)
    

    #%% Column Select
    train = get_feature(train, validation)

    return train