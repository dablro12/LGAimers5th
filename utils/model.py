import pandas as pd 
from tab_transformer_pytorch import FTTransformer
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

def ml_trainer(k_fold_model, X_train, y_train):
    for model in [k_fold_model]:
        model.fit(X_train, y_train)
    return k_fold_model

def tabfn_trainer(k_fold_model, X_train, y_train, batch_size =256):
    n_batches = (len(X_train) + batch_size - 1) // batch_size  # 올림계산
    for i in range(n_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, len(X_train))
        
        X_batch = X_train.iloc[start:end]
        y_batch = y_train.iloc[start:end]
        k_fold_model.fit(X_batch, y_batch, overwrite_warning=True)
        
    return k_fold_model

def bool_process(X):
    bool_cols = X.select_dtypes(include='bool').columns
    for c in bool_cols:
        X[c] = X[c].astype(int)  # True->1, False->0
    return X

def tabnet_trainer(k_fold_model, X_train, y_train, X_val, y_val):
    X_train = bool_process(X_train)
    X_val = bool_process(X_val)
    
    k_fold_model.fit(
        X_train = X_train.values,
        y_train = y_train.values,
        eval_set = [(X_val.values, y_val.values)],
        max_epochs = 100,
        batch_size = 1024,
        virtual_batch_size = 256,
        pin_memory = True,
        warm_start = False,
        num_workers = 16
    )
    return k_fold_model


#% FT Transfomer
def split_cat_num_cols(df, cat_unique_threshold=20):
    """
    df의 컬럼들을 '카테고리형'과 '연속형'으로 자동 분류하여 반환합니다.
    - bool, object, category 타입은 무조건 '카테고리형'
    - int 타입이면서 유니크 값이 cat_unique_threshold 이하라면 카테고리형
    - 나머지는 연속형
    """
    cat_cols = []
    num_cols = []
    
    for col in df.columns:
        col_dtype = df[col].dtype
        
        if pd.api.types.is_bool_dtype(col_dtype):
            # bool -> 카테고리 2개로 간주
            cat_cols.append(col)
        
        elif pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_categorical_dtype(col_dtype):
            # object or category -> 범주형
            cat_cols.append(col)
            
        elif pd.api.types.is_integer_dtype(col_dtype):
            # int인 경우, 유니크 값이 적으면 카테고리로 간주 (예: 20 이하)
            if df[col].nunique() <= cat_unique_threshold:
                cat_cols.append(col)
            else:
                num_cols.append(col)
                
        elif pd.api.types.is_float_dtype(col_dtype):
            # float -> 연속형
            num_cols.append(col)
            
        else:
            # 그 외(dtype을 정확히 파악하기 어려운 경우) -> 일단 연속형으로 분류
            num_cols.append(col)
    
    return cat_cols, num_cols

class TabularDataset(Dataset):
    def __init__(self, df_cat, df_num, labels):
        """
        df_cat: 카테고리 열만 모은 DataFrame (already label-encoded)
        df_num: 연속형 열만 모은 DataFrame
        labels: Series or array, shape (N,)
        """
        self.x_cat = torch.tensor(df_cat.values, dtype=torch.long)
        self.x_num = torch.tensor(df_num.values, dtype=torch.float)
        self.y = torch.tensor(labels.values, dtype=torch.float).view(-1, 1)  # (N, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_num[idx], self.y[idx]


def create_fttransformer_model(cat_cols, num_cols, X):
    """
    cat_cols: 카테고리 열 리스트
    num_cols: 연속형 열 리스트
    X: 전체 DataFrame (이미 label-encoded 된 상태)
    -> 각 카테고리 열의 nunique()를 구해 categories 리스트 생성
    -> FTTransformer 모델 반환
    """
    # 각 카테고리 열별 유니크 개수
    categories_sizes = [X[col].nunique() for col in cat_cols]
    # 연속형 열 개수
    num_continuous = len(num_cols)

    model = FTTransformer(
        categories = categories_sizes,  # 예: [10,5,6,5,8] 등
        num_continuous = num_continuous,
        dim = 32,      # 예시
        dim_out = 1,   # 이진분류라면 1
        depth = 6,
        heads = 8,
        attn_dropout = 0.1,
        ff_dropout = 0.1
    )
    return model


def ftt_trainer(X_train, y_train, X_val, y_val, cat_cols, num_cols, 
                learning_rate = 1e-4,
                epochs = 100,
                batch_size = 32,
    ):
    X_train = bool_process(X_train)
    X_val = bool_process(X_val)
    
    train_ds = TabularDataset(X_train[cat_cols], X_train[num_cols], y_train)
    val_ds   = TabularDataset(X_val[cat_cols],   X_val[num_cols],   y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    
    # 4-3) 모델 생성
    model = create_fttransformer_model(cat_cols, num_cols, X_train)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4-4) 손실함수/옵티마이저
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 4-5) 학습 루프
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for x_cat_batch, x_num_batch, y_batch in train_loader:
            x_cat_batch = x_cat_batch.to(device)
            x_num_batch = x_num_batch.to(device)
            y_batch = y_batch.to(device)

            # forward
            logits = model(x_cat_batch, x_num_batch)  # (batch_size, 1)
            loss = criterion(logits, y_batch)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)

        # 검증 셋 로스(또는 다른 메트릭) 계산
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat_batch, x_num_batch, y_batch in val_loader:
                x_cat_batch = x_cat_batch.to(device)
                x_num_batch = x_num_batch.to(device)
                y_batch = y_batch.to(device)

                val_logits = model(x_cat_batch, x_num_batch)
                v_loss = criterion(val_logits, y_batch)
                val_loss += v_loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model

from ts3l.pl_modules import SCARFLightning
from ts3l.utils.scarf_utils import SCARFDataset, SCARFConfig
from ts3l.utils import TS3LDataModule
from ts3l.utils.embedding_utils import IdentityEmbeddingConfig
from ts3l.utils.backbone_utils import MLPBackboneConfig
from pytorch_lightning import Trainer
