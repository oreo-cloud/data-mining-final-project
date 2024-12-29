import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pokemon = pd.read_csv('pokedex.csv', encoding='big5')

# 編碼 English Name 欄位
label_encoder = LabelEncoder()
pokemon['English Name Encoded'] = label_encoder.fit_transform(pokemon['English Name'])

# 定義特徵與目標值
features = [col for col in pokemon.columns if col not in [
    'Image', 'Index', 'English Name', 'Chinese name', 'Total',
    'HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed', 'Legendary'
]]
# features.append('English Name Encoded')

targets = ['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']



# 儲存結果
xgb_results = {}

for target in targets:
    # 定義目標值
    X = pokemon[features]
    y = pokemon[target]
    
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化 XGBoost 模型
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0.1, 1.0, 10.0]
    }
    
    # 使用 GridSearchCV 搜索最佳參數
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
        n_jobs=1  # 禁用並行處理
    )
    grid_search.fit(X_train, y_train)
    
    # 最佳參數與模型
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # 預測與評估
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # 儲存結果
    xgb_results[target] = {
        'best_params': best_params,
        'mse': mse,
        'predictions': y_pred[:5]  # 儲存前五個預測值
    }

# 輸出結果
for target, result in xgb_results.items():
    print(f"{target}:")
    print(f"  Best Parameters: {result['best_params']}")
    print(f"  Mean Squared Error: {result['mse']:.2f}")
    print(f"  Predictions: {result['predictions']}")
    print()
