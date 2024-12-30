import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# 讀取CSV數據
file_path = 'pokedex.csv'
pokemon = pd.read_csv('pokedex.csv', encoding='big5')

# 檢查image欄位的圖片路徑
image_dir = '.'  # 假設圖片存儲在這個資料夾中
pokemon['image_path'] = pokemon['Image'].apply(lambda x: os.path.join(image_dir, x) if isinstance(x, str) else None)
invalid_paths = pokemon['image_path'].apply(lambda x: not os.path.exists(x) if isinstance(x, str) else True).sum()
print(f"無效圖片路徑數量: {invalid_paths}")

# 定義圖片展平函數
def flatten_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert('RGB')  # 轉換為 RGB 圖片
        img = img.resize(target_size)  # 調整大小
        img_array = np.array(img)  # 轉換為數組
        flattened = img_array.flatten()  # 展平為一維
        return flattened
    except Exception as e:
        return np.zeros((target_size[0] * target_size[1] * 3,))  # 如果圖片無法載入，返回零向量

# 提取所有圖片的展平特徵
pokemon['image_features'] = pokemon['image_path'].apply(lambda x: flatten_image(x))

print("Finished flattening images...")

# 展開圖片特徵並與數據框合併
image_features = np.vstack(pokemon['image_features'].values)
image_features_df = pd.DataFrame(image_features, columns=[f'pixel_{i}' for i in range(image_features.shape[1])])
pokemon = pd.concat([pokemon.reset_index(drop=True), image_features_df], axis=1)

# 移除不需要的欄位，確保所有特徵為數值型
features = [col for col in pokemon.columns if col not in [
    'Image', 'Index', 'English Name', 'Chinese name', 'Total', 'HP', 'Attack',
    'Defense', 'SP. Atk.', 'SP. Def', 'Speed', 'Legendary', 'image_path', 'image_features'
]]

X = pokemon[features]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
targets = ['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']

# 使用 XGBoost 進行訓練與預測
xgb_final_results_with_flattened_images = {}

# 每個目標值的最佳參數
best_params_per_target = {
    'HP': {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 300, 'reg_alpha': 0, 'reg_lambda': 10.0, 'subsample': 0.8},
    'Attack': {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 10.0, 'subsample': 0.8},
    'Defense': {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 1.0, 'subsample': 0.6},
    'SP. Atk.': {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 10.0, 'subsample': 0.8},
    'SP. Def': {'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 200, 'reg_alpha': 1.0, 'reg_lambda': 1.0, 'subsample': 0.6},
    'Speed': {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'reg_alpha': 1.0, 'reg_lambda': 1.0, 'subsample': 0.6}
}

# 使用最佳參數進行預測
for target in targets:
    print(f"Start predicting {target}...")
    y = pokemon[target].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用對應的最佳參數初始化 XGBoost 模型
    best_params = best_params_per_target[target]
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    xgb_model.fit(X_train, y_train)
    
    # 預測與評估
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # 儲存結果
    xgb_final_results_with_flattened_images[target] = {
        'rmse': rmse,
        'predictions': y_pred[:5]
    }

# 顯示結果
for target, result in xgb_final_results_with_flattened_images.items():
    print(f"{target}:")
    print(f"RMSE: {result['rmse']}")
    print(f"Predictions: {result['predictions']}")
    print()
