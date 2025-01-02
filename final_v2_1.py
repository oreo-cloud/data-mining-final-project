# HP defense speed 做分屬性的預測
import pandas as pd
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# 讀取CSV數據
file_path = 'pokedex.csv'
pokemon = pd.read_csv(file_path, encoding='big5')

# 檢查image欄位的圖片路徑
image_dir = '.'  # 假設圖片存儲在這個資料夾中
pokemon['image_path'] = pokemon['Image'].apply(lambda x: os.path.join(image_dir, x) if isinstance(x, str) else None)
invalid_paths = pokemon['image_path'].apply(lambda x: not os.path.isfile(x) if x else True).sum()
print(f"無效圖片路徑數量: {invalid_paths}")

# 加載預訓練的ResNet50模型，並移除全連接層以獲取特徵向量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_model.fc = torch.nn.Identity()  # 移除全連接層
resnet_model = resnet_model.to(device)
resnet_model.eval()

# 定義圖片轉換和特徵提取函數
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_image_features_pytorch(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = resnet_model(img_tensor)
        return features.cpu().numpy().flatten()
    except Exception as e:
        return np.zeros((2048,))  # 如果圖片無法載入，返回零向量

print(f"Start extracting image features...")
# 提取所有圖片特徵
pokemon['image_features'] = pokemon['image_path'].apply(extract_image_features_pytorch)

# 展開圖片特徵並與數據框合併
image_features = np.vstack(pokemon['image_features'].values)
image_features_df = pd.DataFrame(image_features, columns=[f'img_feat_{i}' for i in range(image_features.shape[1])])
pokemon = pd.concat([pokemon.reset_index(drop=True), image_features_df], axis=1)

# 對 English Name 進行 Label Encoding
print("Encoding English Name...")
label_encoder = LabelEncoder()
pokemon['English_Name_Label'] = label_encoder.fit_transform(pokemon['English Name'].fillna('Unknown'))

# 確保 Legendary 為數值型
pokemon['Legendary'] = pokemon['Legendary'].apply(lambda x: 1 if x else 0)

# 分割資料集
targets = ['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']
train_set, test_set = train_test_split(pokemon, test_size=0.2, random_state=42)

# 提取特徵和目標
features = [col for col in train_set.columns if col not in [
    'Image', 'Index', 'English Name', 'Chinese name', 'Total', 'HP', 'Attack',
    'Defense', 'SP. Atk.', 'SP. Def', 'Speed', 'image_path', 'image_features'
]]

X_train = train_set[features].apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = test_set[features].apply(pd.to_numeric, errors='coerce').fillna(0)

final = pd.DataFrame({
    'English Name': test_set['English Name'].reset_index(drop=True),
    'Chinese Name': test_set['Chinese name'].reset_index(drop=True)
})

# 定義最佳超參數
best_params_all_targets = {
    'HP': {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.8},
    'Attack': {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8},
    'Defense': {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8},
    'SP. Atk.': {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8},
    'SP. Def': {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 100, 'subsample': 0.8},
    'Speed': {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.6}
}

# 儲存最終結果
final_results = {}

# 逐一目標使用最佳超參數進行模型訓練
for target in targets:
    print(f"Training final model for {target} with best parameters...")
    y_train = train_set[target].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_test = test_set[target].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 使用該目標的最佳參數初始化 XGBoost 模型
    best_params = best_params_all_targets[target]
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',  # 確保使用 GPU
        random_state=42,
        **best_params
    )

    # 訓練模型
    model.fit(X_train, y_train)

    # 預測與評估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    percentage_rmse = (rmse / (train_set[target].max() - train_set[target].min())) * 100 if (train_set[target].max() - train_set[target].min()) != 0 else float('inf')

    print(f"Final %RMSE for {target}: {percentage_rmse}%")

    # 儲存結果
    final_results[target] = {
        'rmse': rmse,
        'percentage_rmse': percentage_rmse,
        'predictions': y_pred[:5]  # 儲存前五個預測值
    }

    final[f'{target}_Predict'] = y_pred
    final[f'{target}_Actual'] = y_test.reset_index(drop=True)

# 合併所有預測與實際值並輸出為CSV
final.to_csv('predictions_with_actuals.csv', index=False, encoding='utf-8-sig')
print("Predictions and actual values have been saved to 'predictions_with_actuals.csv'")

# 輸出最終結果  
for target, results in final_results.items():
    print(f"Final %RMSE for {target}: {results['percentage_rmse']}%")
    print(f"Predictions for {target}: {results['predictions']}")