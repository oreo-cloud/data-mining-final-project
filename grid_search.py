import pandas as pd
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
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

# 移除不需要的欄位，確保所有特徵為數值型
features = [col for col in pokemon.columns if col not in [
    'Image', 'Index', 'English Name', 'Chinese name', 'Total', 'HP', 'Attack',
    'Defense', 'SP. Atk.', 'SP. Def', 'Speed', 'Legendary', 'image_path', 'image_features'
]]
X = pokemon[features]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

targets = ['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']

# 定義超參數網格
param_grid = {
    'colsample_bytree': [0.6, 0.8],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 300],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1.0, 10.0],
    'subsample': [0.6, 0.8]
}

# 儲存結果
xgb_final_results_with_images = {}

# 逐一目標進行 Grid Search 和訓練
for target in targets:
    print(f"Start training {target} XGBoost models with Grid Search...")
    y = pokemon[target].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化 XGBoost 模型
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # 初始化 GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
    
    # 執行搜索
    grid_search.fit(X_train, y_train)
    
    # 取得最佳參數與模型
    best_params = grid_search.best_params_
    print(f"Best parameters for {target}: {best_params}")
    
    # 使用最佳參數進行訓練
    best_model = grid_search.best_estimator_
    
    print(f"Start predicting {target}...")
    # 預測與評估
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # 儲存結果
    xgb_final_results_with_images[target] = {
        'rmse': rmse,
        'predictions': y_pred[:5],  # 儲存前五個預測值
        'best_params': best_params
    }

# 輸出結果
for target, results in xgb_final_results_with_images.items():
    print(f"RMSE for {target}: {results['rmse']}")
    print(f"Predictions for {target}: {results['predictions']}")
    print(f"Best parameters for {target}: {results['best_params']}")
