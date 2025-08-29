import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def excel_to_tensors(file_path, target_column, test_size=0.2, batch_size=32):
    """
    แปลงข้อมูลจาก Excel เป็น PyTorch Tensors พร้อม DataLoader
    
    Args:
        file_path (str): path ของไฟล์ Excel
        target_column (str): ชื่อ column ที่เป็น target
        test_size (float): สัดส่วนข้อมูล test (default: 0.2)
        batch_size (int): ขนาด batch (default: 32)
    
    Returns:
        dict: ข้อมูลที่ประมวลผลแล้ว
    """
    print(f"📁 อ่านไฟล์: {file_path}")
    
    # 1. อ่านข้อมูลจาก Excel
    df = pd.read_excel(file_path)
    print(f"✅ ข้อมูล: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. แยก features และ target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 3. จัดการ categorical columns (One-hot encoding)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"🔧 One-hot encoding: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols)
    
    # 4. แปลงเป็น numpy
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.float32)
    
    # 5. แบ่งข้อมูล train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=42
    )
    print(f"📊 Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 6. Feature scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    print("⚖️ Feature scaling เรียบร้อย")
    
    # 7. แปลงเป็น PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    X_test_tensor = torch.from_numpy(X_test_scaled).float()
    y_train_tensor = torch.from_numpy(y_train_scaled).float()
    y_test_tensor = torch.from_numpy(y_test_scaled).float()
    
    # 8. สร้าง DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"🚀 PyTorch tensors พร้อม! Features: {X_train_tensor.shape[1]}")
    
    return {
        'X_train': X_train_tensor,
        'X_test': X_test_tensor, 
        'y_train': y_train_tensor,
        'y_test': y_test_tensor,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': list(X.columns),
        'n_features': X_train_tensor.shape[1]
    }

def create_sample_data():
    """สร้างข้อมูลตัวอย่างสำหรับทดสอบ"""
    np.random.seed(42)
    n = 500
    
    data = {
        'area': np.random.normal(100, 20, n),
        'rooms': np.random.randint(1, 5, n),
        'age': np.random.randint(0, 30, n),
        'location': np.random.choice(['A', 'B', 'C'], n),
        'price': np.random.normal(200000, 50000, n)
    }
    
    # สร้าง relationship
    data['price'] = (data['area'] * 2000 + 
                    data['rooms'] * 10000 - 
                    data['age'] * 500 + 
                    np.random.normal(0, 5000, n))
    
    df = pd.DataFrame(data)
    df.to_excel('sample_data.xlsx', index=False)
    print("📝 สร้าง sample_data.xlsx เรียบร้อย")
    return df

def main():
    """ตัวอย่างการใช้งาน"""
    print("=" * 50)
    print("Excel to PyTorch Tensors - Simple Version")
    print("=" * 50)
    
    # สร้างข้อมูลตัวอย่าง
    df_sample = create_sample_data()
    print(f"ตัวอย่างข้อมูล:\n{df_sample.head()}")
    
    # แปลงข้อมูลเป็น tensors
    data = excel_to_tensors('sample_data.xlsx', target_column='price')
    
    # แสดงผลลัพธ์
    print(f"\n📈 ผลลัพธ์:")
    print(f"Features: {data['n_features']}")
    print(f"Feature names: {data['feature_names']}")
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"y_train shape: {data['y_train'].shape}")
    print(f"Train batches: {len(data['train_loader'])}")
    
    # ทดสอบ DataLoader
    print(f"\n🔄 ทดสอบ DataLoader:")
    for i, (batch_X, batch_y) in enumerate(data['train_loader']):
        print(f"Batch {i+1}: X{batch_X.shape}, y{batch_y.shape}")
        if i == 1:  # แสดง 2 batch แรก
            break
    
    # ตัวอย่างการใช้กับ neural network
    import torch.nn as nn
    
    class SimpleNet(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.linear = nn.Linear(n_features, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # สร้าง model และทดสอบ
    model = SimpleNet(data['n_features'])
    sample_input = data['X_train'][:5]
    
    with torch.no_grad():
        predictions = model(sample_input)
        
    print(f"\n🧠 ทดสอบ Neural Network:")
    print(f"Model input shape: {sample_input.shape}")
    print(f"Model predictions: {predictions.flatten()[:3].numpy()}")
    print(f"Actual values: {data['y_train'][:3].numpy()}")
    
    print(f"\n✅ เรียบร้อย! ข้อมูลพร้อมสำหรับ Deep Learning")

if __name__ == "__main__":
    main()