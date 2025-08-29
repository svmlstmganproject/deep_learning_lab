# Lab: การนำเข้าข้อมูลจาก Excel และแปลงเป็น PyTorch Tensors
# สำหรับ Deep Learning

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("PyTorch version:", torch.__version__)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

# =============================================================================
# ส่วนที่ 1: สร้างข้อมูลตัวอย่างใน Excel Format (สำหรับทดสอบ)
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 1: สร้างข้อมูลตัวอย่าง")
print("="*60)

# สร้างข้อมูลตัวอย่าง - House Price Dataset
np.random.seed(42)
n_samples = 1000

data = {
    'area_sqft': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'age_years': np.random.randint(0, 50, n_samples),
    'location': np.random.choice(['Downtown', 'Suburban', 'Rural'], n_samples),
    'garage': np.random.choice([0, 1], n_samples),
    'price': np.random.normal(300000, 100000, n_samples)
}

# สร้าง relationship ระหว่าง features และ target
data['price'] = (data['area_sqft'] * 150 + 
                data['bedrooms'] * 20000 + 
                data['bathrooms'] * 15000 - 
                data['age_years'] * 1000 + 
                data['garage'] * 25000 + 
                np.random.normal(0, 20000, n_samples))

# สร้าง DataFrame
df = pd.DataFrame(data)

# บันทึกเป็น Excel file
df.to_excel('house_prices.xlsx', index=False, sheet_name='Data')
print(f"✓ สร้างไฟล์ house_prices.xlsx เรียบร้อย")
print(f"✓ จำนวนข้อมูล: {len(df)} รายการ")
print(f"✓ จำนวน features: {len(df.columns)-1}")

# แสดงข้อมูลตัวอย่าง
print(f"\nตัวอย่างข้อมูล 5 แถวแรก:")
print(df.head())

print(f"\nข้อมูลสถิติเบื้องต้น:")
print(df.describe())

# =============================================================================
# ส่วนที่ 2: อ่านข้อมูลจาก Excel
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 2: อ่านข้อมูลจาก Excel")
print("="*60)

# วิธีที่ 1: อ่านไฟล์ Excel พื้นฐาน
df_loaded = pd.read_excel('house_prices.xlsx', sheet_name='Data')
print(f"✓ อ่านไฟล์ Excel เรียบร้อย")
print(f"Shape: {df_loaded.shape}")

# วิธีที่ 2: อ่านเฉพาะ columns ที่ต้องการ
selected_columns = ['area_sqft', 'bedrooms', 'bathrooms', 'price']
df_selected = pd.read_excel('house_prices.xlsx', 
                          sheet_name='Data', 
                          usecols=selected_columns)
print(f"✓ อ่านเฉพาะ columns ที่เลือก: {selected_columns}")

# วิธีที่ 3: อ่านเฉพาะบางแถว
df_subset = pd.read_excel('house_prices.xlsx', 
                         sheet_name='Data', 
                         nrows=100)  # อ่านแค่ 100 แถวแรก
print(f"✓ อ่านเฉพาะ 100 แถวแรก: Shape {df_subset.shape}")

# =============================================================================
# ส่วนที่ 3: Data Preprocessing
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 3: Data Preprocessing")
print("="*60)

# ใช้ข้อมูลเต็ม
df_work = df_loaded.copy()

# 1. ตรวจสอบ missing values
print("Missing values:")
print(df_work.isnull().sum())

# 2. ตรวจสอบ data types
print(f"\nData types:")
print(df_work.dtypes)

# 3. Handle categorical data - Location
print(f"\nLocation categories: {df_work['location'].unique()}")

# One-hot encoding สำหรับ location
location_encoded = pd.get_dummies(df_work['location'], prefix='location')
df_work = pd.concat([df_work, location_encoded], axis=1)
df_work = df_work.drop('location', axis=1)

print(f"✓ One-hot encoding สำหรับ location เรียบร้อย")
print(f"New columns: {location_encoded.columns.tolist()}")

# 4. แยก features และ target
feature_columns = [col for col in df_work.columns if col != 'price']
X = df_work[feature_columns]
y = df_work['price']

print(f"\nFeatures ({len(feature_columns)}): {feature_columns}")
print(f"Target: price")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# =============================================================================
# ส่วนที่ 4: แปลงเป็น NumPy Arrays
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 4: แปลงเป็น NumPy Arrays")
print("="*60)

# แปลง pandas DataFrame เป็น NumPy
X_np = X.values
y_np = y.values

print(f"X_np shape: {X_np.shape}, dtype: {X_np.dtype}")
print(f"y_np shape: {y_np.shape}, dtype: {y_np.dtype}")

# ตัวอย่างข้อมูล
print(f"\nตัวอย่าง X_np (3 แถวแรก):")
print(X_np[:3])
print(f"\nตัวอย่าง y_np (5 ค่าแรก): {y_np[:5]}")

# =============================================================================
# ส่วนที่ 5: Feature Scaling/Normalization
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 5: Feature Scaling/Normalization")
print("="*60)

# แบ่งข้อมูล train/test ก่อน scaling
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

print(f"Train set: X={X_train.shape}, y={y_train.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")

# StandardScaler (Z-score normalization)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit เฉพาะ training data
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"✓ Feature scaling เรียบร้อย")
print(f"X_train_scaled - mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print(f"y_train_scaled - mean: {y_train_scaled.mean():.4f}, std: {y_train_scaled.std():.4f}")

# =============================================================================
# ส่วนที่ 6: แปลงเป็น PyTorch Tensors
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 6: แปลงเป็น PyTorch Tensors")
print("="*60)

# วิธีที่ 1: จาก NumPy arrays
X_train_tensor = torch.from_numpy(X_train_scaled).float()
X_test_tensor = torch.from_numpy(X_test_scaled).float()
y_train_tensor = torch.from_numpy(y_train_scaled).float()
y_test_tensor = torch.from_numpy(y_test_scaled).float()

print(f"PyTorch Tensors:")
print(f"X_train_tensor: {X_train_tensor.shape}, dtype: {X_train_tensor.dtype}")
print(f"X_test_tensor: {X_test_tensor.shape}, dtype: {X_test_tensor.dtype}")
print(f"y_train_tensor: {y_train_tensor.shape}, dtype: {y_train_tensor.dtype}")
print(f"y_test_tensor: {y_test_tensor.shape}, dtype: {y_test_tensor.dtype}")

# วิธีที่ 2: สร้าง tensor โดยตรง
X_direct = torch.tensor(X_train_scaled, dtype=torch.float32)
y_direct = torch.tensor(y_train_scaled, dtype=torch.float32)

print(f"\nDirect tensor creation:")
print(f"X_direct: {X_direct.shape}, dtype: {X_direct.dtype}")

# วิธีที่ 3: แปลงจาก pandas โดยตรง
X_from_pandas = torch.from_numpy(X.values).float()
print(f"From pandas: {X_from_pandas.shape}")

# =============================================================================
# ส่วนที่ 7: การตรวจสอบและ Visualization
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 7: การตรวจสอบและ Visualization")
print("="*60)

# ตรวจสอบการแปลง tensor
print("Tensor properties:")
print(f"X_train_tensor requires_grad: {X_train_tensor.requires_grad}")
print(f"X_train_tensor device: {X_train_tensor.device}")
print(f"Memory usage: {X_train_tensor.element_size() * X_train_tensor.nelement() / 1024**2:.2f} MB")

# การเข้าถึงข้อมูล
print(f"\nการเข้าถึงข้อมูล:")
print(f"First sample: {X_train_tensor[0]}")
print(f"First 3 samples of feature 0: {X_train_tensor[:3, 0]}")
print(f"Feature statistics - min: {X_train_tensor.min():.4f}, max: {X_train_tensor.max():.4f}")

# สร้างกราฟ
plt.figure(figsize=(15, 10))

# กราฟที่ 1: Distribution ของ target variable
plt.subplot(2, 3, 1)
plt.hist(y_train.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('House Price Distribution (Original)')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')

# กราฟที่ 2: Distribution หลัง scaling
plt.subplot(2, 3, 2)
plt.hist(y_train_scaled.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('House Price Distribution (Scaled)')
plt.xlabel('Scaled Price')
plt.ylabel('Frequency')

# กราฟที่ 3: Correlation matrix
plt.subplot(2, 3, 3)
correlation_matrix = np.corrcoef(X_train_scaled.T)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
           xticklabels=feature_columns, yticklabels=feature_columns)
plt.title('Feature Correlation Matrix')
plt.xticks(rotation=45)

# กราฟที่ 4: Feature distributions
plt.subplot(2, 3, 4)
for i in range(min(3, X_train_scaled.shape[1])):
    plt.hist(X_train_scaled[:, i], bins=30, alpha=0.5, 
             label=f'{feature_columns[i]}')
plt.legend()
plt.title('Feature Distributions (Scaled)')
plt.xlabel('Scaled Values')
plt.ylabel('Frequency')

# กราฟที่ 5: Scatter plot
plt.subplot(2, 3, 5)
plt.scatter(X_train[:, 0], y_train, alpha=0.5, s=10)
plt.xlabel('Area (sqft)')
plt.ylabel('Price ($)')
plt.title('Area vs Price')

# กราฟที่ 6: Data summary
plt.subplot(2, 3, 6)
summary_data = {
    'Total Samples': len(df),
    'Training Samples': len(X_train),
    'Test Samples': len(X_test),
    'Features': X_train.shape[1],
    'Memory (MB)': X_train_tensor.element_size() * X_train_tensor.nelement() / 1024**2
}

bars = plt.bar(range(len(summary_data)), list(summary_data.values()))
plt.xticks(range(len(summary_data)), list(summary_data.keys()), rotation=45)
plt.title('Dataset Summary')

# เพิ่มค่าบน bars
for bar, value in zip(bars, summary_data.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
             f'{value:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# ส่วนที่ 8: การสร้าง DataLoader สำหรับ PyTorch
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 8: การสร้าง DataLoader")
print("="*60)

from torch.utils.data import TensorDataset, DataLoader

# สร้าง TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# สร้าง DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"✓ DataLoader สร้างเรียบร้อย")
print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# ทดสอบ DataLoader
for batch_idx, (data, target) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}: X shape = {data.shape}, y shape = {target.shape}")
    if batch_idx == 2:  # แสดงเฉพาะ 3 batch แรก
        break

# =============================================================================
# ส่วนที่ 9: ตัวอย่างการใช้งานกับ ANN
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 9: ตัวอย่างการใช้งานกับ ANN")
print("="*60)

import torch.nn as nn

# สร้าง simple neural network
class HousePricePredictor(nn.Module):
    def __init__(self, input_features):
        super(HousePricePredictor, self).__init__()
        self.linear = nn.Linear(input_features, 1)
    
    def forward(self, x):
        return self.linear(x)

# สร้าง model
model = HousePricePredictor(input_features=X_train_tensor.shape[1])
print(f"Model created: {X_train_tensor.shape[1]} inputs → 1 output")

# ทดสอบ forward pass
with torch.no_grad():
    sample_prediction = model(X_train_tensor[:5])
    print(f"Sample predictions: {sample_prediction.flatten()}")
    print(f"Actual values: {y_train_tensor[:5]}")

# =============================================================================
# ส่วนที่ 10: สรุปและ Best Practices
# =============================================================================

print("\n" + "="*60)
print("ส่วนที่ 10: สรุปและ Best Practices")
print("="*60)

summary = f"""
📊 สรุปการทำงาน:

1. ✅ อ่านข้อมูลจาก Excel: {df.shape[0]:,} samples, {df.shape[1]} columns
2. ✅ Data preprocessing: One-hot encoding, scaling
3. ✅ Train/Test split: {len(X_train)}/{len(X_test)} samples
4. ✅ Feature scaling: StandardScaler
5. ✅ แปลงเป็น PyTorch tensors: {X_train_tensor.dtype}
6. ✅ สร้าง DataLoader: batch_size={batch_size}
7. ✅ ทดสอบกับ Neural Network

🎯 Best Practices:
- ใช้ .float() สำหรับ tensors ใน neural networks
- Scale features ก่อนแปลงเป็น tensors
- ใช้ DataLoader สำหรับ batch processing
- เก็บ scalers ไว้สำหรับ inverse transform
- ตรวจสอบ tensor shapes และ dtypes เสมอ

💾 Memory Usage: {X_train_tensor.element_size() * X_train_tensor.nelement() / 1024**2:.2f} MB
"""

print(summary)

# บันทึก scalers และ tensors
torch.save({
    'X_train_tensor': X_train_tensor,
    'X_test_tensor': X_test_tensor,
    'y_train_tensor': y_train_tensor,
    'y_test_tensor': y_test_tensor,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_columns': feature_columns
}, 'processed_data.pth')

print("✅ บันทึกข้อมูลที่ประมวลผลแล้วใน 'processed_data.pth'")

print("\n🎉 Lab เสร็จสิ้น! พร้อมสำหรับการ train neural networks")