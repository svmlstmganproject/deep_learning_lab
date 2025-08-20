import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ตัวอย่างที่ 1: ANN 1 Layer แบบง่ายที่สุด
# =============================================================================

class SimpleANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleANN, self).__init__()
        # สร้าง 1 layer เดียว (Linear layer)
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # Forward pass ผ่าน 1 layer
        output = self.linear(x)
        return output

# สร้าง model
model1 = SimpleANN(input_size=2, output_size=1)
print("Simple ANN (1 layer):")
print(model1)
print(f"Parameters: {sum(p.numel() for p in model1.parameters())}")

# ทดสอบ
test_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
output1 = model1(test_input)
print(f"Input: {test_input}")
print(f"Output: {output1}")

print("\n" + "="*60)

# =============================================================================
# ตัวอย่างที่ 2: ANN 1 Layer สำหรับ Binary Classification
# =============================================================================

class BinaryClassifierANN(nn.Module):
    def __init__(self, input_features):
        super(BinaryClassifierANN, self).__init__()
        self.linear = nn.Linear(input_features, 1)  # 1 output สำหรับ binary
        
    def forward(self, x):
        # Linear transformation + Sigmoid activation
        x = self.linear(x)
        x = torch.sigmoid(x)  # สำหรับ binary classification
        return x

# สร้าง model สำหรับ binary classification
binary_model = BinaryClassifierANN(input_features=4)
print("Binary Classification ANN (1 layer + sigmoid):")
print(binary_model)

# สร้างข้อมูลตัวอย่าง
sample_data = torch.randn(5, 4)  # 5 samples, 4 features
predictions = binary_model(sample_data)
print(f"\nSample predictions (probabilities): {predictions.flatten()}")

print("\n" + "="*60)

# =============================================================================
# ตัวอย่างที่ 3: ANN 1 Layer สำหรับ Multi-class Classification
# =============================================================================

class MultiClassANN(nn.Module):
    def __init__(self, input_features, num_classes):
        super(MultiClassANN, self).__init__()
        self.linear = nn.Linear(input_features, num_classes)
        
    def forward(self, x):
        # Linear transformation + Softmax
        x = self.linear(x)
        # ใช้ log_softmax สำหรับ numerical stability
        x = F.log_softmax(x, dim=1)
        return x

# สร้าง model สำหรับ 3-class classification
multiclass_model = MultiClassANN(input_features=10, num_classes=3)
print("Multi-class Classification ANN (1 layer + softmax):")
print(multiclass_model)

# ทดสอบ
test_batch = torch.randn(4, 10)  # 4 samples, 10 features
class_predictions = multiclass_model(test_batch)
print(f"\nClass predictions (log probabilities):")
print(class_predictions)

# แปลงเป็น probabilities
probabilities = torch.exp(class_predictions)
print(f"\nClass probabilities:")
print(probabilities)

print("\n" + "="*60)

# =============================================================================
# ตัวอย่างที่ 4: ANN 1 Layer สำหรับ Regression
# =============================================================================

class RegressionANN(nn.Module):
    def __init__(self, input_features, output_features=1):
        super(RegressionANN, self).__init__()
        self.linear = nn.Linear(input_features, output_features)
        
    def forward(self, x):
        # สำหรับ regression ไม่ต้องใส่ activation function
        return self.linear(x)

# สร้าง model สำหรับ regression
regression_model = RegressionANN(input_features=5, output_features=1)
print("Regression ANN (1 layer, no activation):")
print(regression_model)

# ทดสอบ regression
regression_input = torch.randn(3, 5)
regression_output = regression_model(regression_input)
print(f"\nRegression predictions: {regression_output.flatten()}")

print("\n" + "="*60)

# =============================================================================
# ตัวอย่างที่ 5: Custom ANN 1 Layer พร้อม Weight Initialization
# =============================================================================

class CustomANN(nn.Module):
    def __init__(self, input_size, output_size, activation='relu'):
        super(CustomANN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        
        # Custom weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """กำหนดค่าเริ่มต้นของ weights"""
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = self.linear(x)
        
        # เลือก activation function
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        elif self.activation == 'none':
            pass  # ไม่ใส่ activation
            
        return x

# ทดสอบ activation functions ต่างๆ
activations = ['relu', 'sigmoid', 'tanh', 'none']
input_data = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

print("Custom ANN with different activations:")
for act in activations:
    model = CustomANN(5, 3, activation=act)
    output = model(input_data)
    print(f"{act:>8}: {output.detach().numpy().flatten()}")

print("\n" + "="*60)

# =============================================================================
# ตัวอย่างที่ 6: การดู Model Parameters
# =============================================================================

def print_model_info(model, model_name):
    """แสดงข้อมูลของ model"""
    print(f"\n{model_name}:")
    print(f"Architecture: {model}")
    
    # นับจำนวน parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # แสดง weight และ bias
    for name, param in model.named_parameters():
        print(f"{name}: shape {param.shape}")
    
    print("-" * 40)

# แสดงข้อมูลของ models
print_model_info(model1, "Simple ANN")
print_model_info(binary_model, "Binary Classifier")
print_model_info(regression_model, "Regression Model")

# =============================================================================
# ตัวอย่างที่ 7: การ Train ANN 1 Layer (Linear Regression)
# =============================================================================

print("\n" + "="*60)
print("ตัวอย่าง: Training ANN 1 Layer สำหรับ Linear Regression")
print("="*60)

# สร้างข้อมูล synthetic สำหรับ linear regression
torch.manual_seed(42)
n_samples = 100
X = torch.randn(n_samples, 1)  # 1 feature
true_weight = 3.0
true_bias = 1.5
noise = torch.randn(n_samples, 1) * 0.1
y = true_weight * X + true_bias + noise

# สร้าง model
linear_model = RegressionANN(input_features=1, output_features=1)

# Loss function และ optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.01)

# Training loop
losses = []
epochs = 200

print(f"Training เริ่มต้น...")
print(f"True weight: {true_weight}, True bias: {true_bias}")

for epoch in range(epochs):
    # Forward pass
    predictions = linear_model(X)
    loss = criterion(predictions, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

# แสดงผลลัพธ์
learned_weight = linear_model.linear.weight.item()
learned_bias = linear_model.linear.bias.item()

print(f"\nผลลัพธ์การ training:")
print(f"Learned weight: {learned_weight:.4f} (True: {true_weight})")
print(f"Learned bias: {learned_bias:.4f} (True: {true_bias})")
print(f"Final loss: {losses[-1]:.6f}")

# =============================================================================
# สรุป
# =============================================================================

print("\n" + "="*60)
print("สรุป: ANN 1 Layer")
print("="*60)
print("""
ANN 1 Layer ประกอบด้วย:
1. nn.Linear(input_size, output_size) - การแปลง linear
2. Activation function (optional) - ReLU, Sigmoid, Tanh, หรือไม่ใส่

การใช้งาน:
- Binary Classification: output=1 + sigmoid
- Multi-class Classification: output=num_classes + softmax  
- Regression: output=target_dim + ไม่ใส่ activation

ข้อจำกัด:
- ไม่สามารถเรียนรู้ non-linear patterns ที่ซับซ้อนได้
- เหมาะสำหรับปัญหาที่มีความสัมพันธ์เชิงเส้น
- จำนวน parameters น้อย: input_size × output_size + output_size
""")

print("\n🎉 ตัวอย่าง ANN 1 Layer เสร็จสิ้น!")