"""
Lab 3 Example 1: พื้นฐาน torch.nn.Module และ Linear layers
การสร้าง Neural Network แบบง่าย และการใช้งาน activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("Lab 3 Example 1: torch.nn.Module และ Linear layers")
print("=" * 60)

# 1. การสร้าง Simple Neural Network โดยใช้ nn.Module
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        
        # กำหนด layers
        self.layer1 = nn.Linear(input_size, hidden_size)    # Input -> Hidden
        self.layer2 = nn.Linear(hidden_size, output_size)   # Hidden -> Output
        
    def forward(self, x):
        # Forward pass
        x = self.layer1(x)          # ผ่าน linear layer แรก
        x = torch.relu(x)           # ใช้ ReLU activation
        x = self.layer2(x)          # ผ่าน linear layer ที่สอง
        return x

# 2. สร้าง model และทดสอบ
print("\n1. สร้าง Simple Neural Network")
print("-" * 30)

# กำหนดขนาด
input_size = 4      # จำนวน features
hidden_size = 8     # จำนวน neurons ใน hidden layer
output_size = 2     # จำนวน classes

# สร้าง model
model = SimpleNet(input_size, hidden_size, output_size)
print(f"Model structure:")
print(model)

# แสดงข้อมูล parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

# 3. ทดสอบการใช้งานด้วยข้อมูลตัวอย่าง
print("\n2. ทดสอบการใช้งาน model")
print("-" * 30)

# สร้างข้อมูลตัวอย่าง
batch_size = 3
sample_input = torch.randn(batch_size, input_size)
print(f"Input shape: {sample_input.shape}")
print(f"Input data:\n{sample_input}")

# Forward pass
with torch.no_grad():  # ไม่คำนวณ gradient
    output = model(sample_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output data:\n{output}")

# 4. ทดสอบ activation functions ต่างๆ
print("\n3. เปรียบเทียบ Activation Functions")
print("-" * 40)

class ActivationDemo(nn.Module):
    def __init__(self):
        super(ActivationDemo, self).__init__()
        self.linear = nn.Linear(5, 3)
        
    def forward(self, x, activation='relu'):
        x = self.linear(x)
        
        if activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'tanh':
            return torch.tanh(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)
        else:
            return x  # linear (no activation)

# สร้าง demo model
demo_model = ActivationDemo()
sample_data = torch.randn(2, 5)
print(f"Input data:\n{sample_data}")

# ทดสอบ activation functions ต่างๆ
activations = ['linear', 'relu', 'sigmoid', 'tanh', 'softmax']
with torch.no_grad():
    for act in activations:
        output = demo_model(sample_data, activation=act)
        print(f"\n{act.upper()} output:")
        print(output)

# 5. การดู weights และ biases
print("\n4. การดู Parameters ของ Model")
print("-" * 35)

print("SimpleNet parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
    print(f"Values:\n{param.data}")
    print()

# 6. ตัวอย่างการใช้ nn.Sequential (วิธีง่าย)
print("\n5. การใช้ nn.Sequential")
print("-" * 25)

sequential_model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)

print("Sequential model:")
print(sequential_model)

# ทดสอบให้ผลลัพธ์เหมือนกับ SimpleNet
with torch.no_grad():
    seq_output = sequential_model(sample_input)
    print(f"\nSequential model output shape: {seq_output.shape}")

print("\n" + "=" * 60)
print("จบ Example 1: พื้นฐาน torch.nn.Module และ Linear layers")
print("=" * 60)