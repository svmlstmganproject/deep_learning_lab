"""
Lab 3 Example 2: การสร้าง Custom Layers
การสร้าง layers ที่กำหนดเอง และการรวม layers เข้าด้วยกน
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("Lab 3 Example 2: การสร้าง Custom Layers")
print("=" * 60)

# 1. สร้าง Custom Linear Layer แบบ Manual
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # สร้าง parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
            
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        # Xavier initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Manual matrix multiplication: x @ W^T + b
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output
    
    def __repr__(self):
        return f'CustomLinear(in_features={self.in_features}, out_features={self.out_features})'

# 2. สร้าง Custom Activation Layer
class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        
    def forward(self, x):
        # Manual ReLU implementation
        return torch.max(torch.zeros_like(x), x)

# 3. สร้าง Block Layer (รวม Linear + Activation + Dropout)
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', dropout_rate=0.0):
        super(LinearBlock, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # เลือก activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = None  # Linear
            
    def forward(self, x):
        x = self.linear(x)
        
        if self.activation is not None:
            x = self.activation(x)
            
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x

# 4. สร้าง Residual Block (Skip Connection)
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )
        
    def forward(self, x):
        residual = x  # เก็บ input เดิม
        x = self.block(x)
        x = x + residual  # Skip connection
        return F.relu(x)

# 5. สร้าง Advanced Neural Network ด้วย Custom Layers
class AdvancedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(AdvancedNet, self).__init__()
        
        layers = []
        current_size = input_size
        
        # สร้าง hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            # ใช้ custom linear สำหรับ layer แรก
            if i == 0:
                layers.append(CustomLinear(current_size, hidden_size))
                layers.append(CustomReLU())
            else:
                # ใช้ LinearBlock สำหรับ layer อื่นๆ
                layers.append(LinearBlock(
                    current_size, 
                    hidden_size, 
                    activation='relu',
                    dropout_rate=0.1
                ))
            
            # เพิ่ม Residual Block ถ้า hidden_size เท่ากัน
            if i > 0 and hidden_size == hidden_sizes[i-1]:
                layers.append(ResidualBlock(hidden_size))
                
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# ทดสอบการใช้งาน
print("\n1. ทดสอบ Custom Linear Layer")
print("-" * 35)

# เปรียบเทียบ Custom Linear กับ nn.Linear
custom_linear = CustomLinear(4, 3)
builtin_linear = nn.Linear(4, 3)

# Copy weights เพื่อเปรียบเทียบ
with torch.no_grad():
    builtin_linear.weight.data = custom_linear.weight.data.clone()
    builtin_linear.bias.data = custom_linear.bias.data.clone()

# ทดสอบด้วยข้อมูลเดียวกัน
test_input = torch.randn(2, 4)
print(f"Test input shape: {test_input.shape}")

with torch.no_grad():
    custom_output = custom_linear(test_input)
    builtin_output = builtin_linear(test_input)
    
    print(f"Custom Linear output:\n{custom_output}")
    print(f"Built-in Linear output:\n{builtin_output}")
    print(f"Outputs are equal: {torch.allclose(custom_output, builtin_output)}")

# 2. ทดสอบ LinearBlock
print("\n2. ทดสอบ LinearBlock")
print("-" * 25)

block = LinearBlock(4, 6, activation='relu', dropout_rate=0.2)
print(f"LinearBlock structure:\n{block}")

# ทดสอบในโหมด training และ evaluation
block.train()  # Training mode (dropout active)
train_output = block(test_input)
print(f"\nTraining mode output shape: {train_output.shape}")

block.eval()   # Evaluation mode (dropout inactive)  
eval_output = block(test_input)
print(f"Evaluation mode output shape: {eval_output.shape}")

# 3. ทดสอบ ResidualBlock
print("\n3. ทดสอบ ResidualBlock")
print("-" * 25)

res_block = ResidualBlock(6)
test_input_6d = torch.randn(2, 6)

with torch.no_grad():
    res_output = res_block(test_input_6d)
    print(f"Input shape: {test_input_6d.shape}")
    print(f"Residual block output shape: {res_output.shape}")

# 4. ทดสอบ AdvancedNet
print("\n4. ทดสอบ Advanced Neural Network")
print("-" * 35)

# สร้าง advanced model
advanced_model = AdvancedNet(
    input_size=10, 
    hidden_sizes=[16, 16, 8], 
    output_size=3
)

print(f"Advanced Model structure:")
print(advanced_model)

# นับจำนวน parameters
total_params = sum(p.numel() for p in advanced_model.parameters())
trainable_params = sum(p.numel() for p in advanced_model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# ทดสอบการใช้งาน
test_batch = torch.randn(5, 10)
print(f"\nTest input shape: {test_batch.shape}")

with torch.no_grad():
    advanced_output = advanced_model(test_batch)
    print(f"Advanced model output shape: {advanced_output.shape}")
    print(f"Output sample:\n{advanced_output[:2]}")  # แสดง 2 samples แรก

# 5. แสดงการทำงานของแต่ละ layer
print("\n5. การติดตาม Forward Pass")
print("-" * 30)

class DebuggingNet(nn.Module):
    def __init__(self):
        super(DebuggingNet, self).__init__()
        self.layer1 = LinearBlock(4, 8, activation='relu')
        self.layer2 = LinearBlock(8, 6, activation='tanh')
        self.layer3 = nn.Linear(6, 2)
        
    def forward(self, x):
        print(f"Input: {x.shape}")
        
        x = self.layer1(x)
        print(f"After layer1: {x.shape}, mean: {x.mean():.4f}")
        
        x = self.layer2(x)
        print(f"After layer2: {x.shape}, mean: {x.mean():.4f}")
        
        x = self.layer3(x)
        print(f"After layer3: {x.shape}, mean: {x.mean():.4f}")
        
        return x

debug_model = DebuggingNet()
debug_input = torch.randn(1, 4)

print("Forward pass debugging:")
with torch.no_grad():
    debug_output = debug_model(debug_input)

print("\n" + "=" * 60)
print("จบ Example 2: การสร้าง Custom Layers")
print("=" * 60)