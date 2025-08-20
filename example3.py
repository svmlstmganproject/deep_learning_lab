# Lab 3: Building Neural Networks
# PyTorch สำหรับ Deep Learning

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("PyTorch version:", torch.__version__)

# =============================================================================
# ส่วนที่ 1: การใช้ torch.nn.Module
# =============================================================================

print("\n" + "="*50)
print("ส่วนที่ 1: การใช้ torch.nn.Module")
print("="*50)

# สร้าง Neural Network แบบง่าย
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        # กำหนด layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward pass
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# สร้างและทดสอบ model
model = SimpleNet(input_size=10, hidden_size=20, output_size=5)
print(f"Model architecture:\n{model}")

# ดูจำนวน parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

# =============================================================================
# ส่วนที่ 2: Linear Layers และ Activation Functions
# =============================================================================

print("\n" + "="*50)
print("ส่วนที่ 2: Linear Layers และ Activation Functions")
print("="*50)

# ทดสอบ Linear layer
linear_layer = nn.Linear(5, 3)
input_data = torch.randn(2, 5)  # batch_size=2, input_features=5

print(f"Input shape: {input_data.shape}")
output = linear_layer(input_data)
print(f"Output shape: {output.shape}")
print(f"Linear layer weight shape: {linear_layer.weight.shape}")
print(f"Linear layer bias shape: {linear_layer.bias.shape}")

# ทดสอบ Activation Functions
print("\n--- Activation Functions ---")
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {x}")

activations = {
    'ReLU': torch.relu(x),
    'Sigmoid': torch.sigmoid(x),
    'Tanh': torch.tanh(x),
    'LeakyReLU': F.leaky_relu(x, negative_slope=0.1),
    'Softmax': F.softmax(x, dim=0)
}

for name, result in activations.items():
    print(f"{name}: {result}")

# =============================================================================
# ส่วนที่ 3: การสร้าง Custom Layers
# =============================================================================

print("\n" + "="*50)
print("ส่วนที่ 3: การสร้าง Custom Layers")
print("="*50)

# Custom Linear Layer
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # สร้าง parameters เอง
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        # คำนวณ linear transformation เอง
        return torch.mm(x, self.weight.t()) + self.bias

# ทดสอบ Custom Layer
custom_layer = CustomLinear(4, 2)
test_input = torch.randn(3, 4)  # batch_size=3, features=4
custom_output = custom_layer(test_input)

print(f"Custom layer input shape: {test_input.shape}")
print(f"Custom layer output shape: {custom_output.shape}")

# Custom Activation Layer
class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()
        
    def forward(self, x):
        # Custom activation: x^2 if x > 0, else 0.1*x
        return torch.where(x > 0, x**2, 0.1*x)

custom_activation = CustomActivation()
test_values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
activated = custom_activation(test_values)
print(f"\nCustom activation input: {test_values}")
print(f"Custom activation output: {activated}")

# =============================================================================
# ส่วนที่ 4: Forward Pass Implementation
# =============================================================================

print("\n" + "="*50)
print("ส่วนที่ 4: Forward Pass Implementation")
print("="*50)

# Multi-Layer Neural Network
class MultiLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(MultiLayerNet, self).__init__()
        
        # สร้าง layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # สร้าง sequential model
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# สร้าง model
mlp = MultiLayerNet(
    input_dim=784,      # เหมือน MNIST (28x28)
    hidden_dims=[256, 128, 64],
    output_dim=10,      # 10 classes
    dropout_rate=0.3
)

print(f"Multi-layer network:\n{mlp}")

# ทดสอบ forward pass
batch_size = 32
sample_input = torch.randn(batch_size, 784)
output = mlp(sample_input)

print(f"\nForward pass test:")
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")

# =============================================================================
# ส่วนที่ 5: การใช้ Different Network Architectures
# =============================================================================

print("\n" + "="*50)
print("ส่วนที่ 5: Network Architectures ต่างๆ")
print("="*50)

# 1. Sequential Model
sequential_model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, 10),
    nn.Softmax(dim=1)
)

print("Sequential Model:")
print(sequential_model)

# 2. ModuleList for dynamic layers
class DynamicNet(nn.Module):
    def __init__(self, layer_sizes):
        super(DynamicNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) 
            for i in range(len(layer_sizes)-1)
        ])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # ไม่ใส่ activation ใน output layer
                x = torch.relu(x)
        return x

dynamic_net = DynamicNet([784, 512, 256, 128, 10])
print(f"\nDynamic Network layers: {len(dynamic_net.layers)}")

# 3. Network with Skip Connections
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        
    def forward(self, x):
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual  # Skip connection
        return torch.relu(out)

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=2):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        
        for block in self.res_blocks:
            x = block(x)
            
        x = self.output_layer(x)
        return x

resnet = ResNet(100, 64, 10, num_blocks=3)
print(f"\nResNet with skip connections created")

# =============================================================================
# ส่วนที่ 6: Model Information และ Utilities
# =============================================================================

print("\n" + "="*50)
print("ส่วนที่ 6: Model Information และ Utilities")
print("="*50)

def model_summary(model, input_shape):
    """แสดงข้อมูลสรุปของ model"""
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {
                'input_shape': list(input[0].size()),
                'output_shape': list(output.size()) if hasattr(output, 'size') else [],
                'nb_params': sum([p.numel() for p in module.parameters()])
            }
        
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # สร้าง summary dictionary
    summary = {}
    hooks = []
    
    # Register hooks
    model.apply(register_hook)
    
    # Forward pass
    model(torch.zeros(1, *input_shape))
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Print summary
    print("="*70)
    print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
    print("="*70)
    
    total_params = 0
    for layer_name, layer_info in summary.items():
        print(f"{layer_name:<25} {str(layer_info['output_shape']):<20} {layer_info['nb_params']:<15}")
        total_params += layer_info['nb_params']
    
    print("="*70)
    print(f"Total params: {total_params:,}")
    print("="*70)

# ทดสอบ model summary
print("\nModel Summary for MultiLayerNet:")
model_summary(mlp, (784,))

# =============================================================================
# ส่วนที่ 7: Exercises
# =============================================================================

print("\n" + "="*50)
print("ส่วนที่ 7: แบบฝึกหัด")
print("="*50)

print("""
แบบฝึกหัด:

1. สร้าง Neural Network สำหรับ XOR problem
   - Input: 2 features
   - Hidden layer: 4 neurons  
   - Output: 1 neuron
   - ใช้ sigmoid activation

2. สร้าง Custom Layer ที่ทำ Batch Normalization เอง

3. สร้าง Network ที่มี multiple outputs (multi-task learning)

4. ทดลองเปรียบเทียบ activation functions ต่างๆ กับข้อมูลเดียวกัน

5. สร้าง Network ที่สามารถ handle variable input size ได้
""")

# ตัวอย่างคำตอบ Exercise 1: XOR Network
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

xor_net = XORNet()
print(f"\nXOR Network created: {xor_net}")

# ทดสอบ XOR network
xor_data = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
xor_predictions = xor_net(xor_data)
print(f"XOR predictions: {xor_predictions.flatten()}")

print("\n" + "="*50)
print("Lab 3 เสร็จสิ้น! 🎉")
print("="*50)