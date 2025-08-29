"""
Lab 3 Example 3: Forward Pass Implementation และ Advanced Techniques
การใช้งาน hooks, gradient checking, และ advanced forward pass patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("Lab 3 Example 3: Forward Pass Implementation")
print("=" * 60)

# 1. Multi-Input Multi-Output Network
class MultiIONet(nn.Module):
    def __init__(self, input1_size, input2_size, output1_size, output2_size):
        super(MultiIONet, self).__init__()
        
        # แยก path สำหรับ input แต่ละตัว
        self.path1 = nn.Sequential(
            nn.Linear(input1_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(input2_size, 64),
            nn.ReLU(), 
            nn.Linear(64, 32)
        )
        
        # รวม features จาก 2 paths
        self.fusion = nn.Linear(64, 128)  # 32 + 32 = 64
        
        # Output heads แยกกัน
        self.head1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output1_size)
        )
        
        self.head2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output2_size)
        )
        
    def forward(self, input1, input2):
        # Process แต่ละ input แยกกัน
        feat1 = self.path1(input1)
        feat2 = self.path2(input2)
        
        # รวม features
        combined = torch.cat([feat1, feat2], dim=1)
        fused = F.relu(self.fusion(combined))
        
        # สร้าง outputs แยกกัน
        output1 = self.head1(fused)
        output2 = self.head2(fused)
        
        return output1, output2

# 2. Conditional Network (เลือก path ตาม condition)
class ConditionalNet(nn.Module):
    def __init__(self, input_size, num_conditions, output_size):
        super(ConditionalNet, self).__init__()
        
        self.input_size = input_size
        self.num_conditions = num_conditions
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Condition-specific layers
        self.condition_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            ) for _ in range(num_conditions)
        ])
        
    def forward(self, x, condition):
        # Shared processing
        shared_feat = self.shared(x)
        
        # เลือก layer ตาม condition
        if isinstance(condition, int):
            # Single condition
            output = self.condition_layers[condition](shared_feat)
        else:
            # Batch of conditions
            batch_size = x.size(0)
            outputs = []
            
            for i in range(batch_size):
                cond = condition[i].item() if torch.is_tensor(condition[i]) else condition[i]
                single_output = self.condition_layers[cond](shared_feat[i:i+1])
                outputs.append(single_output)
                
            output = torch.cat(outputs, dim=0)
            
        return output

# 3. Attention Mechanism
class SimpleAttention(nn.Module):
    def __init__(self, feature_size):
        super(SimpleAttention, self).__init__()
        self.feature_size = feature_size
        self.attention_weights = nn.Linear(feature_size, 1)
        
    def forward(self, features):
        # features: (batch_size, seq_len, feature_size)
        
        # คำนวณ attention scores
        scores = self.attention_weights(features)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # normalize ตาม sequence dimension
        
        # คำนวณ weighted sum
        attended = torch.sum(features * weights, dim=1)  # (batch_size, feature_size)
        
        return attended, weights

class AttentionNet(nn.Module):
    def __init__(self, input_size, seq_len, output_size):
        super(AttentionNet, self).__init__()
        
        self.seq_len = seq_len
        self.encoder = nn.Linear(input_size, 64)
        self.attention = SimpleAttention(64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # Encode แต่ละ time step
        encoded = self.encoder(x.view(-1, input_size))  # (batch_size * seq_len, 64)
        encoded = encoded.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 64)
        
        # Apply attention
        attended, attention_weights = self.attention(encoded)
        
        # Final classification
        output = self.classifier(attended)
        
        return output, attention_weights

# 4. Hook Functions สำหรับ debugging
def register_hooks(model):
    """ลงทะเบียน hooks เพื่อดู intermediate outputs"""
    
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # ลงทะเบียน hooks สำหรับทุก layers
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, nn.Tanh)):
            layer.register_forward_hook(get_activation(name))
    
    return activations

# ทดสอบการใช้งาน
print("\n1. ทดสอบ Multi-Input Multi-Output Network")
print("-" * 45)

# สร้าง multi-IO model
multi_model = MultiIONet(input1_size=10, input2_size=8, output1_size=5, output2_size=3)

# ทดสอบ
input1 = torch.randn(4, 10)  # batch_size=4
input2 = torch.randn(4, 8)

print(f"Input1 shape: {input1.shape}")
print(f"Input2 shape: {input2.shape}")

with torch.no_grad():
    out1, out2 = multi_model(input1, input2)
    print(f"Output1 shape: {out1.shape}")
    print(f"Output2 shape: {out2.shape}")

# 2. ทดสอบ Conditional Network
print("\n2. ทดสอบ Conditional Network")
print("-" * 30)

cond_model = ConditionalNet(input_size=6, num_conditions=3, output_size=4)

# ทดสอบด้วย single condition
test_input = torch.randn(2, 6)
condition = 1  # เลือก condition ที่ 1

with torch.no_grad():
    cond_output = cond_model(test_input, condition)
    print(f"Input shape: {test_input.shape}")
    print(f"Condition: {condition}")
    print(f"Output shape: {cond_output.shape}")

# ทดสอบด้วย batch conditions
batch_conditions = [0, 2]  # condition ต่างกันในแต่ละ sample
with torch.no_grad():
    batch_cond_output = cond_model(test_input, batch_conditions)
    print(f"Batch conditions: {batch_conditions}")
    print(f"Batch output shape: {batch_cond_output.shape}")

# 3. ทดสอบ Attention Network
print("\n3. ทดสอบ Attention Network")
print("-" * 30)

att_model = AttentionNet(input_size=5, seq_len=7, output_size=3)

# สร้าง sequence data
seq_input = torch.randn(3, 7, 5)  # (batch_size, seq_len, input_size)
print(f"Sequence input shape: {seq_input.shape}")

with torch.no_grad():
    att_output, att_weights = att_model(seq_input)
    print(f"Attention output shape: {att_output.shape}")
    print(f"Attention weights shape: {att_weights.shape}")
    print(f"Sample attention weights for first item:")
    print(att_weights[0].squeeze().numpy())

# 4. ทดสอบ Hooks
print("\n4. ทดสอบ Forward Hooks")
print("-" * 25)

# สร้าง simple model สำหรับ demo hooks
class SimpleHookNet(nn.Module):
    def __init__(self):
        super(SimpleHookNet, self).__init__()
        self.layer1 = nn.Linear(4, 8)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(8, 6)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(6, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x

hook_model = SimpleHookNet()
activations = register_hooks(hook_model)

# ทดสอบและดู activations
hook_input = torch.randn(2, 4)
hook_output = hook_model(hook_input)

print(f"Model output shape: {hook_output.shape}")
print("\nCaptured activations:")
for name, activation in activations.items():
    if activation is not None:
        print(f"{name}: {activation.shape}, mean: {activation.mean():.4f}")

# 5. Custom Forward Pass with Multiple Paths
print("\n5. Advanced Forward Pass Example")
print("-" * 35)

class MultiPathNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiPathNet, self).__init__()
        
        # Path 1: Deep narrow
        self.path1 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Path 2: Shallow wide  
        self.path2 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        
        # Path 3: Direct connection
        self.path3 = nn.Linear(input_size, 8)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(24, 32),  # 8 + 8 + 8 = 24
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x, use_paths=None):
        if use_paths is None:
            use_paths = [True, True, True]  # ใช้ทุก paths
            
        paths_outputs = []
        
        if use_paths[0]:
            out1 = self.path1(x)
            paths_outputs.append(out1)
            
        if use_paths[1]:
            out2 = self.path2(x)
            paths_outputs.append(out2)
            
        if use_paths[2]:
            out3 = self.path3(x)
            paths_outputs.append(out3)
        
        # รวม outputs ที่มี
        if len(paths_outputs) > 1:
            combined = torch.cat(paths_outputs, dim=1)
        else:
            combined = paths_outputs[0]
            
        # Adjust fusion layer input size if needed
        if combined.size(1) != 24:
            # Create a temporary fusion layer with correct input size
            temp_fusion = nn.Sequential(
                nn.Linear(combined.size(1), 32),
                nn.ReLU(),
                nn.Linear(32, self.fusion[-1].out_features)
            )
            output = temp_fusion(combined)
        else:
            output = self.fusion(combined)
            
        return output

multipath_model = MultiPathNet(input_size=10, output_size=5)
mp_input = torch.randn(3, 10)

print(f"Input shape: {mp_input.shape}")

# ทดสอบการใช้ path ต่างๆ
path_configs = [
    ([True, True, True], "All paths"),
    ([True, False, False], "Deep narrow only"),
    ([False, True, False], "Shallow wide only"),
    ([False, False, True], "Direct connection only"),
    ([True, True, False], "Deep + Shallow")
]

with torch.no_grad():
    for paths, description in path_configs:
        try:
            mp_output = multipath_model(mp_input, use_paths=paths)
            print(f"{description}: output shape {mp_output.shape}")
        except Exception as e:
            print(f"{description}: Error - {e}")

# 6. Gradient Flow Visualization
print("\n6. การติดตาม Gradient Flow")
print("-" * 30)

class GradientTracker(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GradientTracker, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        ])
        
        self.activations = nn.ModuleList([
            nn.ReLU(),
            nn.ReLU(), 
            nn.ReLU()
        ])
        
    def forward(self, x):
        for i, (layer, activation) in enumerate(zip(self.layers[:-1], self.activations)):
            x = layer(x)
            x = activation(x)
            
        x = self.layers[-1](x)  # Final layer without activation
        return x
    
    def get_gradient_norms(self):
        """คำนวณ gradient norms สำหรับแต่ละ layer"""
        grad_norms = []
        for i, layer in enumerate(self.layers):
            if layer.weight.grad is not None:
                grad_norm = layer.weight.grad.norm().item()
                grad_norms.append((f"layer_{i}", grad_norm))
            else:
                grad_norms.append((f"layer_{i}", 0.0))
        return grad_norms

# สร้าง model และทดสอบ gradient tracking
grad_model = GradientTracker(input_size=8, hidden_size=16, output_size=3)
grad_input = torch.randn(5, 8, requires_grad=True)
target = torch.randint(0, 3, (5,))

# Forward pass
output = grad_model(grad_input)
loss = F.cross_entropy(output, target)

# Backward pass
loss.backward()

# ดู gradient norms
print("Gradient norms after backward pass:")
grad_norms = grad_model.get_gradient_norms()
for layer_name, norm in grad_norms:
    print(f"{layer_name}: {norm:.6f}")

# 7. Model Ensemble Forward Pass
print("\n7. Model Ensemble")
print("-" * 20)

class EnsembleNet(nn.Module):
    def __init__(self, input_size, output_size, num_models=3):
        super(EnsembleNet, self).__init__()
        
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, output_size)
            ) for _ in range(num_models)
        ])
        
        self.num_models = num_models
        
    def forward(self, x, mode='average'):
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
            
        outputs = torch.stack(outputs)  # (num_models, batch_size, output_size)
        
        if mode == 'average':
            return torch.mean(outputs, dim=0)
        elif mode == 'max':
            return torch.max(outputs, dim=0)[0]
        elif mode == 'all':
            return outputs
        else:
            raise ValueError(f"Unknown mode: {mode}")

ensemble_model = EnsembleNet(input_size=6, output_size=4, num_models=3)
ensemble_input = torch.randn(2, 6)

print(f"Ensemble input shape: {ensemble_input.shape}")

with torch.no_grad():
    # ทดสอบ modes ต่างๆ
    avg_output = ensemble_model(ensemble_input, mode='average')
    max_output = ensemble_model(ensemble_input, mode='max')
    all_outputs = ensemble_model(ensemble_input, mode='all')
    
    print(f"Average output shape: {avg_output.shape}")
    print(f"Max output shape: {max_output.shape}")
    print(f"All outputs shape: {all_outputs.shape}")

# 8. Dynamic Architecture
print("\n8. Dynamic Architecture")
print("-" * 25)

class DynamicNet(nn.Module):
    def __init__(self, input_size, base_hidden_size, output_size):
        super(DynamicNet, self).__init__()
        
        self.input_size = input_size
        self.base_hidden_size = base_hidden_size
        self.output_size = output_size
        
        # Pre-defined layers ของขนาดต่างๆ
        self.small_path = nn.Sequential(
            nn.Linear(input_size, base_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(base_hidden_size // 2, output_size)
        )
        
        self.medium_path = nn.Sequential(
            nn.Linear(input_size, base_hidden_size),
            nn.ReLU(),
            nn.Linear(base_hidden_size, base_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(base_hidden_size // 2, output_size)
        )
        
        self.large_path = nn.Sequential(
            nn.Linear(input_size, base_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(base_hidden_size * 2, base_hidden_size),
            nn.ReLU(),
            nn.Linear(base_hidden_size, base_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(base_hidden_size // 2, output_size)
        )
        
    def forward(self, x, complexity='medium'):
        if complexity == 'small':
            return self.small_path(x)
        elif complexity == 'medium':
            return self.medium_path(x)
        elif complexity == 'large':
            return self.large_path(x)
        else:
            # Auto-select based on input complexity (example heuristic)
            input_variance = torch.var(x, dim=1).mean()
            
            if input_variance < 0.5:
                return self.small_path(x)
            elif input_variance < 1.5:
                return self.medium_path(x)
            else:
                return self.large_path(x)

dynamic_model = DynamicNet(input_size=8, base_hidden_size=32, output_size=5)

# ทดสอบ dynamic selection
test_inputs = [
    torch.randn(2, 8) * 0.1,  # Low variance
    torch.randn(2, 8) * 1.0,  # Medium variance  
    torch.randn(2, 8) * 3.0   # High variance
]

print("Dynamic architecture selection:")
with torch.no_grad():
    for i, test_input in enumerate(test_inputs):
        variance = torch.var(test_input, dim=1).mean().item()
        
        # Manual selection
        if variance < 0.5:
            complexity = 'small'
        elif variance < 1.5:
            complexity = 'medium'
        else:
            complexity = 'large'
            
        output = dynamic_model(test_input, complexity='auto')
        print(f"Input {i+1}: variance={variance:.3f}, auto-selected complexity, output shape: {output.shape}")

# 9. การวัดประสิทธิภาพ
print("\n9. Performance Measurement")
print("-" * 30)

import time

def measure_forward_time(model, input_tensor, num_runs=100):
    """วัดเวลาการทำ forward pass"""
    model.eval()
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

# เปรียบเทียบประสิทธิภาพของ models ต่างๆ
models_to_test = [
    ("Simple Linear", nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))),
    ("Multi-layer", nn.Sequential(
        nn.Linear(100, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(), 
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 10)
    )),
    ("Dynamic Small", dynamic_model)
]

test_input_100d = torch.randn(32, 100)  # batch_size=32

print("Forward pass timing comparison:")
for name, model in models_to_test:
    if name == "Dynamic Small":
        # สำหรับ dynamic model ต้องปรับ input size
        dynamic_input = torch.randn(32, 8)
        avg_time = measure_forward_time(model, dynamic_input, num_runs=50)
    else:
        avg_time = measure_forward_time(model, test_input_100d, num_runs=50)
    
    print(f"{name}: {avg_time*1000:.3f} ms per forward pass")

print("\n" + "=" * 60)
print("สรุป Lab 3: Building Neural Networks")
print("=" * 60)
print("✓ การใช้ torch.nn.Module และ Linear layers")
print("✓ การสร้าง custom layers และ activation functions") 
print("✓ Forward pass implementation แบบ advanced")
print("✓ Multi-input/output networks")
print("✓ Attention mechanisms")
print("✓ Hook functions สำหรับ debugging")
print("✓ Dynamic และ ensemble architectures")
print("✓ Performance measurement")
print("=" * 60)