import numpy as np
import torch
from torch import nn

# Generate synthetic data
def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# DataLoader
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data loader."""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

# Define the model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, X):
        return self.linear(X)

# Loss function
def loss_fn(y_hat, y):
    return nn.MSELoss()(y_hat, y)

# Training function
def train(model, data_iter, num_epochs, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            l = loss_fn(model(X), y)
            l.backward()
            optimizer.step()
        print(f'epoch {epoch + 1}, loss {l:f}')

# Main script
if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    model = LinearRegression()
    num_epochs = 3
    lr = 0.03

    train(model, data_iter, num_epochs, lr)

    w = model.linear.weight.data
    b = model.linear.bias.data
    print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
    print(f'error in estimating b: {true_b - b}')
