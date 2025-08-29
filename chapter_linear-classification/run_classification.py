#!/usr/bin/env python3
"""
Classification Model Implementation
Based on the classification.ipynb notebook
"""

import torch
from d2l import torch as d2l

print("✓ All imports successful!")
print(f"PyTorch version: {torch.__version__}")

# The Classifier Class
class Classifier(d2l.Module):
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

# Configure optimizers
@d2l.add_to_class(d2l.Module)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)

# Accuracy implementation
@d2l.add_to_class(Classifier)
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    preds = Y_hat.argmax(axis=1).type(Y.dtype)
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    return compare.mean() if averaged else compare

print("✓ Classifier class and methods defined successfully!")

# Test the implementation
if __name__ == "__main__":
    print("\n=== Testing Classification Implementation ===")
    
    # Create a simple test
    X = torch.randn(10, 5)  # 10 samples, 5 features
    Y = torch.randint(0, 3, (10,))  # 3 classes
    
    # Create a simple linear classifier
    class SimpleClassifier(Classifier):
        def __init__(self, num_inputs, num_outputs, lr=0.1):
            super().__init__()
            self.save_hyperparameters()
            self.net = torch.nn.Linear(num_inputs, num_outputs)
            self.loss = torch.nn.CrossEntropyLoss()
        
        def forward(self, X):
            return self.net(X)
    
    # Test the classifier
    model = SimpleClassifier(5, 3)
    Y_hat = model(X)
    accuracy = model.accuracy(Y_hat, Y)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print("✓ All tests passed!") 