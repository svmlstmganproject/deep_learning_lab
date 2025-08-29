#!/usr/bin/env python3
"""
Classification Model Implementation with Flet UI
Based on the classification.ipynb notebook
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flet as ft
from typing import List, Tuple
import random

# Import D2L functionality (simplified version for standalone use)
class D2LModule:
    """Simplified D2L Module for standalone use"""
    def __init__(self):
        self.hyperparameters = {}
        self.training_history = {'loss': [], 'accuracy': []}
    
    def save_hyperparameters(self):
        """Save hyperparameters"""
        pass
    
    def plot(self, metric, value, train=True):
        """Track metrics for plotting"""
        if train:
            if metric not in self.training_history:
                self.training_history[metric] = []
            self.training_history[metric].append(value.item() if hasattr(value, 'item') else value)

def add_to_class(cls):
    """Decorator to add methods to a class"""
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator

# The Classifier Class
class Classifier(D2LModule):
    """The base class of classification models."""
    def __init__(self, lr=0.1):
        super().__init__()
        self.lr = lr
        self.loss = None
        self.accuracy = None
    
    def validation_step(self, batch):
        X, Y = batch[:-1], batch[-1]
        Y_hat = self.forward(X)
        self.plot('loss', self.loss(Y_hat, Y), train=False)
        self.plot('acc', self.accuracy(Y_hat, Y), train=False)
        return Y_hat

# Configure optimizers
@add_to_class(D2LModule)
def configure_optimizers(self):
    return optim.SGD(self.parameters(), lr=self.lr)

# Accuracy implementation
@add_to_class(Classifier)
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    preds = Y_hat.argmax(dim=1).type(Y.dtype)
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    return compare.mean() if averaged else compare

# Simple Linear Classifier
class SimpleClassifier(Classifier):
    def __init__(self, num_inputs, num_outputs, lr=0.1):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = nn.Linear(num_inputs, num_outputs)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, X):
        return self.net(X)

class ClassificationApp:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.epochs = 100
        self.lr = 0.1
        
    def generate_data(self, num_samples=100, num_features=5, num_classes=3):
        """Generate synthetic classification data"""
        # Generate random data
        X = torch.randn(num_samples, num_features)
        
        # Create labels based on a simple rule
        Y = torch.randint(0, num_classes, (num_samples,))
        
        # Split into train/test
        train_size = int(0.8 * num_samples)
        self.X_train = X[:train_size]
        self.Y_train = Y[:train_size]
        self.X_test = X[train_size:]
        self.Y_test = Y[train_size:]
        
        return f"Generated {num_samples} samples with {num_features} features and {num_classes} classes"
    
    def train_model(self):
        """Train the classification model"""
        if self.X_train is None:
            return "Please generate data first!"
        
        # Create model
        num_features = self.X_train.shape[1]
        num_classes = len(torch.unique(self.Y_train))
        self.model = SimpleClassifier(num_features, num_classes, self.lr)
        
        # Setup optimizer
        optimizer = self.model.configure_optimizers()
        
        # Training loop
        train_losses = []
        train_accuracies = []
        
        for epoch in range(self.epochs):
            # Forward pass
            Y_hat = self.model(self.X_train)
            loss = self.model.loss(Y_hat, self.Y_train)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            accuracy = self.model.accuracy(Y_hat, self.Y_train)
            
            train_losses.append(loss.item())
            train_accuracies.append(accuracy.item())
            
            # Update training history
            self.model.plot('loss', loss, train=True)
            self.model.plot('accuracy', accuracy, train=True)
        
        # Test accuracy
        with torch.no_grad():
            Y_hat_test = self.model(self.X_test)
            test_accuracy = self.model.accuracy(Y_hat_test, self.Y_test)
        
        return f"Training completed!\nFinal Train Accuracy: {train_accuracies[-1]:.4f}\nTest Accuracy: {test_accuracy:.4f}"
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None:
            return "Please train the model first!"
        
        try:
            # Convert input to tensor
            if isinstance(input_data, list):
                X = torch.tensor(input_data, dtype=torch.float32)
            else:
                X = torch.tensor([input_data], dtype=torch.float32)
            
            # Make prediction
            with torch.no_grad():
                Y_hat = self.model(X)
                predictions = Y_hat.argmax(dim=1)
            
            return f"Prediction: Class {predictions.item()}"
        except Exception as e:
            return f"Prediction error: {str(e)}"

def main(page: ft.Page):
    page.title = "Classification Model with PyTorch"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    
    # Initialize the classification app
    classifier_app = ClassificationApp()
    
    # UI Components
    status_text = ft.Text("Ready to start classification!", size=16, weight=ft.FontWeight.BOLD)
    output_text = ft.Text("", size=14, color=ft.Colors.BLUE_600)
    
    # Input fields
    num_samples_input = ft.TextField(
        label="Number of Samples",
        value="100",
        width=200
    )
    
    num_features_input = ft.TextField(
        label="Number of Features", 
        value="5",
        width=200
    )
    
    num_classes_input = ft.TextField(
        label="Number of Classes",
        value="3", 
        width=200
    )
    
    learning_rate_input = ft.TextField(
        label="Learning Rate",
        value="0.1",
        width=200
    )
    
    epochs_input = ft.TextField(
        label="Epochs",
        value="100",
        width=200
    )
    
    # Buttons
    generate_btn = ft.ElevatedButton(
        "Generate Data",
        on_click=lambda _: generate_data_clicked()
    )
    
    train_btn = ft.ElevatedButton(
        "Train Model",
        on_click=lambda _: train_model_clicked()
    )
    
    predict_btn = ft.ElevatedButton(
        "Test Prediction",
        on_click=lambda _: predict_clicked()
    )
    
    def generate_data_clicked():
        try:
            num_samples = int(num_samples_input.value)
            num_features = int(num_features_input.value)
            num_classes = int(num_classes_input.value)
            
            result = classifier_app.generate_data(num_samples, num_features, num_classes)
            output_text.value = result
            status_text.value = "Data generated successfully!"
            status_text.color = ft.Colors.GREEN
            page.update()
        except Exception as e:
            output_text.value = f"Error generating data: {str(e)}"
            status_text.value = "Error occurred!"
            status_text.color = ft.Colors.RED
            page.update()
    
    def train_model_clicked():
        try:
            classifier_app.lr = float(learning_rate_input.value)
            classifier_app.epochs = int(epochs_input.value)
            
            result = classifier_app.train_model()
            output_text.value = result
            status_text.value = "Model trained successfully!"
            status_text.color = ft.Colors.GREEN
            page.update()
        except Exception as e:
            output_text.value = f"Error training model: {str(e)}"
            status_text.value = "Error occurred!"
            status_text.color = ft.Colors.RED
            page.update()
    
    def predict_clicked():
        try:
            # Generate random test data
            test_data = [random.uniform(-2, 2) for _ in range(5)]
            result = classifier_app.predict(test_data)
            output_text.value = f"Test data: {test_data}\n{result}"
            status_text.value = "Prediction completed!"
            status_text.color = ft.Colors.GREEN
            page.update()
        except Exception as e:
            output_text.value = f"Error making prediction: {str(e)}"
            status_text.value = "Error occurred!"
            status_text.color = ft.Colors.RED
            page.update()
    
    # Layout
    page.add(
        ft.Text("Classification Model with PyTorch", size=24, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        status_text,
        ft.Divider(),
        ft.Text("Parameters:", size=18, weight=ft.FontWeight.BOLD),
        ft.Row([
            num_samples_input,
            num_features_input,
            num_classes_input
        ]),
        ft.Row([
            learning_rate_input,
            epochs_input
        ]),
        ft.Divider(),
        ft.Text("Actions:", size=18, weight=ft.FontWeight.BOLD),
        ft.Row([
            generate_btn,
            train_btn,
            predict_btn
        ]),
        ft.Divider(),
        ft.Text("Output:", size=18, weight=ft.FontWeight.BOLD),
        output_text
    )

if __name__ == "__main__":
    print("Starting Classification App with Flet...")
    print("✓ PyTorch version:", torch.__version__)
    print("✓ All imports successful!")
    ft.app(target=main) 