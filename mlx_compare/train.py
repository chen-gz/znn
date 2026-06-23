import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from torchvision import datasets, transforms
import numpy as np

import sys
run_large = "--large" in sys.argv

# Define the MLP models
class MLP(nn.Module):
    def __init__(self, large=False):
        super().__init__()
        self.large = large
        if large:
            self.fc1 = nn.Linear(784, 2048)
            self.fc2 = nn.Linear(2048, 2048)
            self.fc3 = nn.Linear(2048, 1024)
            self.fc4 = nn.Linear(1024, 10)
        else:
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

    def __call__(self, x):
        if self.large:
            x = mx.maximum(self.fc1(x), 0)
            x = mx.maximum(self.fc2(x), 0)
            x = mx.maximum(self.fc3(x), 0)
            return self.fc4(x)
        else:
            x = mx.maximum(self.fc1(x), 0)
            x = mx.maximum(self.fc2(x), 0)
            return self.fc3(x)

# Load data using torchvision (shared from pytorch_compare/data_py)
train_dataset = datasets.FashionMNIST(root='./data_py', train=True, download=True)
test_dataset = datasets.FashionMNIST(root='./data_py', train=False, download=True)

# Convert to numpy arrays and normalize
train_images = (train_dataset.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
train_labels = train_dataset.targets.numpy().astype(np.int32)
test_images = (test_dataset.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
test_labels = test_dataset.targets.numpy().astype(np.int32)

model = MLP(large=run_large)
mx.eval(model.parameters())

# Initialize weights using Kaiming (He) normal initialization to match Zig
# In MLX, nn.Linear weights are already initialized with a normal distribution scaled by 1/sqrt(fan_in),
# which is equivalent to He initialization for ReLU.

# Loss function
def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

# Gradient function
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Optimizer (Momentum SGD)
lr = 0.05
optimizer = optim.SGD(learning_rate=lr, momentum=0.9)

batch_size = 64
num_train = len(train_images)
num_batches = num_train // batch_size

if run_large:
    print(f"Starting training with MLX GPU (Large model: 784 -> 2048 -> 2048 -> 1024 -> 10)...")
else:
    print(f"Starting training with MLX GPU (3-layer NN: 784 -> 128 -> 64 -> 10)...")

for epoch in range(15):
    start_time = time.time()
    
    # Shuffle indices
    indices = np.random.permutation(num_train)
    
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    # Update learning rate manually to match ExponentialLR decay of 0.90 per epoch
    optimizer.learning_rate = lr * (0.90 ** epoch)
    
    for b in range(num_batches):
        batch_indices = indices[b * batch_size : (b + 1) * batch_size]
        batch_x = mx.array(train_images[batch_indices])
        batch_y = mx.array(train_labels[batch_indices])
        
        # Forward & backward pass
        loss, grads = loss_and_grad_fn(model, batch_x, batch_y)
        optimizer.update(model, grads)
        
        # Evaluate to trigger GPU computation (MLX uses lazy evaluation)
        mx.eval(model.parameters(), optimizer.state)
        
        epoch_loss += loss.item()
        
        preds = mx.argmax(model(batch_x), axis=1)
        correct += mx.sum(preds == batch_y).item()
        total += batch_size
        
    train_loss = epoch_loss / num_batches
    train_acc = correct / total
    
    # Evaluate test set
    test_x = mx.array(test_images)
    test_y = mx.array(test_labels)
    test_preds = mx.argmax(model(test_x), axis=1)
    test_acc = mx.sum(test_preds == test_y).item() / len(test_labels)
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1:2}/15 | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Time: {elapsed:.2f}s")
