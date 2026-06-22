import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Use CPU for a fair comparison with our Zig CPU implementation
device = torch.device('cpu')

# Transform to load fashion mnist
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download dataset (handled by torchvision)
train_dataset = datasets.FashionMNIST(root='./data_py', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data_py', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Define 3-layer MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP().to(device)

# Initialize weights using Kaiming (He) normal initialization to match Zig's initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0.0)

model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
# Momentum SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
# Learning rate decay: lr = lr * 0.90 each epoch
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

print(f"Starting training on {device} (3-layer NN: 784 -> 128 -> 64 -> 10)...")

for epoch in range(15):
    start_time = time.time()
    
    # Training pass
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
        
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    
    # Evaluation pass
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += batch_y.size(0)
            test_correct += predicted.eq(batch_y).sum().item()
            
    eval_loss = test_loss / len(test_loader)
    eval_acc = test_correct / test_total
    
    scheduler.step()
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1:2}/15 | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {eval_loss:.4f} | Test Acc: {eval_acc*100:.2f}% | Time: {elapsed:.2f}s")
