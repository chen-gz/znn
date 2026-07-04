import torch

def solve_analytical(x, y):
    # w = cov(x, y) / var(x)
    # b = mean_y - w * mean_x
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    
    num = torch.sum((x - mean_x) * (y - mean_y))
    den = torch.sum((x - mean_x) ** 2)
    
    w = num / den
    b = mean_y - w * mean_x
    return w.item(), b.item()

def solve_gradient_descent(x, y, lr=0.05, epochs=100):
    # Initialize parameters
    w = torch.zeros(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    for epoch in range(1, epochs + 1):
        # Forward pass
        y_pred = x @ w + b
        
        # Loss computation (MSE)
        loss = torch.mean((y_pred - y) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Update weights (disable grad tracking to update in-place)
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            
            # Zero out gradients for the next step
            w.grad.zero_()
            b.grad.zero_()
            
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3}/{epochs}: Loss = {loss.item():.6f} | w = {w.item():.4f} | b = {b.item():.4f}")
            
    return w.item(), b.item()

def main():
    print("=========================================")
    print("Linear Regression using PyTorch (Python)")
    print("=========================================\n")
    
    # 1. Generate synthetic data: y = 2.5 * x - 1.2 + noise
    N = 100
    true_w = 2.5
    true_b = -1.2
    
    # Set seed for reproducibility
    torch.manual_seed(12345)
    
    x = (torch.rand(N, 1) * 4.0) - 2.0
    noise = (torch.rand(N, 1) * 0.1) - 0.05
    y = true_w * x + true_b + noise
    
    # 2. Solve using Gradient Descent
    print("--- Running Gradient Descent ---")
    gd_w, gd_b = solve_gradient_descent(x, y, lr=0.05, epochs=100)
    
    # 3. Solve using Analytical Formula
    analytical_w, analytical_b = solve_analytical(x, y)
    
    print("\nTraining complete!")
    print(f"Final Learned Model (GD):   y = {gd_w:.4f} * x + {gd_b:.4f}")
    print(f"Analytical Model (Formula): y = {analytical_w:.4f} * x + {analytical_b:.4f}")
    print(f"Ground Truth Model:         y = {true_w:.2f} * x + {true_b:.2f}")

if __name__ == "__main__":
    main()
