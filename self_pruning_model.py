import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

class PrunableLinear(nn.Module):
    """
    Part 1: The 'Prunable' Linear Layer
    A custom linear layer with learnable gates for each weight.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Gate scores tensor with the exact same shape as the weight tensor
        # This parameter will learn whether a weight is active (1) or pruned (0)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize standard weights with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize gate scores to a positive constant (e.g. 2.0).
        # Ensures sigmoid(gate_scores) starts close to 1 (unpruned) at the beginning of training.
        nn.init.constant_(self.gate_scores, 2.0)
        
    def forward(self, x):
        # 1. Transform gate_scores to gates between 0 and 1 using the Sigmoid function
        gates = torch.sigmoid(self.gate_scores)
        
        # 2. Calculate "pruned weights" via element-wise multiplication
        pruned_weights = self.weight * gates
        
        # 3. Perform standard linear layer operation
        return F.linear(x, pruned_weights, self.bias)


class SelfPruningNetwork(nn.Module):
    """
    Standard Feed-Forward Neural Network consisting of custom PrunableLinear layers.
    Architecture aimed at classifying the 32x32x3 CIFAR-10 images.
    """
    def __init__(self):
        super(SelfPruningNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        # Dense network with 3 hidden layers + 1 output layer
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)
        
        # Optional: Add dropout out layers to prevent standard overfitting (not structurally strictly required for sparsity but stabilizes the baseline)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        
        # No relu on the final layer; raw logits for CrossEntropy
        x = self.fc4(x)
        return x

    def calculate_sparsity_loss(self):
        """
        Part 2: The Sparsity Regularization Loss
        Calculates the L1 norm of all gate values across all PrunableLinear layers.
        Because gates are outputs of the Sigmoid function, they are strictly positive,
        meaning the L1 norm is merely the sum of the gate values.
        """
        sparsity_loss = 0.0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                # Retrieve the active continuous gate values
                gates = torch.sigmoid(m.gate_scores)
                # Combine L1 Norms (Sum of absolute values; here they are always > 0)
                sparsity_loss += torch.sum(gates)
        return sparsity_loss


def evaluate_model_metrics(model, test_loader, device, sparsity_threshold=1e-2):
    """
    Calculates final test accuracy and overall structural sparsity percentage.
    """
    model.eval()
    correct = 0
    total = 0
    
    total_weights = 0
    pruned_weights = 0
    
    with torch.no_grad():
        # Calculate Classification Accuracy
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # Calculate Sparsity percentage
        for m in model.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores)
                # Count total weights
                total_weights += gates.numel()
                # Count weights whose gates shrank under the explicit small threshold
                pruned_weights += torch.sum(gates < sparsity_threshold).item()
                
    accuracy = 100 * correct / total if total > 0 else 0
    sparsity_level = (pruned_weights / total_weights) * 100 if total_weights > 0 else 0
    
    return accuracy, sparsity_level


def get_all_gate_values(model):
    """Extracts all gate values into a flat Numpy array for visualization plotting."""
    all_gates = []
    with torch.no_grad():
         for m in model.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores).flatten()
                all_gates.append(gates.cpu().numpy())
                
    if all_gates:
        return np.concatenate(all_gates)
    return np.array([])


def train_and_evaluate(lam, dataloaders, device, num_epochs=15):
    """
    Part 3: Training and Evaluation Loop
    Trains the self-pruning model using a target lambda coefficient.
    """
    print(f"\n[{'='*40}]\nStarting Experiment -> Lambda: {lam}\n[{'='*40}]")
    
    model = SelfPruningNetwork().to(device)
    # Adam Optimizer updating both the linear weights AND the gate_scores
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Cosine annealing scheduler helps fine-tune accuracy dynamically
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_loader, test_loader = dataloaders
    
    for epoch in range(num_epochs):
        model.train()
        running_class_loss = 0.0
        running_sparsity_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Loss Formulation
            class_loss = criterion(outputs, labels)
            sparsity_loss = model.calculate_sparsity_loss()
            
            # Total Loss = ClassificationLoss + λ * SparsityLoss
            total_loss = class_loss + (lam * sparsity_loss)
            
            # Backpropagation updates weight and gate_scores parameter concurrently
            total_loss.backward()
            optimizer.step()
            
            running_class_loss += class_loss.item()
            running_sparsity_loss += sparsity_loss.item()
            
        scheduler.step()
        
        avg_c_loss = running_class_loss / len(train_loader)
        avg_s_loss = running_sparsity_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Class Loss: {avg_c_loss:.4f} | "
              f"Sparse Loss: {avg_s_loss:.4f} | "
              f"Total: {(avg_c_loss + lam * avg_s_loss):.4f}")
        
    accuracy, sparsity = evaluate_model_metrics(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {accuracy:.2f}% | Final Sparsity Level: {sparsity:.2f}%")
    
    return model, accuracy, sparsity


def main():
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device}")
    
    # 2. CIFAR-10 Data Loading & Preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Store standard datasets in a sub-folder to keep directory clean
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    dataloaders = (train_loader, test_loader)
    
    # 3. Parameter experimentation sweep per prompt criteria (low, medium, high lambda)
    lambdas = [1e-5, 1e-4, 1e-3]
    results = []
    
    best_model = None
    best_gates = None
    max_balanced_score = -float('inf') 
    
    # Run loop
    for lam in lambdas:
        model, acc, sparsity = train_and_evaluate(lam, dataloaders, device, num_epochs=12) # Reduced epochs for case-study fast eval
        results.append({
            'Lambda': float(lam), 
            'Test Accuracy (%)': round(acc, 2), 
            'Sparsity Level (%)': round(sparsity, 2)
        })
        
        # Save the model gates that best demonstrates the trade-off success
        # The best model has a balanced ratio of remaining accurate while still establishing high structural pruning
        balance_score = acc * (sparsity + 0.1)
        if balance_score > max_balanced_score:
            max_balanced_score = balance_score
            best_model = model
            best_gates = get_all_gate_values(model)
    
    # Ensure export directories exist
    os.makedirs('results', exist_ok=True)
    
    # 4. Result Visualizations (Gate distribution histogram)
    plt.figure(figsize=(10, 6))
    # We expect a large spike at zero, mapping the "Pruned" network, and another separate distributed cluster further away scaling up to 1.
    plt.hist(best_gates, bins=50, color='indigo', alpha=0.8, edgecolor='black')
    plt.title('Distribution of Continuous Sigmoid Gate Values (Best Model)', fontsize=14)
    plt.xlabel('Gate Value (~0 = Pruned, ~1 = Retained Active)', fontsize=12)
    plt.ylabel('Frequency parameter mapping (Log Scale)', fontsize=12)
    plt.yscale('log') # Log scale is required since heavily pruned parameters overwhelmingly dominate the linear display ratio mapping
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('gate_distribution.png', dpi=300)
    print("\n[+] Exported 'gate_distribution.png' to root path.")

    # 5. Output Result Data Tracking via Markdown Tables
    results_df = pd.DataFrame(results)
    markdown_table = results_df.to_markdown(index=False)
    
    print("\n--- Final Experimentation Results ---")
    print(markdown_table)
    
    with open('results/result_table.md', 'w') as f:
        f.write("# Sparsity vs Test Accuracy Trade-off Matrix\n\n")
        f.write(markdown_table)
        f.write("\n")
    print("[+] Exported 'results/result_table.md'")

if __name__ == '__main__':
    # Force reproducibility seed mapping
    torch.manual_seed(42)
    np.random.seed(42)
    main()
