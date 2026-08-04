import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# ==========================================
# 1. SURROGATE NEURAL NETWORK ARCHITECTURE
# ==========================================
class ThermalPINN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=5):
        """
        Inputs:
            [x, y, z, time, material_k, target_temp] (6 dimensions)
        Outputs:
            Predicted Temperature field T(x,y,z,t)
        """
        super(ThermalPINN, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


# ==========================================
# 2. PHYSICS-INFORMED LOSS (HEAT EQUATION PDE)
# ==========================================
def pde_residual_loss(model, x, y, z, t, k_diff, target_T):
    """
    Computes the residual of 3D transient heat conduction:
    dT/dt - alpha * (d2T/dx2 + d2T/dy2 + d2T/dz2) = 0
    """
    # Combine inputs into a single tensor requiring gradient tracking
    inputs = torch.cat([x, y, z, t, k_diff, target_T], dim=1)
    inputs.requires_grad_(True)
    
    # Predict T
    T = model(inputs)
    
    # First derivatives
    grads = torch.autograd.grad(T, inputs, torch.ones_like(T), create_graph=True)[0]
    dT_dx = grads[:, 0:1]
    dT_dy = grads[:, 1:2]
    dT_dz = grads[:, 2:3]
    dT_dt = grads[:, 3:4]
    
    # Second derivatives (Laplacian)
    d2T_dx2 = torch.autograd.grad(dT_dx, inputs, torch.ones_like(dT_dx), create_graph=True)[0][:, 0:1]
    d2T_dy2 = torch.autograd.grad(dT_dy, inputs, torch.ones_like(dT_dy), create_graph=True)[0][:, 1:2]
    d2T_dz2 = torch.autograd.grad(dT_dz, inputs, torch.ones_like(dT_dz), create_graph=True)[0][:, 2:3]
    
    laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
    
    # Heat equation residual: dT/dt - alpha * del^2(T)
    # Scaled thermal diffusivity alpha ~ k_diff
    alpha = k_diff * 0.01 
    pde_residual = dT_dt - alpha * laplacian
    
    return torch.mean(pde_residual ** 2)


# ==========================================
# 3. TRAINING DATA GENERATOR
# ==========================================
def generate_training_data(n_samples=5000):
    """Generates synthetic domain/boundary points for fixture geometry"""
    x = torch.FloatTensor(n_samples, 1).uniform_(-15.0, 15.0).to(device)
    y = torch.FloatTensor(n_samples, 1).uniform_(-15.0, 15.0).to(device)
    z = torch.FloatTensor(n_samples, 1).uniform_(0.0, 20.0).to(device)
    t = torch.FloatTensor(n_samples, 1).uniform_(0.0, 3600.0).to(device)  # up to 60 mins
    
    # Material conductivities: 17-4PH (~18 W/mK), 316L (~16 W/mK), Alumina (~30 W/mK)
    k_diff = torch.FloatTensor(n_samples, 1).uniform_(15.0, 32.0).to(device)
    
    # Heat setting temperatures (450C to 600C)
    target_T = torch.FloatTensor(n_samples, 1).uniform_(450.0, 600.0).to(device)
    
    return x, y, z, t, k_diff, target_T


# ==========================================
# 4. MODEL TRAINING LOOP
# ==========================================
def train():
    model = ThermalPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)
    
    epochs = 5000
    print("--- Starting Thermal PINN Surrogate Training ---")
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Sample points in domain
        x, y, z, t, k_diff, target_T = generate_training_data(n_samples=2048)
        
        # 1. Physics loss (PDE compliance)
        loss_pde = pde_residual_loss(model, x, y, z, t, k_diff, target_T)
        
        # 2. Boundary condition loss (e.g., surface heated to target_T at max t)
        boundary_inputs = torch.cat([x, y, z, torch.ones_like(t)*3600.0, k_diff, target_T], dim=1)
        pred_boundary_T = model(boundary_inputs)
        loss_bc = torch.mean((pred_boundary_T - target_T) ** 2)
        
        # Total Loss
        total_loss = loss_pde + 10.0 * loss_bc
        
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        
        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.6f} | PDE Loss: {loss_pde.item():.6f} | BC Loss: {loss_bc.item():.6f}")

    # Save trained weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/thermal_pinn_surrogate.pth")
    print("✅ Model weights saved to 'models/thermal_pinn_surrogate.pth'")

if __name__ == "__main__":
    train()