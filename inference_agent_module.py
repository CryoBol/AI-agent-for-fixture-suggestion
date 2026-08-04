import torch
import numpy as np

class ThermalFEMPredictor:
    def __init__(self, model_path="models/thermal_pinn_surrogate.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ThermalPINN().to(self.device)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict_mesh_thermal_field(self, waist_dia, disc_dia, length, target_temp, soak_time_mins, material_k):
        """Generates 3D temperature fields on a spatial evaluation grid using the trained neural net."""
        # Create 3D evaluation grid
        x = np.linspace(-disc_dia/2, disc_dia/2, 15)
        y = np.linspace(-disc_dia/2, disc_dia/2, 15)
        z = np.linspace(0, length, 15)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Flatten spatial coordinates
        x_flat = torch.FloatTensor(X.flatten()[:, None]).to(self.device)
        y_flat = torch.FloatTensor(Y.flatten()[:, None]).to(self.device)
        z_flat = torch.FloatTensor(Z.flatten()[:, None]).to(self.device)
        
        # Constant parameters repeated across all mesh points
        n_pts = x_flat.shape[0]
        t_flat = torch.FloatTensor(np.full((n_pts, 1), soak_time_mins * 60.0)).to(self.device)
        k_flat = torch.FloatTensor(np.full((n_pts, 1), material_k)).to(self.device)
        temp_flat = torch.FloatTensor(np.full((n_pts, 1), target_temp)).to(self.device)
        
        # Pass through trained agent network
        inputs = torch.cat([x_flat, y_flat, z_flat, t_flat, k_flat, temp_flat], dim=1)
        with torch.no_grad():
            T_pred = self.model(inputs).cpu().numpy().reshape(X.shape)
            
        max_delta_T = float(np.max(T_pred) - np.min(T_pred))
        
        return X, Y, Z, T_pred, max_delta_T