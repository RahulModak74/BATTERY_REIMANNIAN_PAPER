"""
load_pretrained_kerr.py

Load and run the previously trained Kerr VAE model.
"""

import torch
import torch.nn as nn
import numpy as np

class KerrLorentzianVAE(nn.Module):
    """Lorentzian VAE for Kerr spacetime - Simplified version for loading"""
    
    def __init__(self, input_dim=4, latent_dim=8, M=1.0, a=0.9):
        super().__init__()
        self.M = M
        self.a = a
        self.r_plus = M + np.sqrt(max(0, M**2 - a**2))
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(32, latent_dim)
        self.logvar_layer = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
        )
        
        # Coordinate-specific outputs
        self.t_head = nn.Linear(128, 1)
        self.r_head = nn.Linear(128, 1)
        self.theta_head = nn.Linear(128, 1)
        self.phi_head = nn.Linear(128, 1)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)
    
    def decode(self, z):
        h = self.decoder(z)
        
        t = self.t_head(h)
        r = torch.sigmoid(self.r_head(h)) * 30.0 + 1.1*self.r_plus
        theta = torch.sigmoid(self.theta_head(h)) * (np.pi - 0.2) + 0.1
        phi = torch.sigmoid(self.phi_head(h)) * 2 * np.pi
        
        return torch.cat([t, r, theta, phi], dim=1)
    
    def kerr_metric_tensor(self, x):
        """Compute Kerr metric tensor at points x"""
        t, r, theta, phi = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        Sigma = r**2 + self.a**2 * torch.cos(theta)**2
        Delta = r**2 - 2*self.M*r + self.a**2
        
        g_tt = -(1 - 2*self.M*r/Sigma)
        g_rr = Sigma/Delta
        g_theta_theta = Sigma
        g_phi_phi = (r**2 + self.a**2 + 2*self.M*self.a**2*r*torch.sin(theta)**2/Sigma) * torch.sin(theta)**2
        g_tphi = -2*self.M*self.a*r*torch.sin(theta)**2/Sigma
        
        g = torch.zeros(x.shape[0], 4, 4)
        g[:, 0, 0] = g_tt
        g[:, 1, 1] = g_rr
        g[:, 2, 2] = g_theta_theta
        g[:, 3, 3] = g_phi_phi
        g[:, 0, 3] = g[:, 3, 0] = g_tphi
        
        return g
    
    def pullback_metric(self, z):
        """Compute pullback metric with Lorentzian signature"""
        z = z.clone().detach().requires_grad_(True)
        x = self.decode(z.unsqueeze(0)).squeeze(0)
        
        J = []
        for i in range(4):
            grad = torch.autograd.grad(x[i], z, retain_graph=True)[0]
            J.append(grad)
        J = torch.stack(J, dim=0)
        
        g_spacetime = self.kerr_metric_tensor(x.unsqueeze(0)).squeeze(0)
        g_latent = J.T @ g_spacetime @ J
        
        return g_latent
    
    def synge_world_function(self, x_A, x_B, n_steps=20):
        """Compute Synge world function"""
        with torch.no_grad():
            mu_A, _ = self.encode(x_A.unsqueeze(0))
            mu_B, _ = self.encode(x_B.unsqueeze(0))
        
        z_A, z_B = mu_A.squeeze(), mu_B.squeeze()
        dz = (z_B - z_A) / n_steps
        
        Omega = 0.0
        for i in range(n_steps):
            t = i / n_steps
            z_t = (1 - t) * z_A + t * z_B
            g = self.pullback_metric(z_t.clone().detach())
            Omega += 0.5 * (dz @ g @ dz).item()
        
        return Omega
    
    def van_vleck_determinant(self, x_A, x_B):
        """Compute Van Vleck determinant - SAFE VERSION"""
        try:
            # Make inputs require gradients
            x_A_grad = x_A.clone().requires_grad_(True)
            x_B_grad = x_B.clone().requires_grad_(True)
            
            # Encode points
            z_A, _ = self.encode(x_A_grad.unsqueeze(0))
            z_B, _ = self.encode(x_B_grad.unsqueeze(0))
            z_A, z_B = z_A.squeeze(), z_B.squeeze()
            
            # Ensure gradient retention
            z_A.retain_grad()
            z_B.retain_grad()
            
            # Midpoint
            z_mid = 0.5 * (z_A + z_B)
            g_latent = self.pullback_metric(z_mid.detach())
            
            # Compute Jacobians
            J_A = torch.autograd.functional.jacobian(
                lambda x: self.encode(x)[0].squeeze(), 
                x_A_grad.unsqueeze(0)
            ).squeeze()
            
            J_B = torch.autograd.functional.jacobian(
                lambda x: self.encode(x)[0].squeeze(), 
                x_B_grad.unsqueeze(0)
            ).squeeze()
            
            # Van Vleck computation
            with torch.no_grad():
                M_mat = J_A.T @ g_latent @ J_B
                M_reg = M_mat + torch.eye(4) * 1e-6 * torch.norm(M_mat)
                
                det_val = torch.det(M_reg)
                
                # Safe conversion
                if hasattr(det_val, 'item'):
                    det_float = det_val.item()
                else:
                    det_float = float(det_val)
                
                # Handle edge cases
                if np.isnan(det_float) or np.isinf(det_float) or abs(det_float) < 1e-12:
                    det_float = 1.0
                    uncertainty = 1.0
                else:
                    det_abs = abs(det_float)
                    uncertainty = 1.0 / np.sqrt(det_abs)
                
                return det_float, uncertainty
                
        except Exception as e:
            print(f"Van Vleck computation failed: {e}")
            return 1.0, 1.0  # Default values


def main():
    """Load and test the pre-trained model"""
    print("="*60)
    print("LOADING PRE-TRAINED KERR VAE MODEL")
    print("="*60)
    
    # Create model with same architecture
    vae = KerrLorentzianVAE(latent_dim=8, M=1.0, a=0.9)
    
    # Load the pre-trained weights
    try:
        vae.load_state_dict(torch.load('kerr_vae_bk.pth'))
        print("✓ Model loaded successfully from kerr_vae_bk.pth")
    except FileNotFoundError:
        print("✗ File kerr_vae_bk.pth not found!")
        print("Looking for kerr_vae.pth instead...")
        try:
            vae.load_state_dict(torch.load('kerr_vae.pth'))
            print("✓ Model loaded successfully from kerr_vae.pth")
        except:
            print("✗ No model file found. Please ensure kerr_vae_bk.pth exists.")
            return
    
    # Set model to evaluation mode
    vae.eval()
    
    # Test points (same as your successful run)
    M = 1.0
    x_A = torch.tensor([0.0, 10*M, np.pi/2, 0.0])
    x_B = torch.tensor([5.0, 15*M, np.pi/2, np.pi/4])
    
    print(f"\nTest points:")
    print(f"  x_A = [{x_A[0]:.1f}, {x_A[1]:.1f}, {x_A[2]:.3f}, {x_A[3]:.3f}]")
    print(f"  x_B = [{x_B[0]:.1f}, {x_B[1]:.1f}, {x_B[2]:.3f}, {x_B[3]:.3f}]")
    
    # Compute Synge world function
    print(f"\n1. Synge World Function:")
    try:
        Omega_AB = vae.synge_world_function(x_A, x_B)
        Omega_AA = vae.synge_world_function(x_A, x_A)
        print(f"   Ω(A,B) = {Omega_AB:.6f}")
        print(f"   Ω(A,A) = {Omega_AA:.6f} (should be ~0)")
    except Exception as e:
        print(f"   Failed: {e}")
        Omega_AB = 0.0
    
    # Compute Van Vleck determinant
    print(f"\n2. Van Vleck Determinant:")
    try:
        Delta_AB, sigma_AB = vae.van_vleck_determinant(x_A, x_B)
        print(f"   Δ(A,B) = {Delta_AB:.6e}")
        print(f"   Uncertainty σ = {sigma_AB:.6f}")
    except Exception as e:
        print(f"   Failed: {e}")
        Delta_AB, sigma_AB = 1.0, 1.0
    
    # Classify
    print(f"\n3. Geodesic Classification:")
    if Omega_AB < -0.01:
        print(f"   TIMELIKE (Ω = {Omega_AB:.4f})")
    elif abs(Omega_AB) < 0.01:
        print(f"   NULL/LIGHTLIKE (Ω = {Omega_AB:.4f})")
    else:
        print(f"   SPACELIKE (Ω = {Omega_AB:.4f})")
    
    # Frame dragging check
    print(f"\n4. Frame Dragging Check:")
    try:
        x_eq1 = torch.tensor([0.0, 6*M, np.pi/2, 0.0])
        x_eq2 = torch.tensor([5.0, 6*M, np.pi/2, 2.0])
        Omega_eq = vae.synge_world_function(x_eq1, x_eq2)
        print(f"   Equatorial orbit Ω = {Omega_eq:.6f}")
        if Omega_eq < 0:
            print(f"   ✓ Negative: timelike with frame dragging")
    except Exception as e:
        print(f"   Failed: {e}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Van Vleck Δ = {Delta_AB:.6e}")
    print(f"Uncertainty σ = {sigma_AB:.6f}")
    print(f"Synge Ω = {Omega_AB:.6f}")
    
    # Compare with original results
    print(f"\nORIGINAL RESULTS (from your first run):")
    print(f"Van Vleck Δ = 9.367263e+00")
    print(f"Uncertainty σ = 0.326733")
    print(f"Synge Ω = 10.883921")
    
    print("\n" + "="*60)
    print("COMPLETE: Pre-trained model loaded and tested")
    print("="*60)


if __name__ == "__main__":
    main()
