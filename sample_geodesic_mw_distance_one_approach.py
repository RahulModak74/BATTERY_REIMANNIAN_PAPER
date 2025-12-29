def mw_distance_type2(x1, x2, vae_model, physics_weights):
    # Full Riemannian distance along geodesic
    z1 = vae_model.encode(x1)
    z2 = vae_model.encode(x2)
    
    # Geodesic in latent space (straight line for flat approx)
    def geodesic(t):
        return (1 - t) * z1 + t * z2
    
    # Integral of metric along path
    t = torch.linspace(0, 1, steps=100)
    total = 0.0
    for ti in t:
        z = geodesic(ti)
        J = torch.autograd.functional.jacobian(vae_model.decode, z)
        g = J.T @ torch.diag(physics_weights) @ J
        dz_dt = z2 - z1
        total += torch.dot(dz_dt, g @ dz_dt)
    
    return 0.5 * total * (1 / len(t))
