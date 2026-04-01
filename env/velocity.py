import math

K_PHI = 0.45
K_BLOCKAGE = 0.05
Z_REF = 7
PHI_MIN = 0.2
PHI_MAX = 0.5


def compute_velocity_triangles(params):
    rpm = params["rpm"]
    r2 = params["r2"]
    r1 = params["r1"]
    Z = params["Z"]

    beta2 = math.radians(params["blade_angle"])   # outlet blade angle
    alpha2 = math.radians(params["alpha2"])       # flow angle
    beta1 = math.radians(30.0)   # 

    omega = rpm * 2 * math.pi / 60

    # -----------------------
    # Blade speeds
    # -----------------------
    U2 = omega * r2
    U1 = omega * r1

    # -----------------------
    # Approximate the meridional velocity using a bounded flow coefficient.
    # -----------------------
    phi = K_PHI * math.sin(beta2)
    phi *= 1 - K_BLOCKAGE * ((Z - Z_REF) / Z_REF)
    phi = max(PHI_MIN, min(PHI_MAX, phi))
    Ca = phi * U2

    # -----------------------
    # Velocity triangles
    # -----------------------
    W1 = Ca / math.sin(beta1 + 1e-6)
    W2 = Ca / math.sin(beta2 + 1e-6)

    Cu2 = U2 - W2 * math.cos(beta2)

    # leakage velocity (very crude but acceptable for now)
    Uc = 0.1 * U2

    return {
        "U2": U2,
        "U1": U1,
        "W1": W1,
        "W2": W2,
        "Cu2": Cu2,
        "Ca": Ca,
        "phi": phi,
        "Uc": Uc
    }
def compute_mass_flow(params, velocities):
    rho = params["rho1"]
    r2 = params["r2"]
    b2 = params["b2"]

    A = 2 * math.pi * r2 * b2   # annulus area
    Ca = velocities["Ca"]

    return rho * A * Ca
