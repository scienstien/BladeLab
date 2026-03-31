import math

def compute_velocity_triangles(params):
    rpm = params["rpm"]
    r2 = params["r2"]
    r1 = params["r1"]

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
    # Assume axial inlet flow (simplification)
    # -----------------------
    Ca = 0.3 * U2   # axial velocity (you can tune this later)

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
        "Uc": Uc
    }
def compute_mass_flow(params, velocities):
    rho = params["rho1"]
    r2 = params["r2"]
    b2 = params["b2"]

    A = 2 * math.pi * r2 * b2   # annulus area
    Ca = velocities["Ca"]

    return rho * A * Ca