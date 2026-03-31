import math
from env.config import COEFFS
from env.velocity import compute_velocity_triangles, compute_mass_flow


# --- LOSSES ---

def blade_loading_loss(params):
    W1 = params["W1"]
    W2 = params["W2"]
    U2 = params["U2"]
    D1 = params["D1"]
    D2 = params["D2"]
    z = params["Z"]
    Cp = params["Cp"]
    T1 = params["T1"]
    T2 = params["T2"]

    K_bl = COEFFS["k_bl"]

    term1 = 1 - (W2 / W1)
    term2 = (Cp * (T2 - T1)) / (U2**2)
    term3 = W1 / U2
    term4 = z / math.pi * (1 - D1 / D2) + 2 * D1 / D2

    loss = 0.5 * (term1 + term2 / (term3 * term4))**2 * U2**2
    return K_bl * loss


def incidence_loss(params):
    W1 = params["W1"]
    F_mc = COEFFS["k_inc"]

    return F_mc * (W1**2) / 2


def disk_friction_loss(params):
    rho = params["rho1"]
    mu = params["mu"]
    U2 = params["U2"]
    r2 = params["r2"]
    m_dot = params["m_dot"]

    term1 = 0.0402 * rho * (U2 / r2)**3 * (r2**5)
    term2 = (U2 * r2 * rho / mu)**0.2

    return term1 / (term2 * m_dot)


def skin_friction_loss(params):
    Cf = COEFFS["k_sf"]

    W = params["W2"]
    D1 = params["D1"]
    D2 = params["D2"]

    return 2 * Cf * ((W + D2/2) / ((D1 + D2)/2)) * W**2


def clearance_loss(params):
    epsilon = params["clearance"]
    b2 = params["b2"]
    Z = params["Z"]

    r1 = params["r1"]
    r2 = params["r2"]

    W2 = params["W2"]
    rho1 = params["rho1"]
    rho2 = params["rho2"]

    term = (r2**2 - r1**2) / ((r2 - r1) * (1 - rho2 / rho1))

    loss = 0.6 * epsilon * W2 / b2 * ((4 * math.pi) / (b2 * Z) * term)**2
    return loss


def leakage_loss(params):
    m_dot = params["m_dot"]
    Uc = params["Uc"]
    U2 = params["U2"]

    return (m_dot * Uc * U2) / (2 * m_dot + 1e-6)


def recirculation_loss(params):
    W1 = params["W1"]
    W2 = params["W2"]
    U2 = params["U2"]

    D1 = params["D1"]
    D2 = params["D2"]
    z = params["Z"]

    Cp = params["Cp"]
    T1 = params["T1"]
    T2 = params["T2"]

    alpha2 = math.radians(params["alpha2"])
    K_bl = COEFFS["k_bl"]

    term1 = 1 - W2 / W1
    term2 = (Cp * (T2 - T1)) / (U2**2)
    term3 = W1 / U2
    term4 = z / math.pi * (1 - D1 / D2) + 2 * D1 / D2

    core = term1 + term2 / (term3 * term4)

    return 0.02 * math.tan(alpha2) * (core**2) * U2**2


# -------------------------------
# TOTAL LOSS
# -------------------------------

def compute_losses(params):
    return (
        blade_loading_loss(params)
        + incidence_loss(params)
        + disk_friction_loss(params)
        + skin_friction_loss(params)
        + clearance_loss(params)
        + leakage_loss(params)
        + recirculation_loss(params)
    )

def compute_efficiency(H, losses):
    return H / (H + losses + 1e-6)

def compute_pressure_ratio(H, losses, params):
    gamma = 1.2

    H_net = max(0.0, H - losses)

    term = 1 + H_net / (params["Cp"] * params["T"])

    # safety clamp (prevents weird negatives or explosions)
    term = max(1e-6, term)

    return term ** (gamma / (gamma - 1))

# def compute_physics(params):
#     m_dot = compute_flow(params)
#     H = compute_head(params)
#     losses = compute_losses(params, m_dot)
#     eff = compute_efficiency(H, losses)
#     PR = compute_pressure_ratio(H, losses ,params)

#     return {
#         "mass_flow": m_dot,
#         "head": H,
#         "losses": losses,
#         "efficiency": eff,
#         "pressure_ratio": PR
#     }
def compute_physics(params):
    
    # 1. velocities
    vel = compute_velocity_triangles(params)

    params.update(vel)

    # 2. mass flow
    m_dot = compute_mass_flow(params, vel)
    params["m_dot"] = m_dot

    # 3. losses
    losses = compute_losses(params)

    # 4. head (fix this too)
    H = vel["U2"] * vel["Cu2"]

    # 5. efficiency
    eff = H / (H + losses + 1e-6)

    # 6. pressure ratio
    gamma = 1.2
    term = 1 + (H - losses) / (params["Cp"] * params["T1"])
    term = max(1e-6, term)

    PR = term ** (gamma / (gamma - 1))

    return {
        "mass_flow": m_dot,
        "head": H,
        "losses": losses,
        "efficiency": eff,
        "pressure_ratio": PR
    }