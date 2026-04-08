import math

from env.config import COEFFS
from env.velocity import compute_mass_flow, compute_velocity_triangles

EPS = 1e-6


def _safe_div(numerator, denominator):
    return numerator / (denominator if abs(denominator) > EPS else (EPS if denominator >= 0 else -EPS))


def _smooth_positive(value):
    return 0.5 * (value + math.sqrt(value * value + EPS))


def blade_loading_loss(params):
    W1 = max(params["W1"], EPS)
    W2 = max(params["W2"], EPS)
    U2 = max(abs(params["U2"]), EPS)
    D1 = max(params["D1"], EPS)
    D2 = max(params["D2"], D1 + EPS)
    z = max(params["Z"], 1)
    b2 = max(params.get("b2", 0.01), EPS)
    Cu2 = abs(params.get("Cu2", 0.0))

    mean_radius = 0.25 * (D1 + D2)
    blade_pitch = 2.0 * math.pi * mean_radius / z
    chord = math.hypot(0.5 * (D2 - D1), b2)
    solidity = max(chord / max(blade_pitch, EPS), 0.3)
    diffusion_factor = 1.0 - _safe_div(W2, W1) + _safe_div(Cu2, 2.0 * solidity * W1)
    diffusion_factor = _smooth_positive(diffusion_factor)

    return COEFFS["k_bl"] * 0.5 * diffusion_factor**2 * U2**2


def incidence_loss(params):
    W1 = max(params["W1"], EPS)
    U1 = max(abs(params.get("U1", 0.0)), EPS)
    Ca = abs(params.get("Ca", 0.0))
    beta1_metal = math.radians(params.get("inlet_blade_angle", 30.0))
    beta1_flow = math.atan2(Ca, U1)
    incidence = beta1_metal - beta1_flow

    return COEFFS["k_inc"] * 0.5 * W1**2 * (math.sin(incidence) ** 2) * (1.0 + _safe_div(W1, U1))


def disk_friction_loss(params):
    rho = max(params["rho1"], EPS)
    mu = max(params["mu"], EPS)
    U2 = max(abs(params["U2"]), EPS)
    r2 = max(params["r2"], EPS)
    m_dot = max(params["m_dot"], EPS)

    reynolds = max(rho * U2 * r2 / mu, EPS)
    friction_coeff = 0.031 / (reynolds**0.2)

    return friction_coeff * rho * U2**3 * r2**2 / m_dot


def skin_friction_loss(params):
    Cf = COEFFS["k_sf"]

    D1 = params["D1"]
    D2 = params["D2"]
    b2 = max(params["b2"], EPS)
    b1 = max(params.get("b1", 1.35 * b2), EPS)
    W1 = max(params["W1"], EPS)
    W2 = max(params["W2"], EPS)
    Z = max(params.get("Z", 1), 1)

    r1 = D1 / 2.0
    r2 = D2 / 2.0
    r_mean = 0.5 * (r1 + r2)
    b_mean = 0.5 * (b1 + b2)

    pitch = 2.0 * math.pi * r_mean / Z
    area = b_mean * pitch
    wetted_perimeter = 2.0 * (b_mean + pitch)
    hydraulic_diameter = max(4.0 * area / max(wetted_perimeter, EPS), EPS)

    beta1 = math.radians(params.get("inlet_blade_angle", 30.0))
    beta2 = math.radians(params["blade_angle"])
    beta_m = 0.5 * (beta1 + beta2)

    path_length = abs(r2 - r1) / max(abs(math.cos(beta_m)), 0.1)
    mean_relative_velocity = (W1 + W2) / 2.0

    return 0.5 * Cf * (path_length / hydraulic_diameter) * mean_relative_velocity**2


def clearance_loss(params):
    epsilon = max(params["clearance"], 0.0)
    b2 = max(params["b2"], EPS)
    Z = max(params["Z"], 1)

    r2 = max(params["r2"], EPS)
    rs1 = max(params["r1"], EPS)
    b1 = max(params.get("b1", 1.35 * b2), EPS)
    rh1 = max(params.get("rh1", rs1 - b1), EPS)
    rho1 = max(params["rho1"], EPS)
    rho2 = max(params["rho2"], EPS)
    U2 = max(abs(params["U2"]), EPS)
    Ctheta2 = params["Cu2"]
    Cm2 = abs(params["Ca"])

    geom_denominator = max((r2 - rs1) * (1.0 + rho2 / rho1), EPS)
    geom_term = max((rs1**2 - rh1**2) / geom_denominator, 0.0)
    flow_term = max(_safe_div(Ctheta2, U2) * _safe_div(Cm2, U2), 0.0)
    normalized_loss = (
        0.6
        * (epsilon / b2)
        * abs(Ctheta2) / U2
        * math.sqrt(max((4.0 * math.pi / max(b2 * Z, EPS)) * geom_term * flow_term, 0.0))
    )
    return normalized_loss * U2**2


def leakage_loss(params):
    clearance = max(params.get("clearance", 0.0), 0.0)
    b2 = max(params.get("b2", 0.01), EPS)
    U2 = max(abs(params["U2"]), EPS)
    Uc = abs(params.get("Uc", 0.1 * U2))
    Cu2 = max(params.get("Cu2", 0.0), 0.0)
    rho2 = max(params.get("rho2", params.get("rho1", 1.0)), EPS)
    k_leak = params.get("k_leak", COEFFS.get("k_leak", 0.02))

    delta_p = rho2 * max(U2 * Cu2, 0.0)
    leakage_velocity = math.sqrt(max(2.0 * delta_p / rho2, 0.0) + EPS)
    leakage_factor = k_leak * (clearance / b2) * (1.0 + _safe_div(Uc, U2))

    return 0.5 * leakage_factor * leakage_velocity**2


def recirculation_loss(params):
    U2 = max(abs(params["U2"]), EPS)
    Ca = abs(params.get("Ca", 0.0))
    Cu2 = max(abs(params.get("Cu2", 0.0)), EPS)
    alpha2_target = math.radians(params.get("alpha2", 60.0))
    alpha2_flow = math.atan2(Ca, Cu2)
    flow_deviation = alpha2_target - alpha2_flow
    phi = params.get("phi", _safe_div(Ca, U2))

    return 0.5 * 0.02 * U2**2 * (math.tan(flow_deviation) ** 2) * (1.0 + max(phi, 0.0))


def compute_losses(params):
    return {
        "blade_loading": blade_loading_loss(params),
        "incidence": incidence_loss(params),
        "disk_friction": disk_friction_loss(params),
        "skin_friction": skin_friction_loss(params),
        "clearance": clearance_loss(params),
        "leakage": leakage_loss(params),
        "recirculation": recirculation_loss(params),
    }


def compute_efficiency(head, losses):
    head = max(head, EPS)
    losses = max(losses, 0.0)
    return min(0.98, max(EPS, head / (head + losses + EPS)))


def compute_pressure_ratio(head, losses, params):
    gamma = params.get("gamma", 1.4)
    cp = max(params["Cp"], EPS)
    t1 = max(params["T1"], EPS)
    efficiency = compute_efficiency(head, losses)
    isentropic_head = max(efficiency * max(head, 0.0), 0.0)
    temperature_ratio = 1.0 + isentropic_head / (cp * t1)
    raw_pressure_ratio = max(1.0, temperature_ratio ** (gamma / max(gamma - 1.0, EPS)))
    max_pressure_ratio = max(params.get("max_pressure_ratio", 6.0), 1.0 + EPS)
    excess_ratio = raw_pressure_ratio - 1.0
    available_ratio = max_pressure_ratio - 1.0

    return 1.0 + available_ratio * excess_ratio / (available_ratio + excess_ratio + EPS)


def build_physics_inputs(state):
    params = dict(state)
    params["D1"] = 2.0 * params["r1"]
    params["D2"] = 2.0 * params["r2"]

    velocities = compute_velocity_triangles(params)
    params.update(velocities)

    params["m_dot"] = compute_mass_flow(params, velocities)
    return params


def compute_physics(state):
    params = build_physics_inputs(state)

    loss_breakdown = compute_losses(params)
    total_losses = sum(loss_breakdown.values())
    head = params["U2"] * params["Cu2"]
    efficiency = compute_efficiency(head, total_losses)
    pressure_ratio = compute_pressure_ratio(head, total_losses, params)

    return {
        **params,
        "mass_flow": params["m_dot"],
        "head": head,
        "losses": total_losses,
        "loss_breakdown": loss_breakdown,
        "efficiency": efficiency,
        "pressure_ratio": pressure_ratio,
    }
