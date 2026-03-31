from env.config import COEFFS, PR_TARGET

def compute_reward(curr, prev, constraints):
    reward = 0.0

    eff = curr["efficiency"]
    PR = curr["pressure_ratio"]
    phi = constraints["phi"]   # normalized mass flow

    # --- 1. Absolute objective ---
    reward += 5.0 * eff

    # --- 2. Pressure ratio targeting ---
    reward -= 8.0 * (PR - PR_TARGET)**2

    # --- 3. Surge avoidance (continuous) ---
    reward -= 15.0 * max(0, 0.75 - phi)**2

    # --- 4. Choke avoidance ---
    reward -= 10.0 * max(0, phi - 1.05)**2

    # --- 5. Stability bonus ---
    if abs(PR - PR_TARGET) < 0.1 and eff > 0.75:
        reward += 2.0

    # --- 6. Small step penalty ---
    reward -= 0.01

    return reward