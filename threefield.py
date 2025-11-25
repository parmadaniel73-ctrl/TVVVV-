# ================================================================
# TVVVV Three-Field Cosmology – PURE PYTHON – Pydroid 3 ready
# Fields: S(t), phi(t), scale factor a(t)
# No NumPy, no SciPy, no matplotlib → runs everywhere
# Author: Daniel Parma + Grok 2025
# ================================================================

import math
import time

start = time.time()

# ------------------- Parametry modelu -------------------
gamma_S   = 1.0       # kinetický koeficient S
lambda_c  = 0.5       # vazba S ↔ ϕ
U0        = 1e-120    # konstanta v U(S) – odpovídá dnešní Λ
U2        = 1.0       # tuhost potenciálu S

m_phi     = 0.5       # hmotnost pole ϕ
rho_m0    = 0.3       # dnešní hustota hmoty (v Planckových jednotkách ≈ 0.3)
rho_r0    = 1e-4      # radiace (zanedbatelná dnes)
rho_L     = 1e-120    # klasická Λ (téměř nula)

# Počáteční podmínky (dnešní doba = t = t_max)
t_max = 14.0          # ~14 miliard let v libovolných jednotkách
steps = 5000
dt = t_max / steps

S     = 1.2          # dnešní hodnota Sₘ (určuje Λeff)
Sdot  = 0.0
phi   = 0.0
phidot = 0.0
a     = 1.0          # dnešní a = 1

# Seznamy pro výstup
t_list = [0.0]
S_list = [S]
a_list = [a]
H_list = []
rhoS_list = []
rhoPhi_list = []
Lambda_eff_list = []

# ------------------- Potenciály -------------------
def U(S):
    return U0 + 0.5 * U2 * S * S

def dU_dS(S):
    return U2 * S

def V(phi):
    return 0.5 * m_phi * m_phi * phi * phi

def dV_dphi(phi):
    return m_phi * m_phi * phi

# ------------------- Hustoty -------------------
def rho_S(S, Sdot):
    return 0.5 * gamma_S * Sdot* Sdot + U(S)

def rho_phi(phi, phidot, S):
    return 0.5 * phidot* phidot + V(phi) + lambda_c * S * phi

def total_rho(a, S, Sdot, phi, phidot):
    rho_m = rho_m0 / (a * a * a)
    rho_r = rho_r0 / (a * a * a * a)
    return rho_m + rho_r + rho_S(S, Sdot) + rho_phi(phi, phidot, S) + rho_L

# ------------------- RK4 krok -------------------
def rk4_step(S, Sdot, phi, phidot, a):
    # Friedmann
    rho = total_rho(a, S, Sdot, phi, phidot)
    H = math.sqrt(max(rho / 3.0, 0.0))

    # S'' = -3H Sdot - dU/dS - lambda ϕ
    Sddot = -3.0 * H * Sdot - dU_dS(S) - lambda_c * phi

    # ϕ'' = -3H ϕdot - dV/dϕ - lambda S
    phiddot = -3.0 * H * phidot - dV_dphi(phi) - lambda_c * S

    # a' = a H
    adot = a * H

    return Sdot, Sddot, phidot, phiddot, adot

# ------------------- Hlavní smyčka -------------------
print("TVVVV Three-Field Cosmology – running pure Python...")
for step in range(steps):
    # RK4
    k1_Sdot, k1_Sddot, k1_phidot, k1_phiddot, k1_adot = rk4_step(S, Sdot, phi, phidot, a)
    k2_Sdot, k2_Sddot, k2_phidot, k2_phiddot, k2_adot = rk4_step(
        S + 0.5*dt*k1_Sdot, Sdot + 0.5*dt*k1_Sddot,
        phi + 0.5*dt*k1_phidot, phidot + 0.5*dt*k1_phiddot,
        a + 0.5*dt*k1_adot)
    k3_Sdot, k3_Sddot, k3_phidot, k3_phiddot, k3_adot = rk4_step(
        S + 0.5*dt*k2_Sdot, Sdot + 0.5*dt*k2_Sddot,
        phi + 0.5*dt*k2_phidot, phidot + 0.5*dt*k2_phiddot,
        a + 0.5*dt*k2_adot)
    k4_Sdot, k4_Sddot, k4_phidot, k4_phiddot, k4_adot = rk4_step(
        S + dt*k3_Sdot, Sdot + dt*k3_Sddot,
        phi + dt*k3_phidot, phidot + dt*k3_phiddot,
        a + dt*k3_adot)

    S     += dt * (k1_Sdot  + 2*k2_Sdot  + 2*k3_Sdot  + k4_Sdot)  / 6.0
    Sdot  += dt * (k1_Sddot + 2*k2_Sddot + 2*k3_Sddot + k4_Sddot)/ 6.0
    phi   += dt * (k1_phidot  + 2*k2_phidot  + 2*k3_phidot  + k4_phidot)  / 6.0
    phidot+= dt * (k1_phiddot + 2*k2_phiddot + 2*k3_phiddot + k4_phiddot)/ 6.0
    a     += dt * (k1_adot + 2*k2_adot + 2*k3_adot + k4_adot) / 6.0

    if a <= 0: a = 1e-12

    # Ukládání každých 100 kroků
    if step % 100 == 0 or step == steps-1:
        t = step * dt
        rho = total_rho(a, S, Sdot, phi, phidot)
        H = math.sqrt(max(rho / 3.0, 0.0))
        Lambda_eff = 3 * H * H - rho_m0/(a*a*a) - rho_r0/(a**4) - rho_L

        t_list.append(t)
        S_list.append(S)
        a_list.append(a)
        H_list.append(H)
        rhoS_list.append(rho_S(S, Sdot))
        rhoPhi_list.append(rho_phi(phi, phidot, S))
        Lambda_eff_list.append(Lambda_eff)

# ------------------- Výstup -------------------
print("\n=== TVVVV THREE-FIELD COSMOLOGY ===")
print(f"Cas: {t_list[-1]:.1f} (arbitr. units)")
print(f"Scale factor a = {a_list[-1]:.6f}")
print(f"Sₘ = {S_list[-1]:.6e}")
print(f"ϕ = {phi:.6e}")
print(f"H = {H_list[-1]:.6e}")
print(f"rho_S = {rhoS_list[-1]:.6e}")
print(f"rho_ϕ = {rhoPhi_list[-1]:.6e}")
print(f"Λ_eff ≈ {Lambda_eff_list[-1]:.6e} (dnes)")
print(f"\nHotovo za {time.time()-start:.2f} sekund na Pydroid 3!\n")

# Krátký vývoj a(t) a Λeff
print("Vývoj a(t):")
for i in range(0, len(t_list), len(t_list)//10):
    print(f"  t={t_list[i]:4.1f} → a={a_list[i]:.5f}  Λ_eff={Lambda_eff_list[i]:.3e}")