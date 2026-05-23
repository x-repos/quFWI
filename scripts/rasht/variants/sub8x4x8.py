"""
Full Waveform Inversion (FWI) — Rasht-Behesht et al. (2022) on FBPINNs

Faithfully translates the rasht-behesht acoustic FWI physics and SPECFEM data
into the FBPINNs framework (JAX + domain decomposition + vmap).

Key differences from fwi_v15:
  - Scalar potential formulation: network outputs phi(x',z',t), displacements
    are dphi/dx', dphi/dz'  (matching rasht-behesht exactly)
  - Coordinate scaling Lx=Lz=3
  - Soft ICs from SPECFEM wavefield snapshots (no hard constraining_fn)
  - Free-stress BC: P=0 at top surface z'=az/Lz
  - SPECFEM seismogram data (20 receivers, X+Z displacement components)
  - Velocity: alpha = 3 + 2*tanh(NN)*mask  (matching rasht-behesht exactly)

Run with: python fwi_v16.py
Resume:   python fwi_v16.py --resume 200000
"""

import os
import time
import pickle
import glob as glob_mod

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

from qufwi.fbpinns.domains import RectangularDomainND
from qufwi.fbpinns.problems import Problem
from qufwi.fbpinns.decompositions import RectangularDecompositionND
from qufwi.fbpinns.networks import FCN
from qufwi.fbpinns.constants import Constants
from qufwi.fbpinns.trainers import (FBPINNTrainer, get_inputs, FBPINN_model, FBPINN_model_jit,
                               _common_train_initialisation, tree_map_dicts,
                               FBPINN_update, partition)
from qufwi.fbpinns.util.logger import logger
from qufwi.fbpinns.util.jax_util import total_size, flops_cost_analysis
from tensorboardX import SummaryWriter

# Ensure working directory is scripts/ for relative result paths
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(SCRIPTS_DIR)


# ============================================================================
# RASHT-BEHESHT PARAMETERS (matching PINNs_Inversion_Acoustic.py exactly)
# ============================================================================

# SPECFEM grid
nx = 100
nz = 100
n_abs = 10
n_absx = n_abs
n_absz = n_abs

ax_spec = 1.5   # km — domain size in SPECFEM
az_spec = 0.5   # km
xsf = 1.3       # km — x location of seismometers in SPECFEM

dx_grid = ax_spec / nx
dz_grid = az_spec / nz
rho = 1.0

# PINN domain after removing absorbing layers
ax = xsf - n_absx * dx_grid    # 1.3 - 10*0.015 = 1.15 km
az = az_spec - n_absz * dz_grid  # 0.5 - 10*0.005 = 0.45 km

# Coordinate scaling
Lx = 3.0
Lz = 3.0

# Time parameters
t_m = 0.5    # total time for PDE training
t_st = 0.1   # first IC snapshot time
t_s = 0.5    # total seismogram time used
s_spec = 5e-5  # SPECFEM time step

t01 = 2000 * s_spec   # 0.1 s — first IC snapshot
t02 = 2300 * s_spec   # 0.115 s — second IC snapshot
t_la = 5000 * s_spec  # 0.25 s — test data time

# Seismometer configuration
n_event = 1
n_seis = 20
z0_s = az           # z location of first seismometer (surface)
zl_s = 0.06 - n_absz * dz_grid  # z location of last seismometer at depth

# Displacement scaling
u_scl = 1 / 3640

# Velocity parameters
INIT_VEL_PATH = os.path.join(SCRIPTS_DIR, "initial_velocity.npy")
if os.path.exists(INIT_VEL_PATH):
    VEL_BACKGROUND = float(np.load(INIT_VEL_PATH))
    print(f"Loaded initial velocity = {VEL_BACKGROUND} km/s from {INIT_VEL_PATH}")
else:
    VEL_BACKGROUND = 3.0     # km/s (default)
VEL_AMPLITUDE = 2.0      # alpha = 3 + 2*tanh(NN)*mask
VEL_LAYER_SIZES = (2, 20, 20, 20, 20, 20, 1)  # matching rasht layers0

# Inversion box (in physical km, before scaling by Lx/Lz)
z_st_phys = 0.1 - n_absz * dz_grid   # 0.05 km
z_fi_phys = 0.45 - n_absz * dz_grid  # 0.40 km
x_st_phys = 0.7 - n_absx * dx_grid   # 0.55 km
x_fi_phys = 1.25 - n_absx * dx_grid  # 1.10 km

# Convert inversion box to scaled coordinates for the mask
VEL_BOX = (x_st_phys / Lx, x_fi_phys / Lx,
           z_st_phys / Lz, z_fi_phys / Lz)
MASK_STEEPNESS = 1000.0

# Loss weights (matching rasht exactly)
W_PDE = 0.1
W_IC1 = 1.0
W_IC2 = 1.0
W_SEISMO = 1.0
W_BC = 0.1

# Seismogram subsampling
L_F = 100  # subsample every 100 steps from SPECFEM

# FBPINN configuration
N_SUBDOMAINS_X = 8
N_SUBDOMAINS_Z = 4
N_SUBDOMAINS_T = 8
OVERLAP_FRACTION = 0.35

SUBDOMAIN_LAYER_SIZES = [3, 32, 32, 16, 16, 1]

# Training configuration
LEARNING_RATE = 1e-4
TRAINING_STEPS = 500000

# BC sampling
BC_XN = 100
BC_TN = 50

# Collocation points per dimension for PDE constraint
PDE_BATCH_SHAPE = (40, 40, 25)
N_TEST = (40, 20, 10)

SHOW_PLOTS = False
SAVE_PLOTS = True
RUN_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPTS_DIR)),
                        "data", "rasht", "specfem")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_specfem_data(data_dir=None):
    """Load and preprocess SPECFEM data following rasht-behesht exactly.

    Returns dict with keys:
        X_init1, U_ini1x, U_ini1z   — first IC (t'=0)
        X_init2, U_ini2x, U_ini2z   — second IC (t'=t02-t01)
        X_S, Sx, Sz                  — seismogram coords and data
        X_BC_t                       — BC sampling points
        U_specx, U_specz             — test data at t_la
        xx, zz                       — evaluation grid
        l_sub                        — number of time samples per seismometer
    """
    if data_dir is None:
        data_dir = DATA_DIR

    event_dir = os.path.join(data_dir, "event1")

    # --- Wavefield grid ---
    X0 = np.loadtxt(os.path.join(event_dir, "wavefields", "wavefield_grid_for_dumps_000.txt"))
    X0 = X0 / 1000  # m -> km
    X0[:, 0:1] = X0[:, 0:1] / Lx  # scale x
    X0[:, 1:2] = X0[:, 1:2] / Lz  # scale z
    xz = np.concatenate((X0[:, 0:1], X0[:, 1:2]), axis=1)

    # --- Interpolation grid (40x40 in non-absorbing domain) ---
    n_ini = 40
    xx, zz = np.meshgrid(np.linspace(0, ax / Lx, n_ini),
                          np.linspace(0, az / Lz, n_ini))

    # IC coordinate arrays
    X_init1 = np.concatenate((xx.reshape((-1, 1)),
                               zz.reshape((-1, 1)),
                               0.0 * np.ones((n_ini**2, 1))), axis=1)
    X_init2 = np.concatenate((xx.reshape((-1, 1)),
                               zz.reshape((-1, 1)),
                               (t02 - t01) * np.ones((n_ini**2, 1))), axis=1)

    # SPECFEM interpolation grid (in SPECFEM's non-absorbing domain, scaled)
    xf = n_absx * dx_grid  # start of non-absorbing domain
    zf = n_absz * dz_grid
    xxs, zzs = np.meshgrid(np.linspace(xf / Lx, xsf / Lx, n_ini),
                            np.linspace(zf / Lz, az_spec / Lz, n_ini))
    xxzzs = np.concatenate((xxs.reshape((-1, 1)), zzs.reshape((-1, 1))), axis=1)

    # --- Load wavefields ---
    wf_dir = os.path.join(event_dir, "wavefields")
    wfs = sorted(os.listdir(wf_dir))
    # Filter to only wavefield data files (not the grid file)
    wf_files = [f for f in wfs if f.startswith("wavefield0")]
    U0 = [np.loadtxt(os.path.join(wf_dir, f)) for f in wf_files]

    # IC1: wavefield at t01 (file index 0 = wavefield0002000)
    U_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
    U_ini1x = U_ini1[:, 0:1] / u_scl
    U_ini1z = U_ini1[:, 1:2] / u_scl

    # IC2: wavefield at t02 (file index 1 = wavefield0002300)
    U_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
    U_ini2x = U_ini2[:, 0:1] / u_scl
    U_ini2z = U_ini2[:, 1:2] / u_scl

    # Test data: wavefield at t_la (file index 2 = wavefield0005000)
    U_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)
    U_specx = U_spec[:, 0:1] / u_scl
    U_specz = U_spec[:, 1:2] / u_scl

    # --- Z-component seismograms ---
    seis_dir = os.path.join(event_dir, "seismograms")
    sms = sorted(os.listdir(seis_dir))
    smsz = [f for f in sms if f[-6] == 'Z']
    seismo_listz = [np.loadtxt(os.path.join(seis_dir, f)) for f in smsz]

    # Time axis processing (matching rasht exactly)
    t_spec = -seismo_listz[0][0, 0] + seismo_listz[0][:, 0]
    cut_u = t_spec > t_s
    cut_l = t_spec < t_st
    l_su = len(cut_u) - sum(cut_u)
    l_sl = sum(cut_l)

    index = np.arange(l_sl, l_su, L_F)
    l_sub = len(index)
    t_spec_sub = t_spec[index].reshape((-1, 1))
    t_spec_sub = t_spec_sub - t_spec_sub[0]  # shift to start from 0

    for ii in range(len(seismo_listz)):
        seismo_listz[ii] = seismo_listz[ii][index]

    Sz = seismo_listz[0][:, 1].reshape(-1, 1)
    for ii in range(len(seismo_listz) - 1):
        Sz = np.concatenate((Sz, seismo_listz[ii + 1][:, 1].reshape(-1, 1)), axis=0)
    Sz = Sz / u_scl

    # --- X-component seismograms ---
    smsx = [f for f in sms if f[-6] == 'X']
    seismo_listx = [np.loadtxt(os.path.join(seis_dir, f)) for f in smsx]

    for ii in range(len(seismo_listx)):
        seismo_listx[ii] = seismo_listx[ii][index]

    Sx = seismo_listx[0][:, 1].reshape(-1, 1)
    for ii in range(len(seismo_listx) - 1):
        Sx = np.concatenate((Sx, seismo_listx[ii + 1][:, 1].reshape(-1, 1)), axis=0)
    Sx = Sx / u_scl

    # --- Seismometer coordinates (X_S) ---
    d_s = np.abs(zl_s - z0_s) / (n_seis - 1)
    X_S = np.empty([int(np.size(Sz)), 3])
    for i in range(len(seismo_listz)):
        X_S[i * l_sub:(i + 1) * l_sub, :] = np.concatenate(
            (ax / Lx * np.ones((l_sub, 1)),
             (z0_s - i * d_s) / Lz * np.ones((l_sub, 1)),
             t_spec_sub), axis=1)

    # --- BC points (free stress at top surface z'=az/Lz) ---
    x_vec = np.random.rand(BC_XN, 1) * ax / Lx
    t_vec = np.random.rand(BC_TN, 1) * (t_m - t_st)
    xxb, ttb = np.meshgrid(x_vec.ravel(), t_vec.ravel())
    X_BC_t = np.concatenate((xxb.reshape((-1, 1)),
                              az / Lz * np.ones((xxb.reshape((-1, 1)).shape[0], 1)),
                              ttb.reshape((-1, 1))), axis=1)

    return {
        "X_init1": X_init1, "U_ini1x": U_ini1x, "U_ini1z": U_ini1z,
        "X_init2": X_init2, "U_ini2x": U_ini2x, "U_ini2z": U_ini2z,
        "X_S": X_S, "Sx": Sx, "Sz": Sz,
        "X_BC_t": X_BC_t,
        "U_specx": U_specx, "U_specz": U_specz,
        "xx": xx, "zz": zz,
        "l_sub": l_sub,
    }


# ============================================================================
# PROBLEM CLASS
# ============================================================================

class AcousticFWIScalarPotential(Problem):
    """Acoustic FWI using scalar potential formulation (Rasht-Behesht et al.).

    Network outputs phi(x',z',t). Displacements are dphi/dx', dphi/dz'.

    PDE: d²phi/dt² - alpha² * ((1/Lx²)*d²phi/dx'² + (1/Lz²)*d²phi/dz'²) = 0

    Constraints:
      0: PDE (interior collocation)
      1: IC1 (displacement at t'=0)
      2: IC2 (displacement at t'=t02-t01)
      3: Seismograms (displacement at receivers)
      4: BC (free stress P=0 at z'=az/Lz)
    """

    @staticmethod
    def init_params(
        Lx=3.0, Lz=3.0,
        vel_background=3.0, vel_amplitude=2.0,
        vel_layer_sizes=(2, 20, 20, 20, 20, 20, 1),
        vel_box=(0.0, 1.0, 0.0, 1.0),
        mask_steepness=1000.0,
        X_init1=None, U_ini1x=None, U_ini1z=None,
        X_init2=None, U_ini2x=None, U_ini2z=None,
        X_S=None, Sx=None, Sz=None,
        X_BC_t=None,
        U_specx=None, U_specz=None,
        xx_eval=None, zz_eval=None,
        w_pde=0.1, w_ic1=1.0, w_ic2=1.0, w_seismo=1.0, w_bc=0.1,
    ):
        # Initialize velocity network weights (uniform, matching quantum script)
        key = jax.random.PRNGKey(42)
        key, classical_key, _, _, _ = jax.random.split(key, 5)
        vel_layers = []
        keys = jax.random.split(classical_key, len(vel_layer_sizes) - 1)
        for k, m, n in zip(keys, vel_layer_sizes[:-1], vel_layer_sizes[1:]):
            w_key, b_key = jax.random.split(k)
            v = jnp.sqrt(1.0 / m)
            w = jax.random.uniform(w_key, (n, m), minval=-v, maxval=v)
            b = jax.random.uniform(b_key, (n,), minval=-v, maxval=v)
            vel_layers.append((w, b))

        trainable_params = {
            "vel_layers": vel_layers,
        }

        # Upper bounds for input normalization (matching rasht's ub0)
        ub0 = jnp.array([ax / Lx, az / Lz])

        static_params = {
            "dims": (1, 3),  # ud=1 (phi), xd=3 (x', z', t)
            "Lx": float(Lx),
            "Lz": float(Lz),
            "c_fn": AcousticFWIScalarPotential.c_fn,
            "vel_background": float(vel_background),
            "vel_amplitude": float(vel_amplitude),
            "vel_box": jnp.array(vel_box),
            "mask_steepness": float(mask_steepness),
            "ub0": ub0,
            # IC data
            "X_init1": jnp.array(X_init1),
            "U_ini1x": jnp.array(U_ini1x),
            "U_ini1z": jnp.array(U_ini1z),
            "X_init2": jnp.array(X_init2),
            "U_ini2x": jnp.array(U_ini2x),
            "U_ini2z": jnp.array(U_ini2z),
            # Seismogram data
            "X_S": jnp.array(X_S),
            "Sx": jnp.array(Sx),
            "Sz": jnp.array(Sz),
            # BC data
            "X_BC_t": jnp.array(X_BC_t),
            # Test data
            "U_specx": jnp.array(U_specx) if U_specx is not None else None,
            "U_specz": jnp.array(U_specz) if U_specz is not None else None,
            "xx_eval": jnp.array(xx_eval) if xx_eval is not None else None,
            "zz_eval": jnp.array(zz_eval) if zz_eval is not None else None,
            # Loss weights
            "w_pde": float(w_pde),
            "w_ic1": float(w_ic1),
            "w_ic2": float(w_ic2),
            "w_seismo": float(w_seismo),
            "w_bc": float(w_bc),
        }

        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        p = all_params["static"]["problem"]

        # Constraint 0: PDE (interior collocation)
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        # Need phi_xx, phi_zz, phi_tt for the wave equation
        required_ujs_phys = (
            (0, (0, 0)),  # d²phi/dx'²
            (0, (1, 1)),  # d²phi/dz'²
            (0, (2, 2)),  # d²phi/dt²
        )

        # Constraint 1: IC1 — displacement at t'=0
        # Need dphi/dx' and dphi/dz'
        required_ujs_ic1 = (
            (0, (0,)),  # dphi/dx' = u_x
            (0, (1,)),  # dphi/dz' = u_z
        )

        # Constraint 2: IC2 — displacement at t'=t02-t01
        required_ujs_ic2 = (
            (0, (0,)),
            (0, (1,)),
        )

        # Constraint 3: Seismograms — displacement at receivers
        required_ujs_seismo = (
            (0, (0,)),
            (0, (1,)),
        )

        # Constraint 4: BC — free stress P=0 at top surface
        # P = (1/Lx²)*phi_xx + (1/Lz²)*phi_zz
        required_ujs_bc = (
            (0, (0, 0)),  # d²phi/dx'²
            (0, (1, 1)),  # d²phi/dz'²
        )

        return [
            [x_batch_phys, required_ujs_phys],
            [p["X_init1"], p["U_ini1x"], p["U_ini1z"], required_ujs_ic1],
            [p["X_init2"], p["U_ini2x"], p["U_ini2z"], required_ujs_ic2],
            [p["X_S"], p["Sx"], p["Sz"], required_ujs_seismo],
            [p["X_BC_t"], required_ujs_bc],
        ]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        # Identity — no hard constraints (soft ICs from SPECFEM)
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        p = all_params["static"]["problem"]
        c_fn = p["c_fn"]
        Lx_ = p["Lx"]
        Lz_ = p["Lz"]
        w_pde = p["w_pde"]
        w_ic1 = p["w_ic1"]
        w_ic2 = p["w_ic2"]
        w_seismo = p["w_seismo"]
        w_bc = p["w_bc"]

        # --- Constraint 0: PDE ---
        # constraints[0] = [x_batch, phi_xx, phi_zz, phi_tt]
        x_batch_pde, phi_xx, phi_zz, phi_tt = constraints[0]
        alpha = c_fn(all_params, x_batch_pde)  # (n, 1)
        P = (1.0 / Lx_**2) * phi_xx + (1.0 / Lz_**2) * phi_zz
        residual = phi_tt - alpha**2 * P
        loss_pde = jnp.mean(residual**2)

        # --- Constraint 1: IC1 (displacement at t'=0) ---
        # constraints[1] = [X_init1, U_ini1x, U_ini1z, dphi_dx, dphi_dz]
        _, U_ini1x, U_ini1z, phi_x_ic1, phi_z_ic1 = constraints[1]
        loss_ic1 = jnp.mean((phi_x_ic1 - U_ini1x)**2) + jnp.mean((phi_z_ic1 - U_ini1z)**2)

        # --- Constraint 2: IC2 (displacement at t'=t02-t01) ---
        _, U_ini2x, U_ini2z, phi_x_ic2, phi_z_ic2 = constraints[2]
        loss_ic2 = jnp.mean((phi_x_ic2 - U_ini2x)**2) + jnp.mean((phi_z_ic2 - U_ini2z)**2)

        # --- Constraint 3: Seismograms ---
        _, Sx_, Sz_, phi_x_seis, phi_z_seis = constraints[3]
        loss_seismo = jnp.mean((phi_x_seis - Sx_)**2) + jnp.mean((phi_z_seis - Sz_)**2)

        # --- Constraint 4: BC (free stress P=0 at z'=az/Lz) ---
        _, phi_xx_bc, phi_zz_bc = constraints[4]
        P_bc = (1.0 / Lx_**2) * phi_xx_bc + (1.0 / Lz_**2) * phi_zz_bc
        loss_bc = jnp.mean(P_bc**2)

        total = w_pde * loss_pde + w_ic1 * loss_ic1 + w_ic2 * loss_ic2 + w_seismo * loss_seismo + w_bc * loss_bc
        return total, {
            "pde": loss_pde,
            "snapshot1": loss_ic1,
            "snapshot2": loss_ic2,
            "seismogram": loss_seismo,
            "free_surface": loss_bc,
        }

    @staticmethod
    def c_fn(all_params, x_batch):
        """Velocity forward pass matching rasht-behesht exactly.

        Input (x',z') in scaled coords -> normalize to [-1,1] -> 5-hidden-layer tanh NN
        -> alpha = vel_background + vel_amplitude * tanh(NN) * mask
        """
        vel_layers = all_params["trainable"]["problem"]["vel_layers"]
        p = all_params["static"]["problem"]
        c0 = p["vel_background"]
        vel_amp = p["vel_amplitude"]
        vel_box = p["vel_box"]
        lld = p["mask_steepness"]
        ub0 = p["ub0"]

        # Extract spatial coords only
        x_raw = x_batch[:, 0:1]  # x'
        z_raw = x_batch[:, 1:2]  # z'

        # Normalize to [-1, 1] (matching rasht: H = 2*(X/ub0) - 1)
        h = jnp.concatenate([2.0 * x_raw / ub0[0] - 1.0,
                              2.0 * z_raw / ub0[1] - 1.0], axis=1)

        # Forward pass through velocity network
        for w, b in vel_layers[:-1]:
            h = jnp.tanh(h @ w.T + b)
        w, b = vel_layers[-1]
        alpha_star = jnp.tanh(h @ w.T + b)  # (n, 1), range [-1, 1]

        # Smooth spatial mask (product of 4 tanh sigmoids)
        mask = (0.5 * (1.0 + jnp.tanh(lld * (x_raw - vel_box[0]))) *
                0.5 * (1.0 + jnp.tanh(lld * (vel_box[1] - x_raw))) *
                0.5 * (1.0 + jnp.tanh(lld * (z_raw - vel_box[2]))) *
                0.5 * (1.0 + jnp.tanh(lld * (vel_box[3] - z_raw))))

        alpha = c0 + vel_amp * alpha_star * mask
        return alpha

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        # No analytical solution — return NaN so l1/l1n are NaN (avoids div-by-zero)
        return jnp.full((x_batch.shape[0], 1), jnp.nan)


# ============================================================================
# TRUE VELOCITY (for plotting)
# ============================================================================

def alpha_true_fn(x_phys, z_phys):
    """Compute the true acoustic wavespeed (Rasht-Behesht ellipsoidal anomaly).

    Args:
        x_phys, z_phys: physical coordinates in km (NOT scaled)
    """
    a, b = 0.18, 0.1
    c_center = 1.0 - n_absx * dx_grid  # 0.85 km
    d_center = 0.3 - n_absz * dz_grid  # 0.25 km
    g = (x_phys - c_center)**2 / a**2 + (z_phys - d_center)**2 / b**2
    return 3.0 - 0.25 * (1.0 + np.tanh(100 * (1.0 - g)))


# ============================================================================
# CUSTOM TRAINER
# ============================================================================

class FBPINNTrainerFWI16(FBPINNTrainer):
    """FBPINNTrainer subclass for Rasht-Behesht FWI with custom test plots."""

    def __init__(self, c, resume_step=0):
        self.resume_step = resume_step
        self.checkpoint_data = None
        self.metrics = {
            "step": [], "total_loss": [],
            "pde": [], "snapshot1": [], "snapshot2": [],
            "seismogram": [], "free_surface": [],
            "l1_velocity": [],
        }

        if resume_step > 0:
            pattern = os.path.join(c.model_out_dir, 'model_*.jax')
            checkpoint_files = sorted(glob_mod.glob(pattern))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoints found in {c.model_out_dir}")
            checkpoint_path = checkpoint_files[-1]
            print(f"Loading checkpoint: {checkpoint_path}")
            with open(checkpoint_path, "rb") as f:
                self.checkpoint_data = pickle.load(f)
            self.resume_step = self.checkpoint_data[0]
            print(f"Checkpoint loaded (step {self.resume_step})")

            # Load existing metrics so resume doesn't discard previous history
            metrics_path = os.path.join(c.summary_out_dir, "metrics.npz")
            if os.path.exists(metrics_path):
                saved = np.load(metrics_path)
                for k in self.metrics:
                    if k in saved:
                        arr = saved[k]
                        mask = arr <= self.resume_step if k == "step" else np.arange(len(arr)) < np.sum(saved["step"] <= self.resume_step)
                        self.metrics[k] = arr[mask].tolist()
                print(f"Loaded {len(self.metrics['step'])} previous metric entries")

            os.makedirs(c.summary_out_dir, exist_ok=True)
            os.makedirs(c.model_out_dir, exist_ok=True)
            c.save_constants_file()
            logger.info(c)
            writer = SummaryWriter(c.summary_out_dir)
            writer.add_text("constants", str(c).replace("\n", "  \n"))
            self.c, self.writer = c, writer
        else:
            super().__init__(c)

    def _report(self, i, pstep, fstep, u_test_losses, start0, start1, report_time,
                u_exact, x_batch_test, test_inputs, all_params, all_opt_states,
                model_fns, problem, decomposition,
                active, merge_active, active_opt_states, active_params, x_batch,
                lossval):
        """Override to capture loss components at summary_freq."""
        if lossval is not None and hasattr(self, '_loss_components'):
            self.metrics["step"].append(i)
            self.metrics["total_loss"].append(float(lossval))
            for k in ["pde", "snapshot1", "snapshot2", "seismogram", "free_surface"]:
                self.metrics[k].append(float(self._loss_components[k]))
            self.metrics["l1_velocity"].append(float('nan'))  # filled by _test

        result = super()._report(
            i, pstep, fstep, u_test_losses, start0, start1, report_time,
            u_exact, x_batch_test, test_inputs, all_params, all_opt_states,
            model_fns, problem, decomposition,
            active, merge_active, active_opt_states, active_params, x_batch,
            lossval)

        # Save metrics periodically alongside model checkpoints
        if i > 0 and i % self.c.model_save_freq == 0:
            self._save_metrics()

        return result

    def _compute_l1_velocity(self, all_params):
        """Compute L1 velocity error on evaluation grid."""
        p = all_params["static"]["problem"]
        xx_eval = np.array(p["xx_eval"]) if p["xx_eval"] is not None else None
        zz_eval = np.array(p["zz_eval"]) if p["zz_eval"] is not None else None
        if xx_eval is None:
            n_plot = 40
            xx_eval, zz_eval = np.meshgrid(
                np.linspace(0, ax / Lx, n_plot),
                np.linspace(0, az / Lz, n_plot))
        c_true = alpha_true_fn(xx_eval * Lx, zz_eval * Lz)
        grid_pts = np.column_stack([xx_eval.ravel(), zz_eval.ravel(),
                                     np.zeros(xx_eval.size)])
        c_fn = p["c_fn"]
        c_learned = np.array(c_fn(all_params, jnp.array(grid_pts))).reshape(xx_eval.shape)
        return float(np.mean(np.abs(c_learned - c_true)))

    def _save_metrics(self):
        """Save all metrics to NPZ file."""
        out_path = os.path.join(self.c.summary_out_dir, "metrics.npz")
        np.savez(out_path, **{k: np.array(v) for k, v in self.metrics.items()})
        print(f"Metrics saved to {out_path}")

    def _save_physical_data(self, i, all_params, model_fns, decomposition):
        """Save predicted velocity, snapshots, and seismograms to npz."""
        p = all_params["static"]["problem"]
        xx_eval = np.array(p["xx_eval"]) if p["xx_eval"] is not None else None
        zz_eval = np.array(p["zz_eval"]) if p["zz_eval"] is not None else None
        if xx_eval is None:
            n_plot = 40
            xx_eval, zz_eval = np.meshgrid(
                np.linspace(0, ax / Lx, n_plot),
                np.linspace(0, az / Lz, n_plot))

        # Learned velocity
        grid_pts = np.column_stack([xx_eval.ravel(), zz_eval.ravel(),
                                     np.zeros(xx_eval.size)])
        c_fn = p["c_fn"]
        c_learned = np.array(c_fn(all_params, jnp.array(grid_pts))).reshape(xx_eval.shape)

        # Predicted snapshots (IC1 and IC2)
        times = [0.0, t02 - t01]
        snapshots = {}
        for ti, t_val in enumerate(times):
            gp = np.column_stack([xx_eval.ravel(), zz_eval.ravel(),
                                   t_val * np.ones(xx_eval.size)])
            dphi_dx, dphi_dz = self._compute_fbpinn_displacement(
                jnp.array(gp), all_params, model_fns, decomposition)
            snapshots[f"dphi_dx_t{ti}"] = dphi_dx.reshape(xx_eval.shape)
            snapshots[f"dphi_dz_t{ti}"] = dphi_dz.reshape(xx_eval.shape)

        # Predicted seismograms
        X_S = np.array(p["X_S"])
        X_S_jnp = jnp.array(X_S)
        m = all_params["static"]["decomposition"]["m"]
        active_all = jnp.ones(m, dtype=int)
        takes, _, (_, _, _, cut_all, _) = get_inputs(
            X_S_jnp, active_all, all_params, decomposition)
        all_params_cut = {
            "static": cut_all(all_params["static"]),
            "trainable": cut_all(all_params["trainable"]),
        }
        def phi_fn(x_batch):
            u, *_ = FBPINN_model(all_params_cut, x_batch, takes, model_fns, verbose=False)
            return u
        tangent_x = jnp.zeros_like(X_S_jnp).at[:, 0].set(1.0)
        tangent_z = jnp.zeros_like(X_S_jnp).at[:, 1].set(1.0)
        _, seis_dx = jax.jvp(phi_fn, (X_S_jnp,), (tangent_x,))
        _, seis_dz = jax.jvp(phi_fn, (X_S_jnp,), (tangent_z,))

        phys_dir = os.path.join(self.c.summary_out_dir, "physical")
        os.makedirs(phys_dir, exist_ok=True)
        out_path = os.path.join(phys_dir, f"physical_{i:08d}.npz")
        np.savez(out_path,
                 c_learned=c_learned,
                 seis_x=np.array(seis_dx), seis_z=np.array(seis_dz),
                 **snapshots)
        print(f"Physical data saved to {out_path}")

    @staticmethod
    def _collocation_path(summary_dir):
        return os.path.join(summary_dir, "collocation.npz")

    def _save_collocation(self, x_batch_global, constraint_offsets_global, constraint_fs_global):
        """Save training point data for seamless resume."""
        path = self._collocation_path(self.c.summary_out_dir)
        np.savez(path, x_batch_global=np.array(x_batch_global))
        logger.info(f"Saved collocation points to {path}")

    def train(self):
        if self.resume_step == 0:
            # Wrap _common_train_initialisation to save collocation points
            import qufwi.fbpinns.trainers as _trainers
            _orig_init = _trainers._common_train_initialisation
            def _saving_init(*args, **kwargs):
                result = _orig_init(*args, **kwargs)
                # result = (optimiser, opt_states, opt_fn, loss_fn, key,
                #           constraints_global, x_batch_global, offsets, fs, jmapss,
                #           x_batch_test, u_exact)
                self._save_collocation(result[6], result[7], result[8])
                return result
            _trainers._common_train_initialisation = _saving_init
            try:
                result = super().train()
            finally:
                _trainers._common_train_initialisation = _orig_init
            self._save_metrics()
            return result

        # Resume training from checkpoint (same pattern as v15)
        c, writer = self.c, self.writer
        resume_step = self.resume_step

        _, ckpt_all_params, ckpt_all_opt_states, ckpt_active, ckpt_u_test_losses = self.checkpoint_data
        self.checkpoint_data = None

        to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
        ckpt_all_params = jax.tree_util.tree_map(to_jax, ckpt_all_params)
        ckpt_all_opt_states = jax.tree_util.tree_map(to_jax, ckpt_all_opt_states)
        ckpt_active = jnp.array(ckpt_active) if isinstance(ckpt_active, np.ndarray) else ckpt_active

        key = random.PRNGKey(c.seed)
        np.random.seed(c.seed)

        all_params = {"static": {}, "trainable": {}}
        domain, problem, decomposition = c.domain, c.problem, c.decomposition
        for tag, cl, kwargs in zip(
            ["domain", "problem", "decomposition"],
            [domain, problem, decomposition],
            [c.domain_init_kwargs, c.problem_init_kwargs, c.decomposition_init_kwargs],
        ):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]

        from jax import vmap
        network = c.network
        key, *subkeys = random.split(key, all_params["static"]["decomposition"]["m"] + 1)
        args_ = c.network_init_kwargs.values()
        ps_ = vmap(network.init_params, in_axes=(0,) + (None,) * len(args_))(
            jnp.array(subkeys), *args_
        )
        if ps_[0]: all_params["static"]["network"] = {"subdomain": ps_[0]}
        if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}

        model_fns = (
            decomposition.norm_fn, network.network_fn, decomposition.unnorm_fn,
            decomposition.window_fn, problem.constraining_fn,
        )

        all_params["trainable"] = ckpt_all_params["trainable"]
        if "network" in ckpt_all_params["static"]:
            all_params["static"]["network"] = ckpt_all_params["static"]["network"]
        logger.info(f"Resumed trainable parameters from step {resume_step}")

        remaining_steps = c.n_steps - resume_step
        scheduler = c.scheduler(all_params=all_params, n_steps=remaining_steps, **c.scheduler_kwargs)

        (optimiser, _, optimiser_fn, loss_fn, key,
         constraints_global, x_batch_global, constraint_offsets_global, constraint_fs_global, jmapss,
         x_batch_test, u_exact) = _common_train_initialisation(c, key, all_params, problem, domain)

        # Restore saved collocation points so training data is byte-identical
        coll_path = self._collocation_path(c.summary_out_dir)
        if os.path.exists(coll_path):
            saved_xbg = jnp.array(np.load(coll_path)["x_batch_global"])
            if saved_xbg.shape == x_batch_global.shape:
                x_batch_global = saved_xbg
                # Also update PDE constraint (first constraint, index 0)
                pde_size = int(constraint_offsets_global[1]) if len(constraint_offsets_global) > 1 else x_batch_global.shape[0]
                constraints_global[0][0] = x_batch_global[:pde_size]
                logger.info(f"Restored collocation points from {coll_path}")
            else:
                logger.info(f"Collocation shape mismatch ({saved_xbg.shape} vs {x_batch_global.shape}), using fresh")
        else:
            # First resume without saved collocation — save current for future resumes
            self._save_collocation(x_batch_global, constraint_offsets_global, constraint_fs_global)
            logger.info("Saved collocation points for future resumes")

        all_opt_states = ckpt_all_opt_states
        logger.info("Resumed optimizer states from checkpoint")
        del ckpt_all_params, ckpt_all_opt_states

        logger.info("Getting test data inputs..")
        active_test_ = jnp.ones(all_params["static"]["decomposition"]["m"], dtype=int)
        takes_, all_ims_, (_, _, _, cut_all_, _) = get_inputs(
            x_batch_test, active_test_, all_params, decomposition
        )
        test_inputs = (takes_, all_ims_, cut_all_)

        u_test_losses = ckpt_u_test_losses.tolist() if hasattr(ckpt_u_test_losses, 'tolist') else list(ckpt_u_test_losses)

        pstep, fstep = 0, 0
        start0, start1, report_time = time.time(), time.time(), 0.0
        merge_active, active_params, active_opt_states, fixed_params = None, None, None, None
        lossval = None
        active = ckpt_active

        for i, active_ in enumerate(scheduler):
            gi = i + resume_step

            if active_ is not None:
                active = active_

                if i != 0:
                    all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                    all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

                active, merge_active, active_opt_states, active_params, fixed_params, static_params, takess, constraints, x_batch = \
                    self._get_update_inputs(
                        gi, active, all_params, all_opt_states,
                        x_batch_global, constraints_global, constraint_fs_global,
                        constraint_offsets_global, decomposition, problem,
                    )

                startc = time.time()
                logger.info(f"[i: {gi}/{c.n_steps}] Compiling update step..")
                static_params_dynamic, static_params_static = partition(static_params)
                update = FBPINN_update.lower(
                    optimiser_fn, active_opt_states,
                    active_params, fixed_params, static_params_dynamic, static_params_static,
                    takess, constraints, model_fns, jmapss, loss_fn,
                ).compile()
                logger.info(f"[i: {gi}/{c.n_steps}] Compiling done ({time.time() - startc:.2f} s)")
                p, f = total_size(active_params["network"]), flops_cost_analysis(update.cost_analysis())

            lossval, self._loss_components, active_opt_states, active_params = update(
                active_opt_states, active_params, fixed_params, static_params_dynamic,
                takess, constraints,
            )
            pstep, fstep = pstep + p, fstep + f

            u_test_losses, start1, report_time = \
                self._report(
                    gi + 1, pstep, fstep, u_test_losses, start0, start1, report_time,
                    u_exact, x_batch_test, test_inputs, all_params, all_opt_states,
                    model_fns, problem, decomposition,
                    active, merge_active, active_opt_states, active_params, x_batch,
                    lossval,
                )

        # Save metrics to NPZ
        self._save_metrics()
        writer.close()
        logger.info(f"[i: {gi + 1}/{c.n_steps}] Training complete")

        all_params["trainable"] = merge_active(active_params, all_params["trainable"])
        all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

        return all_params

    def _test(self, x_batch_test, u_exact, u_test_losses, x_batch, test_inputs,
              i, pstep, fstep, start0, active, all_params, model_fns, problem, decomposition):

        # Skip super()._test() — exact_solution returns NaN so standard plots are blank.
        # Just track u_test_losses with NaN entries for compatibility.
        u_test_losses.append([i, pstep, fstep, time.time() - start0, float('nan'), float('nan')])

        # Compute L1 velocity error and update the metrics entry for this step
        try:
            l1_vel = self._compute_l1_velocity(all_params)
            if self.metrics["step"] and self.metrics["step"][-1] == i:
                self.metrics["l1_velocity"][-1] = l1_vel
            else:
                # Step 0: no _report entry yet, add one
                self.metrics["step"].append(i)
                self.metrics["total_loss"].append(float('nan'))
                for k in ["pde", "snapshot1", "snapshot2", "seismogram", "free_surface"]:
                    self.metrics[k].append(float('nan'))
                self.metrics["l1_velocity"].append(l1_vel)
            logger.info(f"[i: {i}] L1 velocity: {l1_vel:.6f}")
        except Exception as e:
            logger.info(f"[i: {i}] L1 velocity computation failed: {e}")

        # Custom plots: velocity, displacement, seismograms
        if i % (self.c.test_freq) == 0:
            try:
                f_vel = self._plot_velocity_comparison(i, all_params)
                f_disp = self._plot_displacement_comparison(i, all_params, model_fns, decomposition)
                figs = [("velocity", f_vel), ("displacement", f_disp)]

                try:
                    f_seis = self._plot_seismogram_comparison(i, all_params, model_fns, decomposition)
                    figs.append(("seismograms", f_seis))
                except Exception as e:
                    logger.info(f"Skipping seismogram plot: {e}")

                self._save_figs(i, figs)
            except Exception as e:
                logger.info(f"Skipping custom plots: {e}")

        # Save physical data at model_save_freq
        if i % self.c.model_save_freq == 0:
            try:
                self._save_physical_data(i, all_params, model_fns, decomposition)
            except Exception as e:
                logger.info(f"Skipping physical data save: {e}")

        return u_test_losses

    def _plot_velocity_comparison(self, i, all_params):
        """Compare true vs learned velocity on evaluation grid."""
        p = all_params["static"]["problem"]
        xx_eval = np.array(p["xx_eval"]) if p["xx_eval"] is not None else None
        zz_eval = np.array(p["zz_eval"]) if p["zz_eval"] is not None else None

        if xx_eval is None:
            n_plot = 40
            xx_eval, zz_eval = np.meshgrid(
                np.linspace(0, ax / Lx, n_plot),
                np.linspace(0, az / Lz, n_plot))

        # True velocity (in physical coords)
        c_true = alpha_true_fn(xx_eval * Lx, zz_eval * Lz)

        # Learned velocity
        grid_pts = np.column_stack([xx_eval.ravel(), zz_eval.ravel(),
                                     np.zeros(xx_eval.size)])
        grid_pts_jnp = jnp.array(grid_pts)
        c_fn = p["c_fn"]
        c_learned = np.array(c_fn(all_params, grid_pts_jnp)).reshape(xx_eval.shape)

        c_diff = c_learned - c_true
        l1_error = np.mean(np.abs(c_diff))

        vmin = min(c_true.min(), c_learned.min())
        vmax = max(c_true.max(), c_learned.max())

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        extent = [0, ax, 0, az]

        im0 = axes[0].contourf(xx_eval * Lx, zz_eval * Lz, c_true, 50, cmap='GnBu',
                                 vmin=vmin, vmax=vmax)
        axes[0].set_title('True alpha')
        fig.colorbar(im0, ax=axes[0], shrink=0.5, aspect=10)

        im1 = axes[1].contourf(xx_eval * Lx, zz_eval * Lz, c_learned, 50, cmap='GnBu',
                                 vmin=vmin, vmax=vmax)
        axes[1].set_title('Learned alpha')
        fig.colorbar(im1, ax=axes[1], shrink=0.5, aspect=10)

        im2 = axes[2].contourf(xx_eval * Lx, zz_eval * Lz, c_diff, 50, cmap='RdBu_r')
        axes[2].set_title(f'Difference (L1={l1_error:.4f})')
        fig.colorbar(im2, ax=axes[2], shrink=0.5, aspect=10)

        for ax_ in axes:
            ax_.set_xlabel('x (km)')
            ax_.set_ylabel('z (km)')
            ax_.set_aspect('equal')

        fig.suptitle(f'[{i}] Velocity Inversion', fontsize=12)
        fig.tight_layout()
        return fig

    def _compute_fbpinn_displacement(self, grid_pts_jnp, all_params, model_fns, decomposition):
        """Compute FBPINN displacement (dphi/dx', dphi/dz') at given points via jvp."""
        m = all_params["static"]["decomposition"]["m"]
        active_all = jnp.ones(m, dtype=int)
        takes, _, (_, _, _, cut_all, _) = get_inputs(
            grid_pts_jnp, active_all, all_params, decomposition)
        all_params_cut = {
            "static": cut_all(all_params["static"]),
            "trainable": cut_all(all_params["trainable"]),
        }

        def phi_fn(x_batch):
            u, *_ = FBPINN_model(all_params_cut, x_batch, takes, model_fns, verbose=False)
            return u

        tangent_x = jnp.zeros_like(grid_pts_jnp).at[:, 0].set(1.0)
        tangent_z = jnp.zeros_like(grid_pts_jnp).at[:, 1].set(1.0)
        _, dphi_dx = jax.jvp(phi_fn, (grid_pts_jnp,), (tangent_x,))
        _, dphi_dz = jax.jvp(phi_fn, (grid_pts_jnp,), (tangent_z,))
        return np.array(dphi_dx), np.array(dphi_dz)

    def _plot_displacement_comparison(self, i, all_params, model_fns, decomposition):
        """Compare FBPINN vs SPECFEM |U| at 3 time snapshots.

        Row 1: FBPINN predicted |U|
        Row 2: SPECFEM ground truth |U|
        Row 3: Difference (SPECFEM - FBPINN)
        Columns: t'=0, t'=0.015, t'=0.15
        """
        p = all_params["static"]["problem"]
        xx_eval = np.array(p["xx_eval"]) if p["xx_eval"] is not None else None
        zz_eval = np.array(p["zz_eval"]) if p["zz_eval"] is not None else None

        if xx_eval is None:
            n_plot = 40
            xx_eval, zz_eval = np.meshgrid(
                np.linspace(0, ax / Lx, n_plot),
                np.linspace(0, az / Lz, n_plot))

        n_pts = xx_eval.size

        # SPECFEM ground truth at 3 times
        U_spec_list = []
        for s_ux, s_uz in [(p["U_ini1x"], p["U_ini1z"]),
                           (p["U_ini2x"], p["U_ini2z"]),
                           (p["U_specx"], p["U_specz"])]:
            ux = np.array(s_ux)
            uz = np.array(s_uz)
            U_spec_list.append(np.sqrt(ux**2 + uz**2).reshape(xx_eval.shape))

        times = [0.0, t02 - t01, t_la - t01]
        time_labels = [str(0), str(round(t02 - t01, 4)), str(round(t_la - t01, 4))]

        fig, axes = plt.subplots(3, 3, figsize=(20, 9))

        for col, (t_val, t_label, U_spec) in enumerate(zip(times, time_labels, U_spec_list)):
            # FBPINN prediction
            grid_pts = np.column_stack([xx_eval.ravel(), zz_eval.ravel(),
                                         t_val * np.ones(n_pts)])
            dphi_dx, dphi_dz = self._compute_fbpinn_displacement(
                jnp.array(grid_pts), all_params, model_fns, decomposition)
            U_pinn = np.sqrt(dphi_dx**2 + dphi_dz**2).reshape(xx_eval.shape)

            # Fixed color levels from SPECFEM ground truth range
            levels = np.linspace(U_spec.min(), U_spec.max(), 101)

            # Row 1: FBPINN predicted (same levels as SPECFEM)
            im0 = axes[0, col].contourf(xx_eval * Lx, zz_eval * Lz, U_pinn, levels,
                                          cmap='jet', extend='both')
            axes[0, col].set_title(r'FBPINN $|U|$ $t=$' + t_label)
            axes[0, col].set_aspect('equal')
            fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

            # Row 2: SPECFEM ground truth
            im1 = axes[1, col].contourf(xx_eval * Lx, zz_eval * Lz, U_spec, levels,
                                          cmap='jet', extend='both')
            axes[1, col].set_title(r'SPECFEM $|U|$ $t=$' + t_label)
            axes[1, col].set_aspect('equal')
            fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

            # Row 3: Difference
            U_diff = U_spec - U_pinn
            l1_err = np.mean(np.abs(U_diff))
            im2 = axes[2, col].contourf(xx_eval * Lx, zz_eval * Lz, U_diff, 100,
                                          cmap='RdBu_r')
            axes[2, col].set_title(f'Diff (L1={l1_err:.4f})')
            axes[2, col].set_aspect('equal')
            fig.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

            for row in range(3):
                axes[row, col].set_xlabel('x (km)')
                axes[row, col].set_ylabel('z (km)')

        fig.suptitle(f'[{i}] Wavefield Comparison', fontsize=14)
        fig.tight_layout()
        return fig

    def _plot_seismogram_comparison(self, i, all_params, model_fns, decomposition):
        """Compare observed vs predicted seismograms at a few receivers.

        Computes FBPINN displacement (dphi/dx', dphi/dz') at seismometer
        locations using jax.jvp and overlays on SPECFEM input data.
        """
        p = all_params["static"]["problem"]
        X_S = np.array(p["X_S"])
        Sz = np.array(p["Sz"])
        Sx = np.array(p["Sx"])

        # Compute takes for seismometer coordinates
        m = all_params["static"]["decomposition"]["m"]
        active_all = jnp.ones(m, dtype=int)
        X_S_jnp = jnp.array(X_S)
        takes, _, (_, _, _, cut_all, _) = get_inputs(
            X_S_jnp, active_all, all_params, decomposition)
        all_params_cut = {
            "static": cut_all(all_params["static"]),
            "trainable": cut_all(all_params["trainable"]),
        }

        # Compute displacements dphi/dx' and dphi/dz' via jvp
        def phi_fn(x_batch):
            u, *_ = FBPINN_model(all_params_cut, x_batch, takes, model_fns, verbose=False)
            return u

        # dphi/dx' (tangent vector selects x-direction)
        tangent_x = jnp.zeros_like(X_S_jnp).at[:, 0].set(1.0)
        phi, dphi_dx = jax.jvp(phi_fn, (X_S_jnp,), (tangent_x,))

        # dphi/dz' (tangent vector selects z-direction)
        tangent_z = jnp.zeros_like(X_S_jnp).at[:, 1].set(1.0)
        _, dphi_dz = jax.jvp(phi_fn, (X_S_jnp,), (tangent_z,))

        dphi_dx = np.array(dphi_dx)
        dphi_dz = np.array(dphi_dz)

        # Plot ALL 20 receivers: 2 subplots (X-component, Z-component)
        # Each receiver offset vertically for clarity
        n_total = X_S.shape[0]
        n_per_seis = n_total // n_seis

        fig, axes = plt.subplots(1, 2, figsize=(14, 12))

        # Compute offset for stacking seismograms (tight spacing, overlap OK)
        max_amp = max(np.max(np.abs(Sx)), np.max(np.abs(Sz)),
                      np.max(np.abs(dphi_dx)), np.max(np.abs(dphi_dz)))
        offset_scale = max_amp * 0.5 if max_amp > 0 else 1.0

        for ir in range(n_seis):
            idx_start = ir * n_per_seis
            idx_end = (ir + 1) * n_per_seis
            t_seis = X_S[idx_start:idx_end, 2]
            # R0 on top: reverse so ir=0 gets highest offset
            offset = (n_seis - 1 - ir) * offset_scale

            # Z component first (left subplot)
            axes[0].plot(t_seis, Sz[idx_start:idx_end, 0] + offset,
                          'k-', linewidth=0.8, label='SPECFEM' if ir == 0 else None)
            axes[0].plot(t_seis, dphi_dz[idx_start:idx_end, 0] + offset,
                          'r-', linewidth=0.8, alpha=0.7, label='FBPINN' if ir == 0 else None)
            axes[0].text(t_seis[-1] + 0.005, offset, f'R{ir}', fontsize=7, va='center')

            # X component second (right subplot)
            axes[1].plot(t_seis, Sx[idx_start:idx_end, 0] + offset,
                          'k-', linewidth=0.8, label='SPECFEM' if ir == 0 else None)
            axes[1].plot(t_seis, dphi_dx[idx_start:idx_end, 0] + offset,
                          'r-', linewidth=0.8, alpha=0.7, label='FBPINN' if ir == 0 else None)
            axes[1].text(t_seis[-1] + 0.005, offset, f'R{ir}', fontsize=7, va='center')

        axes[0].set_title('Z-component (dphi/dz\')')
        axes[1].set_title('X-component (dphi/dx\')')
        for ax_ in axes:
            ax_.set_xlabel('Time (s)')
            ax_.legend(fontsize=8, loc='upper left')
            ax_.grid(True, alpha=0.2)

        fig.suptitle(f'[{i}] Seismograms: all {n_seis} receivers', fontsize=12)
        fig.tight_layout()
        return fig


# ============================================================================
# MAIN
# ============================================================================

def main(resume_total_steps=0):

    cache_dir = os.path.join(os.path.dirname(os.path.dirname(SCRIPTS_DIR)), 'results', 'rasht', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'specfem_data.npz')

    if resume_total_steps > 0:
        print("=" * 60)
        print(f"Resuming training, target total steps: {resume_total_steps}")
        print("=" * 60)
        data = dict(np.load(cache_file, allow_pickle=True))
        print(f"Loaded cached data from {cache_file}")
        n_steps = resume_total_steps
    else:
        # Load SPECFEM data
        print("=" * 60)
        print("Step 1: Loading SPECFEM data from rasht-behesht_etal_2021/event1/")
        print("=" * 60)

        data = load_specfem_data()

        # Save preprocessed data for fast re-loading
        np.savez(cache_file, **data)
        print(f"Saved preprocessed data to {cache_file}")
        n_steps = TRAINING_STEPS

    # Print data summary
    n_ic = data["X_init1"].shape[0]
    n_seismo = data["X_S"].shape[0]
    n_bc = data["X_BC_t"].shape[0]
    l_sub = data["l_sub"]
    print(f"\nData summary:")
    print(f"  IC points:        {n_ic} (40x40 grid)")
    print(f"  Seismometer data: {n_seismo} ({n_seis} receivers x {l_sub} time samples)")
    print(f"  BC points:        {n_bc}")
    print(f"  Domain (scaled):  [0, {ax/Lx:.4f}] x [0, {az/Lz:.4f}] x [0, {t_m-t_st:.1f}]")
    print(f"  Domain (phys):    [0, {ax:.3f}] x [0, {az:.3f}] km, t=[{t_st}, {t_m}] s")

    # Configure Constants
    print("\n" + "=" * 60)
    print("Step 2: Configuring FBPINNs training")
    print("=" * 60)

    # Domain bounds in scaled coordinates
    xmin = np.array([0.0, 0.0, 0.0])
    xmax = np.array([ax / Lx, az / Lz, t_m - t_st])

    c = Constants(
        run=RUN_NAME,

        domain=RectangularDomainND,
        domain_init_kwargs=dict(
            xmin=xmin,
            xmax=xmax,
        ),

        problem=AcousticFWIScalarPotential,
        problem_init_kwargs=dict(
            Lx=Lx, Lz=Lz,
            vel_background=VEL_BACKGROUND,
            vel_amplitude=VEL_AMPLITUDE,
            vel_layer_sizes=VEL_LAYER_SIZES,
            vel_box=VEL_BOX,
            mask_steepness=MASK_STEEPNESS,
            X_init1=data["X_init1"],
            U_ini1x=data["U_ini1x"],
            U_ini1z=data["U_ini1z"],
            X_init2=data["X_init2"],
            U_ini2x=data["U_ini2x"],
            U_ini2z=data["U_ini2z"],
            X_S=data["X_S"],
            Sx=data["Sx"],
            Sz=data["Sz"],
            X_BC_t=data["X_BC_t"],
            U_specx=data["U_specx"],
            U_specz=data["U_specz"],
            xx_eval=data["xx"],
            zz_eval=data["zz"],
            w_pde=W_PDE,
            w_ic1=W_IC1,
            w_ic2=W_IC2,
            w_seismo=W_SEISMO,
            w_bc=W_BC,
        ),

        decomposition=RectangularDecompositionND,
        decomposition_init_kwargs=dict(
            subdomain_xs=[
                np.linspace(0, ax / Lx, N_SUBDOMAINS_X),
                np.linspace(0, az / Lz, N_SUBDOMAINS_Z),
                np.linspace(0, t_m - t_st, N_SUBDOMAINS_T),
            ],
            subdomain_ws=[
                OVERLAP_FRACTION * np.ones(N_SUBDOMAINS_X),
                OVERLAP_FRACTION * np.ones(N_SUBDOMAINS_Z),
                OVERLAP_FRACTION * np.ones(N_SUBDOMAINS_T),
            ],
            unnorm=(0., 1.),
        ),

        network=FCN,
        network_init_kwargs=dict(
            layer_sizes=SUBDOMAIN_LAYER_SIZES,
        ),

        # 5 constraints: PDE, IC1, IC2, Seismo, BC
        ns=(PDE_BATCH_SHAPE, (n_ic,), (n_ic,), (n_seismo,), (n_bc,)),
        n_test=N_TEST,
        sampler="sobol",

        n_steps=n_steps,
        optimiser_kwargs=dict(learning_rate=LEARNING_RATE),

        summary_freq=1000, #print to console + save loss metrics
        test_freq=5000, #L1 velocity + plots (velocity, displacement, seismogram)
        model_save_freq=10000, #save model checkpoint (.jax: 10000) and + metrics.npz (1000) + physical data (.npz) (5000) from summary_freq and test_freq

        show_figures=SHOW_PLOTS,
        save_figures=SAVE_PLOTS,
    )

    print(f"\nVelocity network: {VEL_LAYER_SIZES}")
    print(f"Velocity: alpha = {VEL_BACKGROUND} + {VEL_AMPLITUDE}*tanh(NN)*mask")
    print(f"Inversion box (scaled): x'=[{VEL_BOX[0]:.4f}, {VEL_BOX[1]:.4f}], "
          f"z'=[{VEL_BOX[2]:.4f}, {VEL_BOX[3]:.4f}]")
    print(f"Mask steepness: {MASK_STEEPNESS}")
    print(f"Loss weights: PDE={W_PDE}, IC1={W_IC1}, IC2={W_IC2}, "
          f"Seismo={W_SEISMO}, BC={W_BC}")
    print(f"Subdomains: {N_SUBDOMAINS_X}x{N_SUBDOMAINS_Z}x{N_SUBDOMAINS_T} = "
          f"{N_SUBDOMAINS_X * N_SUBDOMAINS_Z * N_SUBDOMAINS_T}")
    print(f"Per-subdomain network: {SUBDOMAIN_LAYER_SIZES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training steps: {n_steps}")

    # Train
    print("\n" + "=" * 60)
    print("Step 3: Training")
    print("=" * 60)

    trainer = FBPINNTrainerFWI16(c, resume_step=n_steps if resume_total_steps > 0 else 0)
    all_params = trainer.train()

    return all_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rasht-Behesht FWI on FBPINNs")
    parser.add_argument('--resume', type=int, default=0, metavar='TOTAL_STEPS',
                        help='Resume from last checkpoint, train to TOTAL_STEPS')
    args = parser.parse_args()

    trained_params = main(resume_total_steps=args.resume)
