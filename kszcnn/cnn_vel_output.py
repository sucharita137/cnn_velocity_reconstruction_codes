#!/usr/bin/env python
# coding: utf-8

# In[3]:


from cnn_vel_density_pos_mass import train_and_eval


# In[4]:


#cnn velocity reconstruction: 

import os, time, random
import numpy as np, h5py
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import RegularGridInterpolator
from numpy.fft import fftn, ifftn, fftfreq

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


# In[5]:


GRID_FILE = "Grids_Mcdm_IllustrisTNG_1P_128_z=0.0.npy"
TRAIN_HALO_FILE = "groups_090_1P_0.hdf5"
TEST_HALO_FILE  = "groups_090_1P_p2_n1.hdf5"   
TRAIN_REAL_IDX = 0
TEST_REAL_IDX = 14

PATCH = 32
BATCH = 16           
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-5
MASS_CUT = 1e11
BOXSIZE = 25.0
MAX_HALOS = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUGMENT = True
PATIENCE = 14
SMOOTH_SCALE = 2.0
CHECKPOINT_DIR = "checkpoints_no_vlin_improved_v2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# pearson penalty weight (try 0.01 - 0.2)
PEARSON_WEIGHT = 0.08

# test-time augmentation toggle (flip x/y)
TTA = True

print("Device:", DEVICE)


# In[6]:


def memmap_grid_slices(grid_file, idxs):
    if isinstance(idxs, int): idxs = [idxs]
    arr = np.load(grid_file, allow_pickle=False, mmap_mode='r')
    return [np.asarray(arr[i], dtype=np.float32) for i in idxs]


# In[7]:


def load_halos(hfile):
    with h5py.File(hfile, "r") as f:
        pos = np.array(f["Group/GroupPos"]) / 1000.0  # ckpc/h -> Mpc/h
        vel = np.array(f["Group/GroupVel"])
        mass = np.array(f["Group/Group_M_Mean200"]) * 1e10
    return pos, vel, mass


# In[8]:


def smooth_density_kspace(rho_cdm, R_smooth, boxsize=BOXSIZE):
    rho = np.asarray(rho_cdm, dtype=np.float32)
    delta = rho / rho.mean() - 1.0
    if R_smooth == 0.0:
        return delta.astype(np.float32)
    N = rho.shape[0]
    dk = fftn(delta)
    kfreq = fftfreq(N, d=boxsize/N)
    kx, ky, kz = np.meshgrid(2*np.pi*kfreq, 2*np.pi*kfreq, 2*np.pi*kfreq, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    W = np.exp(-0.5 * k2 * (R_smooth**2))
    return ifftn(dk * W).real.astype(np.float32)


# In[9]:


def compute_vlin_from_density(rho_cdm, z_snap=0.0, boxsize=BOXSIZE,
                              H0=67.66, Omega_m0=0.3, R_smooth=2.0):
    N = rho_cdm.shape[0]
    a = 1.0 / (1.0 + z_snap)
    Hz = H0 * np.sqrt(Omega_m0 * (1+z_snap)**3 + 1.0 - Omega_m0)
    f = (Omega_m0*(1+z_snap)**3 / (Omega_m0*(1+z_snap)**3 + 1.0 - Omega_m0))**0.55
    delta_x = rho_cdm / np.mean(rho_cdm) - 1.0
    dk = fftn(delta_x)
    kfreq = fftfreq(N, d=boxsize/N)
    kx, ky, kz = np.meshgrid(2*np.pi*kfreq, 2*np.pi*kfreq, 2*np.pi*kfreq, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2_nozero = np.where(k2 == 0, 1.0, k2)
    pref = 1j * a * Hz * f
    vz_k = pref * (kz / k2_nozero) * dk
    vz_k[k2 == 0] = 0.0
    if R_smooth is not None and R_smooth > 0.0:
        W = np.exp(-0.5 * k2 * (R_smooth**2))
        vz_k *= W
    vz_x = ifftn(vz_k).real
    return vz_x.astype(np.float32)


# In[10]:


def build_vlin_interpolator(vlin_grid):
    N = vlin_grid.shape[0]
    cell = BOXSIZE / N
    coords = (np.arange(N) + 0.5) * cell
    return RegularGridInterpolator((coords, coords, coords), vlin_grid,
                                   bounds_error=False, fill_value=0.0)


# In[11]:


# preparing dataset to extract 3D density patch from grids, add coordinates, augment by symmetry operation
# returns the patch together with normalized halo mass and the line-of-sight velocity.

class MultiChannelHaloDataset(Dataset):
    
    def __init__(self, grids_channels, pos, vel, mass,
                 mass_mean=None, mass_std=None,
                 mass_cut=MASS_CUT, max_n=MAX_HALOS,
                 augment=AUGMENT, rng=None):

        if rng is None:
            rng = np.random.RandomState(SEED)
        # Select halos above mass threshold
        mask = mass > mass_cut
        pos, vel, mass = pos[mask], vel[mask], mass[mask]

        # Deterministic subsampling for reproducibility
        if max_n is not None and len(pos) > max_n:
            sel = rng.choice(len(pos), max_n, replace=False)
            pos, vel, mass = pos[sel], vel[sel], mass[sel]
        # store extracted data
        self.pos  = pos
        self.vz   = vel[:, 2].astype(np.float32)
        self.mass = mass.astype(np.float32)

        self.grids   = grids_channels
        self.N       = grids_channels[0].shape[0]
        self.patch   = PATCH
        self.augment = augment

        # Train-based mass normalization
        logm = np.log10(self.mass + 1e-12)
        self.mass_mean = mass_mean if mass_mean is not None else logm.mean()
        self.mass_std  = mass_std  if mass_std  is not None else logm.std() + 1e-12

        print(f"Selected {len(self.pos)} halos (mass_cut={mass_cut}, cap={max_n}).")

    def __len__(self):
        return len(self.pos)

    def _extract_patch_single(self, grid, center):
        #Extract periodic cubic patch centered on halo
        cell = BOXSIZE / self.N
        idx  = (center / cell - 0.5).astype(int)
        r    = self.patch // 2

        xs = [(idx[0] + i) % self.N for i in range(-r, r)]
        ys = [(idx[1] + i) % self.N for i in range(-r, r)]
        zs = [(idx[2] + i) % self.N for i in range(-r, r)]

        return grid[np.ix_(xs, ys, zs)]

    def __getitem__(self, i):
        # Extract density patches
        patches = [self._extract_patch_single(g, self.pos[i]) for g in self.grids]

        # Relative coordinate channels in [-1,1]
        P = self.patch
        coords = (np.arange(P) - (P - 1)/2) / ((P - 1)/2)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

        x = np.stack(patches + [X, Y, Z], axis=0).astype(np.float32)
        x = np.clip(x, -6.0, 6.0)

        # Symmetry augmentation
        if self.augment:
            if np.random.rand() < 0.5: x = x[:, ::-1, :, :]
            if np.random.rand() < 0.5: x = x[:, :, ::-1, :]
            k = np.random.randint(4)
            if k: x = np.rot90(x, k, axes=(1, 2))

        # Scalar halo mass feature
        mass_norm = (np.log10(self.mass[i] + 1e-12) - self.mass_mean) / self.mass_std
        # cnn output
        y = float(self.vz[i])
        return np.ascontiguousarray(x, dtype=np.float32), np.float32(mass_norm), np.float32(y)


# In[12]:


def vlin_at_halos(vlin_grid, halo_pos):
    #Interpolate linear-theory vz field at halo positions
    interp = build_vlin_interpolator(vlin_grid)
    pos_wrapped = (halo_pos % BOXSIZE)
    vals = [np.asarray(interp(tuple(p))).item() for p in pos_wrapped]
    return np.array(vals, dtype=np.float32)


# In[13]:


def conv3d_gn(in_ch, out_ch, kernel=3, stride=1, padding=1, ngroups=8):
    g = min(ngroups, out_ch)
    while out_ch % g != 0 and g > 1:
        g -= 1
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                  padding=padding, bias=False),
        nn.GroupNorm(num_groups=g, num_channels=out_ch),
        nn.ReLU(inplace=True)
    )


# In[14]:


class ResidualBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)

    def forward(self, x):
        out = self.conv1(x); out = self.gn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.gn2(out)
        return self.relu(out + x)


# In[15]:


class CNN_Improved3D(nn.Module):
    def __init__(self, in_ch=4, base_ch=40, dropout=0.15, n_blocks=2):
        super().__init__()
        self.conv_in = conv3d_gn(in_ch, base_ch)
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            conv3d_gn(base_ch, base_ch * 2)
        )
        self.res1 = nn.Sequential(
            *[ResidualBlock3D(base_ch * 2) for _ in range(n_blocks)]
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            conv3d_gn(base_ch * 2, base_ch * 4)
        )
        self.res2 = nn.Sequential(
            *[ResidualBlock3D(base_ch * 4) for _ in range(n_blocks)]
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        feat = base_ch * 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat + 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, mass_scalar):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        f = self.avgpool(x).view(x.size(0), -1)

        if not torch.is_tensor(mass_scalar):
            mass_scalar = torch.tensor(mass_scalar, dtype=f.dtype, device=f.device)
        if mass_scalar.dim() == 1:
            ms = mass_scalar.view(x.size(0), 1)
        else:
            ms = mass_scalar.view(x.size(0), -1)
        ms = ms.to(f.dtype).to(f.device)

        cat = torch.cat([f, ms], dim=1)
        out = self.head(cat).view(-1)
        return out


# In[16]:


def pearson_corr_torch(x, y, eps=1e-6):
    # x, y: tensors (batch,)
    xm = torch.mean(x)
    ym = torch.mean(y)
    xm0 = x - xm
    ym0 = y - ym
    cov = torch.mean(xm0 * ym0)
    sx = torch.sqrt(torch.mean(xm0 * xm0) + eps)
    sy = torch.sqrt(torch.mean(ym0 * ym0) + eps)
    corr = cov / (sx * sy + eps)
    return corr


# In[17]:


def _compute_stats(true, pred):
    mask = np.isfinite(true) & np.isfinite(pred)
    n = int(mask.sum())
    if n == 0:
        return mask, np.nan, np.nan, np.nan, np.nan, np.nan, n
    t = true[mask]; p = pred[mask]
    try:
        corr = float(pearsonr(t, p)[0])
    except Exception:
        corr = np.nan
    bias = float(np.mean(p - t))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    rms_true = float(np.std(t))
    rms_pred = float(np.std(p))
    return mask, corr, bias, rmse, rms_true, rms_pred, n


# In[18]:


def hexbin_panel(ax, true, pred, title, cmap="viridis"):
    mask, corr, bias, rmse, rms_true, rms_pred, n = _compute_stats(true, pred)
    vmax = max(np.max(np.abs(true[mask])), np.max(np.abs(pred[mask]))) if n > 0 else 1.0
    lims = [-vmax, vmax]

    hb = ax.hexbin(
        true[mask], pred[mask],
        gridsize=150,
        cmap=cmap,
        bins='log',
        mincnt=1,
        extent=(lims[0], lims[1], lims[0], lims[1])
    )

    ax.plot(lims, lims, 'r--', lw=1.2, label="1:1")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("True LOS velocity (km/s)")
    ax.set_ylabel("Predicted LOS velocity (km/s)")
    ax.set_title(f"{title}\nρ = {corr:.3f}")

    stats_text = (
        f"N = {n}\n"
        f"ρ = {corr:.3f}\n"
        f"Bias = {bias:.2f}\n"
        f"RMSE = {rmse:.2f}\n"
        f"RMS(true) = {rms_true:.2f}\n"
        f"RMS(pred) = {rms_pred:.2f}"
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va='top',
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85)
    )
    return hb


# In[19]:


train_and_eval()


# In[ ]:




