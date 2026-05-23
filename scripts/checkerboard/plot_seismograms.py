"""Plot raw SPECFEM seismograms for the checkerboard dataset."""

import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "checkerboard", "specfem", "event1", "seismograms")

seis_dir = DATA_DIR
files = sorted(os.listdir(seis_dir))
files_z = [f for f in files if f[-6] == 'Z']
files_x = [f for f in files if f[-6] == 'X']

n_seis = len(files_z)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Z-component
for i, f in enumerate(files_z):
    data = np.loadtxt(os.path.join(seis_dir, f))
    t, amp = data[:, 0], data[:, 1]
    # Offset traces for visibility
    axes[0].plot(t, amp / np.max(np.abs(amp) + 1e-30) * 0.4 + i, 'k', linewidth=0.5)

axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Receiver #")
axes[0].set_title(f"Z-component ({n_seis} receivers)")
axes[0].set_yticks(range(n_seis))
axes[0].set_yticklabels([f.replace("AA.", "").replace(".BXZ.semd", "") for f in files_z], fontsize=6)

# X-component
for i, f in enumerate(files_x):
    data = np.loadtxt(os.path.join(seis_dir, f))
    t, amp = data[:, 0], data[:, 1]
    axes[1].plot(t, amp / np.max(np.abs(amp) + 1e-30) * 0.4 + i, 'k', linewidth=0.5)

axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Receiver #")
axes[1].set_title(f"X-component ({len(files_x)} receivers)")
axes[1].set_yticks(range(len(files_x)))
axes[1].set_yticklabels([f.replace("AA.", "").replace(".BXX.semd", "") for f in files_x], fontsize=6)

plt.suptitle("SPECFEM Seismograms — Checkerboard Model", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "seismograms.png"), dpi=150)
plt.show()
print("Saved seismograms.png")
