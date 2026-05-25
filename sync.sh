#!/usr/bin/env bash
set -euo pipefail


# SYNC MIO
REMOTE=wendian
REMOTE_DIR="~/Workspace/quFWI"
LOCAL_DIR="$HOME/Workspace/1.2-quFWI"


# du -ah --max-depth=2 ~ | sort -rh | head -n 20

# # --- full tree (uncomment to sync everything except .venv) -------------------
rsync -avz --progress --human-readable \
    --exclude='.venv/' \
    --exclude='results/' \
    --exclude='.git/' \
    "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"



# --- pull outputs back (uncomment when needed) -------------------------------
# rsync -avz --progress --human-readable \
#     "$REMOTE:$REMOTE_DIR/examples/fwi/outputs/marmousi_variants/marmousi_sparse400_vxvz/" \
#     "$LOCAL_DIR/examples/fwi/outputs/marmousi_variants/marmousi_sparse400_vxvz/"
# rsync -avz --progress --human-readable \
#     "$REMOTE:$REMOTE_DIR/examples/fwi/outputs/joint/marmousi_joint_sparse400/" \
#     "$LOCAL_DIR/examples/fwi/outputs/joint/marmousi_joint_sparse400/"
