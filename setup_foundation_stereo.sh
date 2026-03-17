#!/usr/bin/env bash
# ============================================================
# setup_foundation_stereo.sh
#
# Clones the NVlabs/FoundationStereo repository into the
# FoundationStereo/ directory and removes the nested .git so
# the code becomes part of this repository (no submodule),
# allowing it to be accessed when cloning to another machine.
#
# Usage:
#   bash setup_foundation_stereo.sh
#
# After running this script:
#   1. Install dependencies:
#        conda env create -f FoundationStereo/environment.yml
#        conda activate foundation_stereo
#        pip install flash-attn
#
#   2. Download pretrained model weights and place under
#      FoundationStereo/pretrained_models/:
#        https://github.com/NVlabs/FoundationStereo#model-weights
#      e.g. the 23-51-11/ folder should go to
#           FoundationStereo/pretrained_models/23-51-11/
#
#   3. (Optional) Commit the source to your own repo so
#      collaborators get it automatically on clone:
#        git add FoundationStereo/
#        git commit -m "Add FoundationStereo source (no nested git)"
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOUNDATION_DIR="${SCRIPT_DIR}/FoundationStereo"

# ---- Already set up? ----
if [ -f "${FOUNDATION_DIR}/core/foundation_stereo.py" ]; then
    echo "FoundationStereo is already set up at ${FOUNDATION_DIR}"
    exit 0
fi

echo "==> Cloning NVlabs/FoundationStereo (shallow clone) ..."
git clone --depth=1 https://github.com/NVlabs/FoundationStereo "${FOUNDATION_DIR}"

echo "==> Removing nested .git directory (converting to plain directory) ..."
rm -rf "${FOUNDATION_DIR}/.git"

echo ""
echo "✓  FoundationStereo source is now in ${FOUNDATION_DIR}"
echo "   The nested .git has been removed; files are plain (no submodule)."
echo ""
echo "Next steps"
echo "----------"
echo "  1. Install dependencies:"
echo "       conda env create -f FoundationStereo/environment.yml"
echo "       conda activate foundation_stereo"
echo "       pip install flash-attn"
echo ""
echo "  2. Download pretrained model weights:"
echo "       https://github.com/NVlabs/FoundationStereo#model-weights"
echo "       Place the folder (e.g. 23-51-11/) under:"
echo "       FoundationStereo/pretrained_models/23-51-11/"
echo ""
echo "  3. (Optional) commit the source so collaborators get it on clone:"
echo "       git add FoundationStereo/"
echo "       git commit -m 'Add FoundationStereo source (no nested git)'"
echo ""
echo "  4. Run FoundationStereo depth estimation:"
echo "       python depth_map_foundation.py \\"
echo "           --calib data/calibration/calib.npz \\"
echo "           --left  data/sessions/my_scene/frames/left_0000.png \\"
echo "           --right data/sessions/my_scene/frames/right_0000.png \\"
echo "           --out-dir data/sessions/my_scene/output/ \\"
echo "           --ckpt FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
echo ""
echo "  Or via the pipeline:"
echo "       python pipeline.py depth --session my_scene --use-foundation-stereo \\"
echo "           --ckpt FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
