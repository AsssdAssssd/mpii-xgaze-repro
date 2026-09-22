#!/usr/bin/env bash
#
# MPII leave-one-out (15 folds) train + test.
#
# Reuses the single configs/config.yaml: before every fold it rewrites the
# paths inside it (experiment.name / experiment.data_dir, and later
# test.pre_trained_model_path) instead of keeping 15 separate yaml files.
#
# The data loader always reads "<data_dir>/train_test_split.json"
# (data_loader/general_data_loader.py), so this script points that name at the
# current fold's split via a symlink.
#
set -euo pipefail

PROJECT_DIR="/home/zhongrui/code/mpii_xgaze_repro"
CONFIG="${PROJECT_DIR}/configs/config.yaml"
DATA_DIR="/home/zhongrui/datasets/mpii2mpii_loo"
PYTHON_BIN="${PYTHON_BIN:-python}"

N_FOLDS="${N_FOLDS:-15}"

cd "${PROJECT_DIR}"

# keep the user's original config around (only once)
cp -n "${CONFIG}" "${CONFIG}.bak"

for k in $(seq 0 $((N_FOLDS - 1))); do
    fold=$(printf "fold%02d" "${k}")
    name="MPII_LOO_repro_${fold}"
    split="${DATA_DIR}/train_test_split_${fold}.json"

    if [[ ! -f "${split}" ]]; then
        echo "[skip] split not found: ${split}" >&2
        continue
    fi

    # the loader always opens "<data_dir>/train_test_split.json"
    ln -sf "${split}" "${DATA_DIR}/train_test_split.json"

    # rewrite the paths in config.yaml for this fold
    sed -i -E "s#^([[:space:]]*name:).*#\1 ${name}#" "${CONFIG}"
    sed -i -E "s#^([[:space:]]*data_dir:).*#\1 \"${DATA_DIR}\"#" "${CONFIG}"

    echo "=================================================="
    echo "[$((k + 1))/${N_FOLDS}] ${name}  test=$(basename "${split}")"
    echo "=================================================="

    # ---- train ----
    "${PYTHON_BIN}" main.py --config "${CONFIG}" --mode train

    # ---- test the checkpoint this fold just produced ----
    weights_dir="${PROJECT_DIR}/exp/${name}/train/weights"
    last_ckpt=$(ls "${weights_dir}"/*_ckpt.pth.tar 2>/dev/null | sort -V | tail -n1 || true)
    if [[ -z "${last_ckpt}" ]]; then
        echo "[warn] no checkpoint in ${weights_dir}, skip test" >&2
        continue
    fi

    # test.pre_trained_model_path is normally a {root}/{epochs} template;
    # point it at the concrete checkpoint we just saved
    sed -i -E "s#^([[:space:]]*pre_trained_model_path:).*#\1 \"${last_ckpt}\"#" "${CONFIG}"

    "${PYTHON_BIN}" main.py --config "${CONFIG}" --mode test
done

echo "all ${N_FOLDS} folds done."
