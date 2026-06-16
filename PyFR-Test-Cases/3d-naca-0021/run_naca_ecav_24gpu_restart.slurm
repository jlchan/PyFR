#!/bin/bash
#SBATCH -J naca-ecav-restart
#SBATCH -o naca-ecav-restart.o%j
#SBATCH -e naca-ecav-restart.e%j
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH -n 24
#SBATCH -t 48:00:00
#SBATCH -A ASC26062
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jesse.chan@oden.utexas.edu

# Continue production run (24 GPU) from the latest checkpoint in OUTDIR.
# Keeps tend = 400 in p3_aa_ecav.ini; PyFR resumes tcurr from the .pyfrs file.
#
# Prerequisites:
#   ./import_partition.sh naca0021_p3_1c.msh 24
#   A prior production job (run_naca_ecav_24gpu.slurm or this script)
#
# Override checkpoint:
#   CHECKPOINT=/path/to/checkpoint-240.0000.pyfrs sbatch run_naca_ecav_24gpu_restart.slurm
#
# Chain jobs automatically:
#   JOB1=$(sbatch run_naca_ecav_24gpu.slurm | awk '{print $NF}')
#   sbatch --dependency=afterok:$JOB1 run_naca_ecav_24gpu_restart.slurm

set -euo pipefail

NP=24
CASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYFR_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
PYFRM="${CASE_DIR}/naca0021_p3_1c.pyfrm"
INI="${CASE_DIR}/p3_aa_ecav.ini"
OUTDIR="/scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-run"

find_latest_checkpoint() {
    local dir="$1"
    local latest="" max_t=""

    shopt -s nullglob
    for f in "${dir}"/naca-ecav-*.pyfrs "${dir}"/checkpoint-*.pyfrs; do
        local base t
        base="$(basename "$f")"
        if [[ "$base" =~ ^naca-ecav-([0-9.]+)\.pyfrs$ ]]; then
            t="${BASH_REMATCH[1]}"
        elif [[ "$base" =~ ^checkpoint-([0-9.]+)\.pyfrs$ ]]; then
            t="${BASH_REMATCH[1]}"
        else
            continue
        fi

        if [[ -z "$max_t" ]] || awk -v a="$t" -v b="$max_t" 'BEGIN{exit !(a>b)}'; then
            max_t="$t"
            latest="$f"
        fi
    done
    shopt -u nullglob

    if [[ -z "$latest" ]]; then
        return 1
    fi

    printf '%s\n' "$latest"
}

cd "${CASE_DIR}"

if [[ ! -f "${PYFRM}" ]]; then
    echo "Error: mesh not found: ${PYFRM}" >&2
    echo "Run: ./import_partition.sh naca0021_p3_1c.msh ${NP}" >&2
    exit 1
fi

if [[ -n "${CHECKPOINT:-}" ]]; then
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Error: checkpoint not found: ${CHECKPOINT}" >&2
        exit 1
    fi
else
    CHECKPOINT="$(find_latest_checkpoint "${OUTDIR}")" || {
        echo "Error: no checkpoints found in ${OUTDIR}" >&2
        exit 1
    }
fi

# shellcheck source=/dev/null
source "${PYFR_ROOT}/pyfr-ls6-env.sh"

mkdir -p "${OUTDIR}"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NNODES} node(s), ${SLURM_NTASKS} rank(s)"
echo "Host: $(hostname)"
echo "Mesh: ${PYFRM}"
echo "Config: ${INI}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTDIR}"
date

pyfr partition list "${PYFRM}"

ibrun -n "${NP}" pyfr -p restart -b cuda "${PYFRM}" "${CHECKPOINT}" "${INI}"

date
