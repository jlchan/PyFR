#!/bin/bash
#SBATCH -J naca-ecav-24gpu
#SBATCH -o naca-ecav-24gpu.o%j
#SBATCH -e naca-ecav-24gpu.e%j
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH -n 24
#SBATCH -t 48:00:00
#SBATCH -A ASC26062
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jesse.chan@oden.utexas.edu

# Production start (24 GPU): restart from warmup at t = 100, run to tend = 400.
#
# Prerequisites:
#   ./import_partition.sh naca0021_p3_1c.msh 24
#   sbatch run_naca_warmup.slurm   (or interactive warmup; see below)
#
# Override checkpoint:
#   CHECKPOINT=/path/to/naca-ecav-100.00.pyfrs sbatch run_naca_ecav_24gpu.slurm
#
# If the job hits the 48 h wall limit before t = 400, submit:
#   sbatch run_naca_ecav_24gpu_restart.slurm

set -euo pipefail

NP=24
CASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYFR_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
PYFRM="${CASE_DIR}/naca0021_p3_1c.pyfrm"
INI="${CASE_DIR}/p3_aa_ecav.ini"
OUTDIR="/scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-run"
CHECKPOINT="${CHECKPOINT:-${OUTDIR}/naca-ecav-100.00.pyfrs}"

cd "${CASE_DIR}"

if [[ ! -f "${PYFRM}" ]]; then
    echo "Error: mesh not found: ${PYFRM}" >&2
    echo "Run: ./import_partition.sh naca0021_p3_1c.msh ${NP}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: restart checkpoint not found: ${CHECKPOINT}" >&2
    echo "Complete the warmup step first (p1_warmup_ecav.ini, tend = 100)." >&2
    exit 1
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
