#!/bin/bash
#SBATCH -J naca-ecav-t200
#SBATCH -o naca-ecav-t200.o%j
#SBATCH -e naca-ecav-t200.e%j
#SBATCH -p gpu-a100
#SBATCH -N 8
#SBATCH -n 24
#SBATCH -t 48:00:00
#SBATCH -A ASC26062
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jesse.chan@oden.utexas.edu

# Direct ECAV run (24 GPU): P3, full Re, t = 0 → 200, no warmup.
#
# Prerequisites:
#   pyfr import naca0021_p3_1c.msh naca0021_p3_1c.pyfrm
#   pyfr partition add naca0021_p3_1c.pyfrm 24

set -euo pipefail

NP=24
CASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYFR_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
PYFRM="${CASE_DIR}/naca0021_p3_1c.pyfrm"
INI="${CASE_DIR}/p3_aa_ecav_t200.ini"
OUTDIR="/scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-t200-run"

cd "${CASE_DIR}"

if [[ ! -f "${PYFRM}" ]]; then
    echo "Error: mesh not found: ${PYFRM}" >&2
    echo "Run: pyfr import naca0021_p3_1c.msh naca0021_p3_1c.pyfrm" >&2
    echo "     pyfr partition add naca0021_p3_1c.pyfrm ${NP}" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${PYFR_ROOT}/pyfr-ls6-env.sh"

mkdir -p "${OUTDIR}"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NNODES} node(s), ${SLURM_NTASKS} rank(s)"
echo "Host: $(hostname)"
echo "Mesh: ${PYFRM}"
echo "Config: ${INI}"
echo "Output: ${OUTDIR}"
date

pyfr partition list "${PYFRM}"

ibrun -n "${NP}" pyfr -p run -b cuda "${PYFRM}" "${INI}"

date
