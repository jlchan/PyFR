#!/bin/bash
#SBATCH -J naca-warmup
#SBATCH -o naca-warmup.o%j
#SBATCH -e naca-warmup.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 06:00:00
#SBATCH -A ASC26062
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jesse.chan@oden.utexas.edu

# Warmup: P1 at Re ~ 27k (mu = 0.000037), tend = 100.
# Produces naca-ecav-100.00.pyfrs for the production restart job.
#
# Uses gpu-a100-small (1 node, up to 48 h) rather than gpu-a100-dev (2 h max).
#
# Prerequisite:
#   ./import_partition.sh naca0021_p3_1c.msh 3

set -euo pipefail

NP=3
CASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYFR_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
PYFRM="${CASE_DIR}/naca0021_p3_1c.pyfrm"
INI="${CASE_DIR}/p1_warmup_ecav.ini"
OUTDIR="/scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-run"

cd "${CASE_DIR}"

if [[ ! -f "${PYFRM}" ]]; then
    echo "Error: mesh not found: ${PYFRM}" >&2
    echo "Run: ./import_partition.sh naca0021_p3_1c.msh ${NP}" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${PYFR_ROOT}/pyfr-ls6-env.sh"

mkdir -p "${OUTDIR}"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NNODES} node(s), ${SLURM_NTASKS} rank(s)"
echo "Mesh: ${PYFRM}"
echo "Config: ${INI}"
echo "Output: ${OUTDIR}"
date

pyfr partition list "${PYFRM}"

ibrun -n "${NP}" pyfr -p run -b cuda "${PYFRM}" "${INI}"

if [[ ! -f "${OUTDIR}/naca-ecav-100.00.pyfrs" ]]; then
    echo "Warning: expected warmup checkpoint not found:" \
         "${OUTDIR}/naca-ecav-100.00.pyfrs" >&2
fi

date
