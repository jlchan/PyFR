# 3D NACA 0021 airfoil in deep stall (ECAV)

Three-dimensional implicit LES of a **NACA 0021** airfoil in deep stall, adapted
from the PyFR benchmark of Park, Witherden, and Vincent (2017)
([DOI 10.2514/1.J055304](https://doi.org/10.2514/1.J055304)). This copy uses
**entropy-conservative artificial viscosity** (ECAV) instead of flux anti-aliasing
or an entropy filter.

Operating conditions (nondimensional): **Re<sub>c</sub> ≈ 270,000**, **M = 0.1**,
chord **c = 1**, span **1c** (periodic in *z*).

## Files

| File | Description |
|------|-------------|
| `naca0021_p3_1c.msh` | Gmsh mesh (P3 curved hexes, 1 chord span) |
| `naca0021_p3_1c.pyfrm` | PyFR mesh (generated; not committed) |
| `p1_warmup_ecav.ini` | Warmup: P1, Re ≈ 27k, `tend = 100` |
| `p3_aa_ecav.ini` | Production: P3, full Re, `tend = 400` (with warmup) |
| `p3_aa_ecav_t200.ini` | Direct run: P3, full Re, `tend = 200` (no warmup) |
| `p3_aa_entropy_filter.ini` | Alternate config (entropy filter + anti-alias) |
| `import_partition.sh` | Import `.msh` and add a mesh partition |
| `run_naca_warmup.slurm` | SLURM: warmup on 3 GPUs |
| `run_naca_ecav_24gpu.slurm` | SLURM: production start (restart from warmup) |
| `run_naca_ecav_24gpu_restart.slurm` | SLURM: continue production from latest checkpoint |
| `run_naca_ecav_24gpu_t200.slurm` | SLURM: direct run to `t = 200` on 24 GPUs |

Solution output (warmup workflow): `/scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-run`

Solution output (direct t = 200 run): `/scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-t200-run`

## ECAV notes

- `system = navier-stokes` — ECAV requires Navier–Stokes
- `[solver-ec-artificial-viscosity] enabled = true`
- `soln-pts` / `flux-pts = gauss-legendre-lobatto` — required for ECAV
- No flux anti-aliasing (incompatible with ECAV)

### Reynolds number

| Phase | `mu` | Re<sub>c</sub> |
|-------|------|----------------|
| Warmup | `0.000037` | ≈ 27,000 |
| Production | `0.0000037` | ≈ 270,000 |

With `rhoc = 1`, `uc = 1`, and chord `c = 1`: Re = `rhoc × uc × c / mu`.

## Environment (TACC Lonestar6)

Load PyFR before any `pyfr` command (login node or batch job):

```bash
source /work/11547/jlchan/ls6/PyFR/pyfr-ls6-env.sh
```

## Stage 0 — Mesh import and partitioning

Run once on a login node. **Import only once** (the mesh is large); add both
partitions needed for warmup (3 ranks) and production (24 ranks):

```bash
cd PyFR-Test-Cases/3d-naca-0021
source ../../pyfr-ls6-env.sh
mkdir -p /scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-run

pyfr import naca0021_p3_1c.msh naca0021_p3_1c.pyfrm
pyfr partition add naca0021_p3_1c.pyfrm 3
pyfr partition add naca0021_p3_1c.pyfrm 24
pyfr partition list naca0021_p3_1c.pyfrm
```

Alternatively, `./import_partition.sh naca0021_p3_1c.msh 3` (and again with `24`)
re-imports the mesh each time and is much slower.

PyFR selects the partition whose part count matches the number of MPI ranks.

## Direct run without warmup (t = 0 → 200)

Skip the P1 warmup and run **P3 at full Re** from uniform freestream ICs in a
single 24-GPU job. Useful for shorter turnaround or when testing stability at
full Reynolds number without the two-stage Park et al. procedure.

| Setting | Value |
|---------|-------|
| Config | `p3_aa_ecav_t200.ini` |
| Order | 3 |
| Re<sub>c</sub> | ≈ 270,000 (`mu = 0.0000037`) |
| `tend` | 200 |
| GPUs | 24 (8 nodes × 3 A100) |
| Queue | `gpu-a100` |
| Wall time | 48 h (estimate ~33 h for t = 0 → 200) |

**Prerequisite** — 24-partition mesh only (no 3-partition warmup mesh needed):

```bash
cd PyFR-Test-Cases/3d-naca-0021
source ../../pyfr-ls6-env.sh
mkdir -p /scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-t200-run

pyfr import naca0021_p3_1c.msh naca0021_p3_1c.pyfrm
pyfr partition add naca0021_p3_1c.pyfrm 24
```

**Submit:**

```bash
sbatch run_naca_ecav_24gpu_t200.slurm
```

This runs:

```bash
ibrun -n 24 pyfr -p run -b cuda naca0021_p3_1c.pyfrm p3_aa_ecav_t200.ini
```

Output is written to a **separate scratch directory** (`naca-0021-ecav-t200-run`)
with basename `naca-ecav-t200-{t:.2f}.pyfrs` so it does not collide with the
warmup workflow. Checkpoints every 20 time units and a wall-clock dump at 47.5 h
are configured the same as `p3_aa_ecav.ini`.

If the job stops before `t = 200`, restart manually from the latest checkpoint:

```bash
ibrun -n 24 pyfr -p restart -b cuda naca0021_p3_1c.pyfrm \
  /scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-t200-run/checkpoint-XXX.pyfrs \
  p3_aa_ecav_t200.ini
```

**Note:** Starting cold at P3 / full Re may be less stable than the warmup
workflow; monitor the job log and NaN-check output early in the run.

## Warmup workflow (t = 0 → 400)

The following stages implement the two-step Park et al. procedure: P1 warmup at
lower Re, then P3 production restart to `tend = 400`.

## Stage 1 — Warmup (t = 0 → 100)

Low-order, higher viscosity following Park et al.: stabilise the impulsive start
before raising order and lowering viscosity.

| Setting | Value |
|---------|-------|
| Config | `p1_warmup_ecav.ini` |
| Order | 1 |
| GPUs | 3 (1 node) |
| Queue | `gpu-a100-small` |
| Wall time | 6 h (estimate ~4.5 h) |

**Batch:**

```bash
sbatch run_naca_warmup.slurm
```

**Interactive (short tests only; `gpu-a100-dev` limited to 2 h):**

```bash
idev -p gpu-a100-dev -N 1 -n 3 -t 2:00:00 -A ASC26062
source /work/11547/jlchan/ls6/PyFR/pyfr-ls6-env.sh
cd /work/11547/jlchan/ls6/PyFR/PyFR-Test-Cases/3d-naca-0021
ibrun -n 3 pyfr -p run -b cuda naca0021_p3_1c.pyfrm p1_warmup_ecav.ini
```

**Expected checkpoint:** `.../naca-ecav-100.00.pyfrs`

## Stage 2 — Production start (t = 100 → 400)

Restart from the warmup solution at full Reynolds number and P3.

| Setting | Value |
|---------|-------|
| Config | `p3_aa_ecav.ini` |
| Order | 3 |
| GPUs | 24 (8 nodes × 3 A100) |
| Queue | `gpu-a100` |
| Wall time | 48 h (max); ~50 h total estimated for t = 100 → 400 |

```bash
sbatch run_naca_ecav_24gpu.slurm
```

This runs:

```bash
ibrun -n 24 pyfr -p restart -b cuda naca0021_p3_1c.pyfrm \
  /scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-run/naca-ecav-100.00.pyfrs \
  p3_aa_ecav.ini
```

Override the checkpoint if needed:

```bash
CHECKPOINT=/path/to/naca-ecav-100.00.pyfrs sbatch run_naca_ecav_24gpu.slurm
```

### Checkpoints during production

`p3_aa_ecav.ini` writes:

- **Field output** every 50 time units: `naca-ecav-{t:.2f}.pyfrs`
- **Restart checkpoints** every 20 time units: `checkpoint-{t:.4f}.pyfrs`
- **Wall-clock dump** at 47.5 h: gated `checkpoint-{t:.4f}.pyfrs` before queue limit

## Stage 3 — Production restart (if wall limit reached)

If the first production job stops before `t = 400`, submit a continuation job.
It automatically picks the latest `naca-ecav-*.pyfrs` or `checkpoint-*.pyfrs`
in the output directory:

```bash
sbatch run_naca_ecav_24gpu_restart.slurm
```

Chain jobs so the restart runs only after the first job succeeds:

```bash
JOB1=$(sbatch run_naca_ecav_24gpu.slurm | awk '{print $NF}')
sbatch --dependency=afterok:$JOB1 run_naca_ecav_24gpu_restart.slurm
```

Override the checkpoint explicitly:

```bash
CHECKPOINT=/scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-run/checkpoint-240.0000.pyfrs \
  sbatch run_naca_ecav_24gpu_restart.slurm
```

PyFR resumes `tcurr` from the `.pyfrs` file and advances to `tend = 400` in
`p3_aa_ecav.ini`.

## Monitoring

```bash
showq -u
squeue -u $USER
tail -f naca-warmup.o<JOBID>          # warmup
tail -f naca-ecav-24gpu.o<JOBID>      # production (warmup path)
tail -f naca-ecav-t200.o<JOBID>       # direct t = 200 run
```

## Postprocess

`pyfr export` defaults to single-precision VTK. This case uses double precision;
pass `--precision double` when exporting:

```bash
pyfr export volume naca0021_p3_1c.pyfrm \
  /scratch/11547/jlchan/pyfr-runs/naca-0021-ecav-run/naca-ecav-100.00.pyfrs \
  naca-ecav-100.00.vtu --precision double
```

## Workflow summary (warmup path)

```
Stage 0   pyfr import + partition (3 and 24 parts)
   │
Stage 1   sbatch run_naca_warmup.slurm          →  naca-ecav-100.00.pyfrs
   │
Stage 2   sbatch run_naca_ecav_24gpu.slurm     →  t = 100 … 400 (≤ 48 h)
   │
Stage 3   sbatch run_naca_ecav_24gpu_restart.slurm  (repeat if needed)
```

## Direct path (no warmup): Stage 0 (24-partition only) →
`sbatch run_naca_ecav_24gpu_t200.slurm` → t = 0 … 200.

All SLURM scripts charge allocation **ASC26062** and email **jesse.chan@oden.utexas.edu**
on BEGIN, END, and FAIL.
