# 2D Kelvin–Helmholtz instability

Periodic KH test on `[-0.5, 0.5]²`, adapted from
[raj-brown/PyFR `2d-KHZ`](https://github.com/raj-brown/PyFR/tree/develop/PyFR-Test-Cases/2d-KHZ)
for **entropy-conservative artificial viscosity** (`shock-capturing = ec-artificial-viscosity`).

## Files

| File | Description |
|------|-------------|
| `kh.ini` | Navier–Stokes + ECAV config (GLL points, small `mu`) |
| `kh.msh` / `kh.pyfrm` | Fine mesh: **128×128** quads |
| `kh-coarse.msh` / `kh-coarse.pyfrm` | Coarse mesh: **64×64** quads |
| `gen_kh_quad_msh.py` | Structured mesh generator |
| `conv_to_vtu.sh` | Batch-export `kh-run/*.pyfrs` to VTK |
| `kh-run/` | Solution output directory (create before running) |

## ECAV notes

The upstream case uses `system = euler` and `shock-capturing = entropy-filter`.
This copy uses:

- `system = navier-stokes` — ECAV is Navier–Stokes only
- `shock-capturing = ec-artificial-viscosity`
- `soln-pts` / `flux-pts = gauss-legendre-lobatto` — required for ECAV
- No flux anti-aliasing (`quad-deg` / `quad-pts` removed; incompatible with ECAV)
- `mu = 1e-4`, `Pr = 0.72` — small viscosity (near-inviscid KH)

Entropy filter and ECAV cannot be combined in a single config today.

## Run

```bash
mkdir -p kh-run

# Coarse mesh (recommended for development)
pyfr -p run -b openmp kh-coarse.pyfrm kh.ini

# Fine mesh
pyfr import kh.msh kh.pyfrm   # if kh.pyfrm not present
pyfr -p run -b openmp kh.pyfrm kh.ini
```

## Postprocess

`pyfr export` defaults to **single** precision VTK output. This case runs in
**double** (`[backend] precision = double` in `kh.ini`), so pass
`-p double` / `--precision double` when exporting to avoid rounding the
solution in VTK.

Single snapshot (match the `.pyfrm` used for the run):

```bash
# Coarse mesh
pyfr export volume kh-coarse.pyfrm kh-run/kh-0.50.pyfrs kh-0.50.vtu \
  --precision double

# Fine mesh
pyfr export volume kh.pyfrm kh-run/kh-0.50.pyfrs kh-0.50.vtu \
  --precision double
```

Batch-export all outputs in `kh-run/` (set `MESH` to the mesh you ran with):

```bash
MESH=kh-coarse.pyfrm   # or kh.pyfrm
for f in kh-run/kh-*.pyfrs; do
  out="${f%.pyfrs}.vtu"
  pyfr export volume "$MESH" "$f" "$out" --precision double
done
```

Or run `./conv_to_vtu.sh` (set `MESH` inside the script if you used the coarse
mesh).

## Regenerate meshes

Meshes are **not** committed (see `PyFR-Test-Cases/.gitignore`). Generate before import:

```bash
python gen_kh_quad_msh.py                      # 64×64  -> kh-coarse.msh
python gen_kh_quad_msh.py 128 128 kh.msh       # 128×128 -> kh.msh
pyfr import kh-coarse.msh kh-coarse.pyfrm
pyfr import kh.msh kh.pyfrm
```

Arguments to `gen_kh_quad_msh.py`: `nx [ny] [output.msh]` (`ny` defaults to `nx`).

Periodic boundaries: `periodic_x_l`, `periodic_x_r`, `periodic_y_l`, `periodic_y_r`.
