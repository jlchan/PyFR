#!/usr/bin/env python3
"""Write structured periodic KH quad meshes in Gmsh 2.2 format.

Domain [-0.5, 0.5]^2 with physical groups matching kh.msh:
  periodic_x_l (2), periodic_x_r (3), periodic_y_l (4), periodic_y_r (5), fluid (1).

Usage:
  python gen_kh_quad_msh.py                    # 64 x 64 -> kh-coarse.msh
  python gen_kh_quad_msh.py 128 128 kh.msh     # 128 x 128 -> kh.msh

Arguments: nx [ny] [output.msh]  (ny defaults to nx; output defaults to kh-coarse.msh)
"""

from __future__ import annotations

import sys
from pathlib import Path

XMIN, XMAX = -0.5, 0.5
YMIN, YMAX = -0.5, 0.5


def node_id(i: int, j: int, nx: int) -> int:
    return 1 + j * (nx + 1) + i


def write_mesh(nx: int, ny: int, path: Path) -> None:
    lines = [
        '$MeshFormat',
        '2.2 0 8',
        '$EndMeshFormat',
        '$PhysicalNames',
        '5',
        '1 2 "periodic_x_l"',
        '1 3 "periodic_x_r"',
        '1 4 "periodic_y_l"',
        '1 5 "periodic_y_r"',
        '2 1 "fluid"',
        '$EndPhysicalNames',
    ]

    n_nodes = (nx + 1) * (ny + 1)
    lines += ['$Nodes', str(n_nodes)]

    for j in range(ny + 1):
        y = YMIN + (YMAX - YMIN) * j / ny
        for i in range(nx + 1):
            x = XMIN + (XMAX - XMIN) * i / nx
            lines.append(f'{node_id(i, j, nx)} {x} {y} 0')

    lines += ['$EndNodes']

    n_line = 2 * (nx + ny)
    n_quad = nx * ny
    lines += ['$Elements', str(n_line + n_quad)]

    eid = 1

    # periodic_y_l (bottom, y = YMIN)
    for i in range(nx):
        lines.append(
            f'{eid} 1 2 4 1 {node_id(i, 0, nx)} {node_id(i + 1, 0, nx)}'
        )
        eid += 1

    # periodic_x_r (right, x = XMAX)
    for j in range(ny):
        lines.append(
            f'{eid} 1 2 3 2 {node_id(nx, j, nx)} {node_id(nx, j + 1, nx)}'
        )
        eid += 1

    # periodic_y_r (top, y = YMAX)
    for i in range(nx):
        lines.append(
            f'{eid} 1 2 5 3 {node_id(i + 1, ny, nx)} {node_id(i, ny, nx)}'
        )
        eid += 1

    # periodic_x_l (left, x = XMIN)
    for j in range(ny):
        lines.append(
            f'{eid} 1 2 2 4 {node_id(0, j + 1, nx)} {node_id(0, j, nx)}'
        )
        eid += 1

    for j in range(ny):
        for i in range(nx):
            n1 = node_id(i, j, nx)
            n2 = node_id(i + 1, j, nx)
            n3 = node_id(i + 1, j + 1, nx)
            n4 = node_id(i, j + 1, nx)
            lines.append(f'{eid} 3 2 1 20 {n1} {n2} {n3} {n4}')
            eid += 1

    lines += ['$EndElements']
    path.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {path} ({nx}x{ny} quads, {n_nodes} nodes, {n_line + n_quad} elements)')


def main() -> None:
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else nx
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('kh-coarse.msh')
    write_mesh(nx, ny, out)


if __name__ == '__main__':
    main()
