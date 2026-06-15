#!/usr/bin/env python3
"""Write gaussian-pulse-quad.msh: periodic square [-1,1]^2, 16x16 quads."""

from pathlib import Path

XMIN, XMAX = -1.0, 1.0
YMIN, YMAX = -1.0, 1.0
NX, NY = 16, 16


def node_id(i, j):
    return 1 + j * (NX + 1) + i


def main():
    lines = [
        '$MeshFormat',
        '2.2 0 8',
        '$EndMeshFormat',
        '$PhysicalNames',
        '5',
        '1 1 "periodic_0_r"',
        '1 2 "periodic_0_l"',
        '1 3 "periodic_1_l"',
        '1 4 "periodic_1_r"',
        '2 5 "Fluid"',
        '$EndPhysicalNames',
    ]

    n_nodes = (NX + 1) * (NY + 1)
    lines += ['$Nodes', str(n_nodes)]

    for j in range(NY + 1):
        y = YMIN + (YMAX - YMIN) * j / NY
        for i in range(NX + 1):
            x = XMIN + (XMAX - XMIN) * i / NX
            lines.append(f'{node_id(i, j)} {x} {y} 0')

    lines += ['$EndNodes']

    n_line = 2 * (NX + NY)
    n_quad = NX * NY
    lines += ['$Elements', str(n_line + n_quad)]

    eid = 1

    for j in range(NY):
        lines.append(f'{eid} 1 2 1 1 {node_id(0, j)} {node_id(0, j + 1)}')
        eid += 1
    for j in range(NY):
        lines.append(f'{eid} 1 2 2 2 {node_id(NX, j)} {node_id(NX, j + 1)}')
        eid += 1
    for i in range(NX):
        lines.append(f'{eid} 1 2 3 3 {node_id(i, 0)} {node_id(i + 1, 0)}')
        eid += 1
    for i in range(NX):
        lines.append(f'{eid} 1 2 4 4 {node_id(i, NY)} {node_id(i + 1, NY)}')
        eid += 1

    for j in range(NY):
        for i in range(NX):
            n1 = node_id(i, j)
            n2 = node_id(i + 1, j)
            n3 = node_id(i + 1, j + 1)
            n4 = node_id(i, j + 1)
            lines.append(f'{eid} 3 2 5 1 {n1} {n2} {n3} {n4}')
            eid += 1

    lines += ['$EndElements']

    out = Path(__file__).with_name('gaussian-pulse-quad.msh')
    out.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {out} ({NX}x{NY} quads, {n_nodes} nodes)')


if __name__ == '__main__':
    main()
