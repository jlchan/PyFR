#!/bin/bash
# Import a Gmsh .msh file and add a PyFR mesh partition.
#
# Usage:
#   ./import_partition.sh mesh.msh [nparts [part_name]]
#
# Examples:
#   ./import_partition.sh naca0021_p3_1c.msh
#   ./import_partition.sh naca0021_p3_1c.msh 3
#   ./import_partition.sh naca0021_p3_1c.msh 3 gpu3

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") mesh.msh [nparts [part_name]]

Convert mesh.msh to mesh.pyfrm with pyfr import, then add a partition.

  nparts     Number of partitions (default: 1)
  part_name  Partition name stored in the .pyfrm (default: nparts)
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit $(( $# < 1 ? 1 : 0 ))
fi

msh="$1"
nparts="${2:-1}"
part_name="${3:-$nparts}"

if [[ ! -f "$msh" ]]; then
    echo "Error: mesh file not found: $msh" >&2
    exit 1
fi

case "$msh" in
    *.msh) ;;
    *)
        echo "Error: expected a .msh file, got: $msh" >&2
        exit 1
        ;;
esac

if ! [[ "$nparts" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: nparts must be a positive integer, got: $nparts" >&2
    exit 1
fi

pyfrm="${msh%.msh}.pyfrm"

echo "Importing $msh -> $pyfrm"
pyfr import "$msh" "$pyfrm"

echo "Adding partition $part_name ($nparts parts) to $pyfrm"
pyfr partition add "$pyfrm" "$nparts" "$part_name"

echo "Done. Partitionings:"
pyfr partition list "$pyfrm"
