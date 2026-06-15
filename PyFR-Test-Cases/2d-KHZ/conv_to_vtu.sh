#!/bin/bash
set -e

MESH="${MESH:-kh.pyfrm}"

for f in kh-run/kh-*.pyfrs; do
 out="${f%.pyfrs}.vtu"
 echo "Converting $f -> $out"
 pyfr export volume "$MESH" "$f" "$out" --precision double
done
