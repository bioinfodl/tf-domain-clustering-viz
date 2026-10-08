#!/usr/bin/env bash
# Refresh this repository from a checkout of the toolkit repository.
# Usage: scripts/sync_from_toolkit.sh /path/to/toolkit
set -euo pipefail

TOOLKIT="${1:?usage: $0 /path/to/toolkit}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$TOOLKIT/TFilament"

for f in domain_clustering_viz.py tf_domains.csv; do
    [ -f "$SRC/$f" ] || { echo "missing: $SRC/$f" >&2; exit 1; }
done

# Same app; only the run command in its docstring differs.
sed 's#streamlit run TFilament/domain_clustering_viz.py#streamlit run app.py#' \
    "$SRC/domain_clustering_viz.py" > "$HERE/app.py"
cp "$SRC/tf_domains.csv" "$HERE/data/tf_domains.csv"

echo "Updated app.py and data/tf_domains.csv from $SRC"
git -C "$HERE" status --short
