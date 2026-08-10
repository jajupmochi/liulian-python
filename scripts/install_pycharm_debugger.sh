#!/bin/bash
# Install the PyCharm-bundled pydevd debugger egg into this repo's .venv.
#
# WHY: the Ray-trial debug attach (hpo_debug_attach, docs/debugging_ray_hpo.md)
# needs pydevd-pycharm matching the EXACT PyCharm build. PyPI lags PyCharm
# releases (e.g. build 262.8665.369 was not on PyPI), so the reliable source is
# the egg SHIPPED INSIDE the PyCharm installation. uv cannot install eggs, so
# this script unzips it straight into site-packages. Re-run after any venv
# rebuild (uv pip sync/install wipes it) or PyCharm upgrade.
#
# Usage (from the repo root):  bash scripts/install_pycharm_debugger.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Locate the PyCharm installation from a running process, else the snap dir.
PYCHARM_BIN=$(readlink "/proc/$(pgrep -f pycharm | head -1)/exe" 2>/dev/null || true)
if [ -n "${PYCHARM_BIN}" ]; then
  PYCHARM_HOME=$(dirname "$(dirname "$PYCHARM_BIN")")
else
  PYCHARM_HOME=$(ls -d /snap/pycharm-professional/*/ 2>/dev/null | sort -V | tail -1)
fi
EGG="${PYCHARM_HOME%/}/debug-eggs/pydevd-pycharm.egg"
if [ ! -f "$EGG" ]; then
  echo "ERROR: debugger egg not found at $EGG — is PyCharm installed? (snap: /snap/pycharm-professional/<rev>/debug-eggs/)" >&2
  exit 1
fi

# 2. Remove any PyPI-installed copy (would shadow/conflict), then extract.
uv pip uninstall pydevd-pycharm 2>/dev/null || true
SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
unzip -qo "$EGG" -d "$SP" -x "EGG-INFO/*"

# 3. Verify.
.venv/bin/python - << 'EOF'
import inspect
import pydevd_pycharm
sig = inspect.signature(pydevd_pycharm.settrace)
assert 'suspend' in sig.parameters
print(f'OK: pydevd_pycharm installed from the PyCharm egg; settrace{sig}')
EOF
echo "Done. Egg source: $EGG"
