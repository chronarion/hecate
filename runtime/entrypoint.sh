#!/bin/sh
# Hecate entrypoint: render the TabbyAPI config from HECATE_* env, import the tier split (int4 / trellis / Grace expert
# tiers), the folder loader and the Hecate runtime (stock int8 base, CUDA graphs, fused router) before TabbyAPI loads the
# model, then warm up.
set -e
cd /opt/tabbyAPI
if [ -n "$HECATE_CONFIG" ]; then cp "$HECATE_CONFIG" /opt/tabbyAPI/config.yml; else python /opt/hecate/render_config.py; fi
python -c "
import sys, runpy
sys.path.insert(0, '/opt/hecate/tier'); import tier_split
sys.path.insert(0, '/opt/hecate/core'); import hecate_loader, hecate_runtime
sys.path.insert(0, '/opt/tabbyAPI'); sys.argv = ['main.py']
runpy.run_path('main.py', run_name='__main__')" & s=$!
trap 'kill -TERM $s; wait $s' TERM INT
[ "${EXL3_WARMUP:-1}" = 1 ] && python -I /opt/hecate/warmup.py
wait $s
