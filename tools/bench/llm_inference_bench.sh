#!/bin/bash
# local-inference-lab/llm-inference-bench v0.8.0, the invocations of its GLM scripts: decode matrix (C=1..32, 30 s), standalone
# prefill (8K / 64K, 3 rounds after a warm-up) and the C=1 context sweep (0-128K). Client in a throwaway python container.
#   SRC=/path/to/llm-inference-bench OUT=./results bash llm_inference_bench.sh
OUT=${OUT:-$PWD/llmbench-out}; SRC=${SRC:?path to a llm-inference-bench checkout (v0.8.0)}; mkdir -p $OUT; L=$OUT/bench.log
echo "### bench start $(date +%T) server $(docker inspect -f '{{.Config.Image}}' hecate)" >> $L
MODEL=$(curl -s http://127.0.0.1:8103/v1/models | python3 -I -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "model $MODEL" >> $L
B() {  # B LABEL ARGS...
  local lab=$1; shift
  echo "### $lab $(date +%T): $*" >> $L
  docker run --rm --network host --user $(id -u):$(id -g) -e HOME=/out -e LLM_BENCH_NO_UPDATE_CHECK=1 \
    -v $SRC:/src:ro -v $OUT:/out python:3.12-slim sh -c \
    "pip install -q --user httpx rich >/dev/null 2>&1; python3 /src/llm_decode_bench.py --host 127.0.0.1 --port 8103 \
     --model '$MODEL' --kv-budget 524288 --no-hw-monitor --display-mode plain --output /out/$lab.json $*" >> $OUT/$lab.log 2>&1
  echo "### $lab rc=$? $(date +%T)" >> $L
}
B decode --skip-prefill --contexts 0 --concurrency 1,2,4,8,16,32 --duration 30 --max-tokens 2048
B prefill-warm --prefill-only --standalone-prefill --prefill-contexts 64k --prefill-duration 10 --token-targeting estimate --max-tokens 1
for r in 1 2 3; do
  B prefill-r$r --prefill-only --standalone-prefill --prefill-contexts 8k,64k --prefill-duration 10 --token-targeting estimate --max-tokens 1
  sleep 10
done
B ctxsweep --skip-prefill --contexts 0,16k,32k,64k,128k --concurrency 1 --duration 30 --max-tokens 2048
docker logs hecate 2>&1 | grep -E "Traceback|out of memory|RuntimeError" | tail -3 >> $L
echo "### bench done $(date +%T)" >> $L
