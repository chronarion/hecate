"""Write TabbyAPI's config.yml from HECATE_* environment variables (context size, cache, drafts, port, model dir).
   HECATE_MAX_SEQ_LEN    longest single conversation in tokens (model limit 1048576)       default 262144
   HECATE_CACHE_TOKENS   KV page pool shared by all conversations (multiple of 256)        default 2 x max_seq_len
   HECATE_CACHE_MODE     KV cache format: Q6 (default), Q8, FP16 ... (ExLlamaV3 cache modes)
   HECATE_DRAFTS         MTP draft tokens per step                                          default 3
   HECATE_PORT, HECATE_MODEL_DIR (directory holding the model folder), HECATE_MODEL_NAME (the folder)"""
import os
e = os.environ.get
max_seq = int(e("HECATE_MAX_SEQ_LEN", "262144"))
cache = int(e("HECATE_CACHE_TOKENS", str(2 * max_seq)))
cache = max(cache, max_seq) // 256 * 256
cfg = f"""network:
  host: 0.0.0.0
  port: {int(e("HECATE_PORT", "8103"))}
  disable_auth: {e("HECATE_DISABLE_AUTH", "true")}
  api_servers: ["OAI"]
  web_ui: false
logging:
  log_prompt: false
  log_requests: false
model:
  model_dir: {e("HECATE_MODEL_DIR", "/models")}
  model_name: {e("HECATE_MODEL_NAME", "GLM-5.3-Hecate")}
  backend: exllamav3
  max_seq_len: {max_seq}
  cache_mode: {e("HECATE_CACHE_MODE", "Q6")}
  cache_size: {cache}
  tensor_parallel: false
  gpu_split_auto: false
  gpu_split: [{e("HECATE_GPU_BUDGET_GB", "250")}, 0]
  autosplit_reserve: [4096]
  chunk_size: {int(e("HECATE_CHUNK", "8192"))}
  reasoning: true
draft_model:
  draft_mode: mtp
  draft_num_tokens: {int(e("HECATE_DRAFTS", "3"))}
"""
open("/opt/tabbyAPI/config.yml", "w").write(cfg)
print(f"hecate: max_seq_len {max_seq}, cache {cache} tokens ({e('HECATE_CACHE_MODE', 'Q6')}), drafts {e('HECATE_DRAFTS', '3')}", flush=True)
