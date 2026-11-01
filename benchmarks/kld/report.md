# GLM-5.3 KLD (reference: stock BF16; KL(ref || candidate), nats, full vocabulary)

| candidate | positions | mean | 95% CI (window bootstrap) | macro mean | median | p95 | p99 | p99.9 | max | JSD | top-1 | NLL ref / cand |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hecate | 36848 | **0.07399** | 0.06648-0.08191 | 0.07553 | 0.00540 | 0.3368 | 1.1519 | 3.158 | 7.120 | 0.01572 | 0.9263 | 1.0974 / 1.1265 |

Per stratum (mean KL / top-1):

| candidate | wiki | code | chat | math | zh | long | analysis | qualification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hecate | 0.0806 / 0.913 | 0.0493 / 0.964 | 0.0828 / 0.893 | 0.0734 / 0.955 | 0.0646 / 0.902 | 0.1025 / 0.939 | 0.07298 | 0.07499 |

Per depth (position in window, mean KL):

| candidate | 0-127 | 128-511 | 512-1023 | 1024-2047 | 6143-8190 |
|---|---:|---:|---:|---:|---:|
| hecate | 0.09888 | 0.07925 | 0.07152 | 0.06298 | 0.10255 |

By reference top-1 probability (share of positions, mean KL):

- hecate: [0.00-0.25] 4.8% 0.1492, [0.25-0.50] 12.2% 0.1521, [0.50-0.75] 13.1% 0.1715, [0.75-0.95] 14.0% 0.1256, [0.95-1.00] 55.9% 0.0147

Worst contexts (per candidate):

- hecate code-0@312: KL 7.12, ..." The exception raised by ArgumentParser objects when\n        there are errors" -> " with"
- hecate long-1@7928: KL 6.31, ..." case is to do nothing, so instead of providing a\n#" -> " function"
- hecate chat-0@167: KL 6.20, ..." British Museum. Dedicated to human history, art and culture, The" -> " British"
- hecate math-0@1301: KL 5.80, ..." very important at home.  He tries to get home in " -> "4"
