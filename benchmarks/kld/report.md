# GLM-5.3 KLD (reference: stock BF16; KL(ref || candidate), nats, full vocabulary)

Candidates are the release test runs: release-1 = v1.0, release-2 = v1.1, release-4 = v1.3, release-7 = v1.4, release-8 = v1.5, release-9 = v1.6, release-10 = v1.7 (v1.2 changed no numerics).

| candidate | positions | mean | 95% CI (window bootstrap) | macro mean | median | p95 | p99 | p99.9 | max | JSD | top-1 | NLL ref / cand |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| release-1 | 36848 | **0.07399** | 0.06648-0.08191 | 0.07553 | 0.00540 | 0.3368 | 1.1519 | 3.158 | 7.120 | 0.01572 | 0.9263 | 1.0974 / 1.1265 |
| release-2 | 36848 | **0.07404** | 0.06659-0.08199 | 0.07560 | 0.00551 | 0.3358 | 1.1500 | 3.264 | 6.718 | 0.01570 | 0.9250 | 1.0974 / 1.1281 |
| release-4 | 36848 | **0.07378** | 0.06534-0.08158 | 0.07521 | 0.00532 | 0.3355 | 1.1697 | 3.249 | 7.115 | 0.01566 | 0.9264 | 1.0974 / 1.1264 |
| release-7 | 36848 | **0.07408** | 0.06641-0.08217 | 0.07553 | 0.00530 | 0.3357 | 1.1653 | 3.349 | 6.330 | 0.01572 | 0.9246 | 1.0974 / 1.1273 |
| release-8 | 36848 | **0.07424** | 0.06638-0.08212 | 0.07575 | 0.00537 | 0.3336 | 1.2126 | 3.205 | 6.868 | 0.01579 | 0.9256 | 1.0974 / 1.1261 |
| release-9 | 36848 | **0.07382** | 0.06613-0.08176 | 0.07520 | 0.00532 | 0.3425 | 1.1727 | 3.102 | 6.579 | 0.01570 | 0.9258 | 1.0974 / 1.1263 |
| release-10 | 36848 | **0.07352** | 0.06579-0.08164 | 0.07518 | 0.00538 | 0.3315 | 1.1657 | 3.212 | 6.350 | 0.01560 | 0.9257 | 1.0974 / 1.1257 |

Per stratum (mean KL / top-1):

| candidate | wiki | code | chat | math | zh | long | analysis | qualification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| release-1 | 0.0806 / 0.913 | 0.0493 / 0.964 | 0.0828 / 0.893 | 0.0734 / 0.955 | 0.0646 / 0.902 | 0.1025 / 0.939 | 0.07298 | 0.07499 |
| release-2 | 0.0811 / 0.912 | 0.0492 / 0.962 | 0.0824 / 0.892 | 0.0738 / 0.953 | 0.0651 / 0.900 | 0.1021 / 0.940 | 0.07265 | 0.07542 |
| release-4 | 0.0801 / 0.914 | 0.0484 / 0.963 | 0.0842 / 0.892 | 0.0751 / 0.956 | 0.0632 / 0.904 | 0.1002 / 0.940 | 0.07282 | 0.07474 |
| release-7 | 0.0801 / 0.913 | 0.0494 / 0.962 | 0.0839 / 0.891 | 0.0761 / 0.953 | 0.0635 / 0.899 | 0.1001 / 0.939 | 0.07338 | 0.07479 |
| release-8 | 0.0803 / 0.911 | 0.0494 / 0.962 | 0.0839 / 0.893 | 0.0758 / 0.953 | 0.0651 / 0.904 | 0.0999 / 0.940 | 0.07293 | 0.07556 |
| release-9 | 0.0813 / 0.911 | 0.0484 / 0.964 | 0.0834 / 0.892 | 0.0747 / 0.954 | 0.0640 / 0.903 | 0.0994 / 0.941 | 0.07292 | 0.07471 |
| release-10 | 0.0803 / 0.912 | 0.0486 / 0.963 | 0.0816 / 0.893 | 0.0757 / 0.955 | 0.0633 / 0.902 | 0.1015 / 0.939 | 0.07284 | 0.07419 |

Per depth (position in window, mean KL):

| candidate | 0-127 | 128-511 | 512-1023 | 1024-2047 | 6143-8190 |
|---|---:|---:|---:|---:|---:|
| release-1 | 0.09888 | 0.07925 | 0.07152 | 0.06298 | 0.10255 |
| release-2 | 0.09986 | 0.07846 | 0.07253 | 0.06289 | 0.10205 |
| release-4 | 0.10167 | 0.07827 | 0.07236 | 0.06271 | 0.10019 |
| release-7 | 0.09952 | 0.07880 | 0.07264 | 0.06332 | 0.10014 |
| release-8 | 0.09844 | 0.07939 | 0.07244 | 0.06377 | 0.09991 |
| release-9 | 0.10189 | 0.07850 | 0.07151 | 0.06331 | 0.09938 |
| release-10 | 0.09886 | 0.07908 | 0.07160 | 0.06223 | 0.10147 |

By reference top-1 probability (share of positions, mean KL):

- release-1: [0.00-0.25] 4.8% 0.1492, [0.25-0.50] 12.2% 0.1521, [0.50-0.75] 13.1% 0.1715, [0.75-0.95] 14.0% 0.1256, [0.95-1.00] 55.9% 0.0147
- release-2: [0.00-0.25] 4.8% 0.1484, [0.25-0.50] 12.2% 0.1534, [0.50-0.75] 13.1% 0.1725, [0.75-0.95] 14.0% 0.1237, [0.95-1.00] 55.9% 0.0148
- release-4: [0.00-0.25] 4.8% 0.1491, [0.25-0.50] 12.2% 0.1530, [0.50-0.75] 13.1% 0.1727, [0.75-0.95] 14.0% 0.1234, [0.95-1.00] 55.9% 0.0144
- release-7: [0.00-0.25] 4.8% 0.1476, [0.25-0.50] 12.2% 0.1522, [0.50-0.75] 13.1% 0.1729, [0.75-0.95] 14.0% 0.1258, [0.95-1.00] 55.9% 0.0146
- release-8: [0.00-0.25] 4.8% 0.1505, [0.25-0.50] 12.2% 0.1532, [0.50-0.75] 13.1% 0.1720, [0.75-0.95] 14.0% 0.1260, [0.95-1.00] 55.9% 0.0146
- release-9: [0.00-0.25] 4.8% 0.1465, [0.25-0.50] 12.2% 0.1536, [0.50-0.75] 13.1% 0.1718, [0.75-0.95] 14.0% 0.1250, [0.95-1.00] 55.9% 0.0144
- release-10: [0.00-0.25] 4.8% 0.1474, [0.25-0.50] 12.2% 0.1518, [0.50-0.75] 13.1% 0.1726, [0.75-0.95] 14.0% 0.1228, [0.95-1.00] 55.9% 0.0146

Paired deltas (same positions; CI from the window bootstrap):

- release-2 - release-1: +0.00005 (95% CI -0.00053 .. +0.00062)
- release-4 - release-1: -0.00021 (95% CI -0.00101 .. +0.00072)
- release-7 - release-1: +0.00009 (95% CI -0.00081 .. +0.00133)
- release-8 - release-1: +0.00026 (95% CI -0.00052 .. +0.00096)
- release-9 - release-1: -0.00017 (95% CI -0.00097 .. +0.00066)
- release-10 - release-1: -0.00047 (95% CI -0.00112 .. +0.00037)

Worst contexts (per candidate):

- release-1 code-0@312: KL 7.12, ..." The exception raised by ArgumentParser objects when\n        there are errors" -> " with"
- release-1 long-1@7928: KL 6.31, ..." case is to do nothing, so instead of providing a\n#" -> " function"
- release-1 chat-0@167: KL 6.20, ..." British Museum. Dedicated to human history, art and culture, The" -> " British"
- release-1 math-0@1301: KL 5.80, ..." very important at home.  He tries to get home in " -> "4"
- release-2 code-0@312: KL 6.72, ..." The exception raised by ArgumentParser objects when\n        there are errors" -> " with"
- release-2 code-1@1387: KL 6.19, ..."def _is_param_expr(arg):\n    return arg is ... or" -> " isinstance"
- release-2 chat-0@1676: KL 6.08, ..." since then. Thousands of people visit Tlachihualte" -> "pet"
- release-2 long-1@7928: KL 6.07, ..." case is to do nothing, so instead of providing a\n#" -> " function"
- release-4 chat-0@167: KL 7.11, ..." British Museum. Dedicated to human history, art and culture, The" -> " British"
- release-4 code-0@312: KL 6.99, ..." The exception raised by ArgumentParser objects when\n        there are errors" -> " with"
- release-4 chat-0@1676: KL 6.86, ..." since then. Thousands of people visit Tlachihualte" -> "pet"
- release-4 math-0@1301: KL 6.38, ..." very important at home.  He tries to get home in " -> "4"
- release-7 code-0@312: KL 6.33, ..." The exception raised by ArgumentParser objects when\n        there are errors" -> " with"
- release-7 chat-0@167: KL 6.29, ..." British Museum. Dedicated to human history, art and culture, The" -> " British"
- release-7 code-1@1387: KL 6.18, ..."def _is_param_expr(arg):\n    return arg is ... or" -> " isinstance"
- release-7 long-1@7928: KL 5.82, ..." case is to do nothing, so instead of providing a\n#" -> " function"
- release-8 chat-0@167: KL 6.87, ..." British Museum. Dedicated to human history, art and culture, The" -> " British"
- release-8 code-0@312: KL 6.63, ..." The exception raised by ArgumentParser objects when\n        there are errors" -> " with"
- release-8 long-1@7928: KL 6.25, ..." case is to do nothing, so instead of providing a\n#" -> " function"
- release-8 code-1@1387: KL 6.18, ..."def _is_param_expr(arg):\n    return arg is ... or" -> " isinstance"
- release-9 math-0@1398: KL 6.58, ...">>180 miles from home\nHe was only able to drive " -> "4"
- release-9 math-0@35: KL 6.35, ..." three for breakfast every morning and bakes muffins for her friends" -> " every"
- release-9 code-0@312: KL 6.27, ..." The exception raised by ArgumentParser objects when\n        there are errors" -> " with"
- release-9 long-1@7928: KL 5.85, ..." case is to do nothing, so instead of providing a\n#" -> " function"
- release-10 chat-0@167: KL 6.35, ..." British Museum. Dedicated to human history, art and culture, The" -> " British"
- release-10 code-0@312: KL 6.29, ..." The exception raised by ArgumentParser objects when\n        there are errors" -> " with"
- release-10 chat-1@881: KL 6.13, ..." I'd avoid packing it! Haha; perhaps strawberries and P" -> "im"
- release-10 long-1@7928: KL 5.89, ..." case is to do nothing, so instead of providing a\n#" -> " function"
