#pragma once

#include <ATen/Tensor.h>
#include "../graph.cuh"

// hecate: optional rank-4 output correction of an int8 linear (unset in the release): us [N][4] fp16,
// v [K][4] fp16, t fp32 scratch (>= 547 floats, zeroed once; see dense_i8.cu), modes [9+] fp32 (per-item flag, [8] = item count), see dense_i8.cu
struct DenseI8LR { at::Tensor us, v, t, modes; };

// hecate: int8 decode GEMV for 1..8 rows (EXL3_I8_LINEARS), see dense_i8.cu
void dense_i8_gr(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scale, at::Tensor& y, at::Tensor& y32,
                 at::Tensor& ctr, int spw, Graph* graph, const DenseI8LR* lr = nullptr);
void dense_i8(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scale, at::Tensor& y, at::Tensor& y32,
              at::Tensor& ctr, int64_t spw);
void dense_i8_lr(const at::Tensor& x, const at::Tensor& w8, const at::Tensor& scale, at::Tensor& y, at::Tensor& y32,
                 at::Tensor& ctr, int64_t spw, const at::Tensor& us, const at::Tensor& v, const at::Tensor& t,
                 const at::Tensor& modes);
// hecate: gated-MLP gate + up int8 GEMVs and the SiLU product in one kernel (SPW 4), see dense_i8.cu
void dense_i8_gu_act_gr(const at::Tensor& x, const at::Tensor& wg, const at::Tensor& sg, at::Tensor& y32g, at::Tensor& ctr,
                        const at::Tensor& wu, const at::Tensor& su, at::Tensor& y32u, at::Tensor& act, int spw, Graph* graph);
void dense_i8_gu_act(const at::Tensor& x, const at::Tensor& wg, const at::Tensor& sg, at::Tensor& y32g, at::Tensor& ctr,
                     const at::Tensor& wu, const at::Tensor& su, at::Tensor& y32u, at::Tensor& act);
// hecate: two int8 GEMVs on one x (SPW 4 both), one launch, see dense_i8.cu
void dense_i8_dual_gr(const at::Tensor& x, const at::Tensor& wa, const at::Tensor& sa, at::Tensor& ya, at::Tensor& y32a,
                      at::Tensor& ctra, const at::Tensor& wb, const at::Tensor& sb, at::Tensor& yb, at::Tensor& y32b,
                      at::Tensor& ctrb, Graph* graph);
void dense_i8_dual(const at::Tensor& x, const at::Tensor& wa, const at::Tensor& sa, at::Tensor& ya, at::Tensor& y32a,
                   at::Tensor& ctra, const at::Tensor& wb, const at::Tensor& sb, at::Tensor& yb, at::Tensor& y32b,
                   at::Tensor& ctrb);
