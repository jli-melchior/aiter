# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HCA-path compress + norm+rope+scatter kernels -- **gfx1250 (RDNA4, wave32)**.

Port of ``fused_compress_attn_hca.py`` (wave64) to gfx1250 wave32.
Key differences:
  - BLOCK_THREADS = 32 (wave32)
  - SLICE = 32 (head_dim elements per block)
  - VEC = SLICE_SZ / 32 (vs /64)
  - Kernel B: D=512 -> VEC=16, requires split load/store paths
  - Kernel names suffixed with "w32"

See ``fused_compress_attn_hca.py`` for the original wave64 documentation.
"""

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import Int32, Stream, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import tdm_ops_gfx1250 as tdm_ops

from .communication_ops_utils import atomic_add_agent
from .fused_compress_attn_common import (
    block_base_bytes_i64,
    emit_group_fp8_nm_asm_scatter,
    state_slot_byte_offset,
)
from .tensor_shim import _run_compiled

BLOCK_THREADS = 32  # 1 wave32 (RDNA4 / gfx1250)
SLICE = 32  # head_dim elements per block (grid-Y split)
_NEG_INF = float("-inf")
_LOG2E = math.log2(math.e)


# ============================================================================
# Kernel A: compress_forward with multi-wave LDS K-split
# ============================================================================


def _build_compress_forward_kernel(
    *,
    head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    slice_size: int = 64,
    enable_prefetch_input: bool = False,
    enable_tdm: bool = False,
):
    """HCA compress_forward with K-axis parallelized across multiple waves.

    Architecture (multi-wave LDS K-split with per-thread VEC):
      - Grid:  (num_compress, NUM_SPLIT=head_dim/slice_size)
      - Block: BLOCK_THREADS = 64 * k_split_num_waves (8 waves on AMD).
      - Per block covers ``slice_size`` head_dim elements of one boundary.
      - Per thread owns ``VEC = slice_size / 64`` contiguous head_dim
        elements starting at lid*VEC within the block's slice.
      - K=ratio split across ``k_split_num_waves`` waves; each wave processes
        K_PER_WAVE = K/NW positions (= 16 for K=128, NW=8).
      - Per-wave local online-softmax -> (m_local, kv_local, w_local) lists
        of VEC values per thread.
      - LDS cross-wave reduction: only wave 0 active; each thread reads
        NW*VEC values from LDS, computes VEC reduced compressed values,
        writes them out via vector buffer_store.

    Tuning knobs:
      - ``k_split_num_waves`` (= NW): trades K-serial chain length for LDS
        reduce cost. Small N -> larger NW (more waves -> more CU coverage);
        large N -> smaller NW (less LDS overhead).
      - ``slice_size``: VEC width per thread. slice_size=64 -> VEC=1 scalar
        (more blocks per boundary -> small-N champion); slice_size=512 ->
        VEC=8 (1 block per boundary, v1-like -> large-N coalesced HBM).

    Phase 1 (state cache) is integrated by splitting each wave's K range at
    ``clamp(window_len, k_start, k_end)`` into a Phase 1 sub-loop reading
    kv_state + score_state (padded softmax when ``s < 0``) and a Phase 2
    sub-loop reading kv_in + score_in. Phase 2 in_row is clamped to >= 0
    so wasted reads in pure-Phase-1 iters stay in-bounds.
    """
    assert (
        head_dim % slice_size == 0
    ), f"head_dim={head_dim} must be divisible by slice_size={slice_size}"
    assert (
        slice_size % 32 == 0
    ), f"slice_size={slice_size} must be a multiple of 32 (wave width)"
    assert slice_size // 32 in (
        1,
        2,
        4,
        8,
        16,
    ), f"VEC={slice_size // 32} must be 1, 2, 4, 8, or 16"
    assert (
        ratio % k_split_num_waves == 0
    ), f"K={ratio} must divide evenly across {k_split_num_waves} waves"
    assert state_size >= ratio, f"state_size={state_size} must be >= K={ratio}"
    assert not (
        enable_tdm and enable_prefetch_input
    ), "TDM and prefetch are mutually exclusive for now"
    D = head_dim
    K = ratio
    DIM_FULL = D
    SLICE_SZ = slice_size
    VEC = SLICE_SZ // BLOCK_THREADS  # per-lane head_dim element count
    NUM_SPLIT = D // SLICE_SZ
    NW = k_split_num_waves
    BLOCK_TH = BLOCK_THREADS * NW
    K_PER_WAVE = K // NW

    # LDS layout: three independent fp32 arrays, each [NW * slice_size].
    LDS_M_ELEMS = NW * SLICE_SZ
    LDS_KV_ELEMS = NW * SLICE_SZ
    LDS_W_ELEMS = NW * SLICE_SZ

    if enable_tdm:
        KV_DB_TOTAL = K * SLICE_SZ

        @fx.struct
        class SharedStorage:
            lds_kv_db: fx.Array[fx.BFloat16, KV_DB_TOTAL, 16]
            lds_m: fx.Array[fx.Float32, LDS_M_ELEMS, 16]
            lds_kv: fx.Array[fx.Float32, LDS_KV_ELEMS, 16]
            lds_w: fx.Array[fx.Float32, LDS_W_ELEMS, 16]

    else:

        @fx.struct
        class SharedStorage:
            lds_m: fx.Array[fx.Float32, LDS_M_ELEMS, 16]
            lds_kv: fx.Array[fx.Float32, LDS_KV_ELEMS, 16]
            lds_w: fx.Array[fx.Float32, LDS_W_ELEMS, 16]

    _tdm_tag = "_TDM" if enable_tdm else ""
    _kname = f"hca_compress_forward_w32_D{D}_R{ratio}_NW{NW}_SL{SLICE_SZ}_S{state_size}{_tdm_tag}_flydsl"
    fm_fast = arith.FastMathFlags.fast

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_TH, 1, 1])
    def kernel(
        kv_in: fx.Tensor,
        kv_in_row_stride: Int32,
        score_in: fx.Tensor,
        score_in_row_stride: Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,  # [num_slots, STATE_SIZE, DIM_FULL] f32
        kv_state_slot_stride: Int32,  # f32 elements
        kv_state_pos_stride: Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: Int32,
        score_state_pos_stride: Int32,
        state_slot_mapping: fx.Tensor,  # [bs] i32
        ape: fx.Tensor,
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32

        pid = fx.block_idx.x
        sid = fx.block_idx.y
        tid = fx.thread_idx.x  # 0..BLOCK_TH-1

        c_zero_i32 = arith.constant(0, type=i32)
        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)

        def fexp_f32(x):
            return llvm.call_intrinsic(
                f32, "llvm.amdgcn.exp2.f32", [x * c_log2e], [], []
            )

        # Per-thread wave / lane (block-local); tid >= 0 -> unsigned div/rem
        # (divui/remui). Wrap back to Int32 for the signed i32 consumers.
        wid = fx.Int32((fx.Uint32(tid) // BLOCK_THREADS).ir_value())  # -> [0, NW)
        lid = fx.Int32((fx.Uint32(tid) % BLOCK_THREADS).ir_value())  # -> [0, 32)

        # -- Load plan row ----------------------------------------------
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_vec = fx.Vector(
            buffer_ops.buffer_load(plan_rsrc, fx.Int32(pid) * 4, vec_width=4, dtype=i32)
        )
        ragged_id = plan_vec[0]
        batch_id = plan_vec[1]
        position = plan_vec[2]
        window_len = plan_vec[3]

        # Sentinel-skip: run the whole body only for position >= 0, as a closure
        # under a runtime `if` (rewriter sees an opaque call -> scf.if).
        def _body():
            # Per-thread head_dim base: each thread owns VEC contiguous
            # elements starting at slice_base + lid * VEC.
            col_off_base = fx.Int32(sid) * SLICE_SZ + lid * VEC

            slot_map_rsrc = buffer_ops.create_buffer_resource(
                state_slot_mapping, max_size=True
            )
            slot = buffer_ops.buffer_load(
                slot_map_rsrc, batch_id, vec_width=1, dtype=i32
            )

            kv_in_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            score_in_rsrc = buffer_ops.create_buffer_resource(score_in, max_size=True)
            # Rebased onto this program's slot — see `state_slot_byte_offset`.
            kv_state_rsrc = buffer_ops.create_buffer_resource(
                kv_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, kv_state_slot_stride),
            )
            score_state_rsrc = buffer_ops.create_buffer_resource(
                score_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, score_state_slot_stride),
            )
            ape_rsrc = buffer_ops.create_buffer_resource(ape, max_size=True)

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()

            def _load_bf16_vec_to_f32(rsrc, base_off_elems_i32):
                """Load VEC contiguous bf16 elements starting at
                ``base_off_elems_i32`` -> list of VEC f32 values.

                VEC=1: unaligned-safe scalar via dword + bit-extract.
                VEC>=2: vectorized i32 buffer_load + bitcast to bf16.
                """
                base_off = fx.Int32(base_off_elems_i32)
                # logical (unsigned) >> for the dword offset (base_off >= 0): fx
                # Int32 >> is arithmetic -> use Uint32 to keep shrui/v_lshrrev_b32.
                off_dw = fx.Int32((fx.Uint32(base_off_elems_i32) >> 1).ir_value())
                if const_expr(VEC == 1):
                    lane_in_dw = base_off & 1
                    raw_s = buffer_ops.buffer_load(rsrc, off_dw, vec_width=1, dtype=i32)
                    # logical shift for the hi-word extract too.
                    hi = fx.Int32((fx.Uint32(raw_s) >> 16).ir_value())
                    lo_or_hi = arith.select(
                        arith.cmpi(CmpIPredicate.eq, lane_in_dw.ir_value(), c_zero_i32),
                        raw_s,
                        hi.ir_value(),
                    )
                    lo16 = arith.andi(lo_or_hi, arith.constant(0xFFFF, type=i32))
                    lo16_v = fx.Vector.from_elements([lo16], dtype=fx.Int32)
                    bf16_pair = lo16_v.bitcast(fx.BFloat16)
                    # raw f32 for the explicit-fastmath float layer downstream.
                    return [bf16_pair[0].to(fx.Float32).ir_value()]
                else:
                    # base must be VEC-aligned (caller guarantees by
                    # col_off_base = sid*SLICE + lid*VEC, both multiples of VEC).
                    dwords = VEC // 2  # VEC bf16 = VEC*2 bytes
                    if const_expr(dwords == 1):
                        # buffer_load(vec_width=1) returns scalar i32; wrap
                        # into vec<1xi32> before bitcast to vec<2xbf16>.
                        raw_s = buffer_ops.buffer_load(
                            rsrc, off_dw, vec_width=1, dtype=i32
                        )
                        raw = fx.Vector.from_elements([raw_s], dtype=fx.Int32)
                    elif const_expr(dwords <= 4):
                        raw = fx.Vector(
                            buffer_ops.buffer_load(
                                rsrc, off_dw, vec_width=dwords, dtype=i32
                            )
                        )
                    else:
                        # dwords > 4 (VEC=16 -> dwords=8): HW max is dwordx4,
                        # split into 2× dwordx4 loads.
                        half_dw = dwords // 2
                        r0 = fx.Vector(
                            buffer_ops.buffer_load(
                                rsrc, off_dw, vec_width=half_dw, dtype=i32
                            )
                        )
                        r1 = fx.Vector(
                            buffer_ops.buffer_load(
                                rsrc, off_dw + half_dw, vec_width=half_dw, dtype=i32
                            )
                        )
                        raw = fx.Vector.from_elements(
                            [r0[i] for i in range(half_dw)]
                            + [r1[i] for i in range(half_dw)],
                            dtype=fx.Int32,
                        )
                    vec_bf16 = raw.bitcast(fx.BFloat16)
                    # raw f32 for the explicit-fastmath float layer downstream.
                    return [vec_bf16[i].to(fx.Float32).ir_value() for i in range(VEC)]

            def _load_f32_vec(rsrc, base_off_elems_i32):
                """Load VEC f32 (raw ir.Values) starting at base -> list of VEC."""
                if const_expr(VEC <= 4):
                    raw = buffer_ops.buffer_load(
                        rsrc, base_off_elems_i32, vec_width=VEC, dtype=f32
                    )
                    if const_expr(VEC == 1):
                        # vec_width=1 returns scalar, not 1-vec.
                        return [raw]
                    return [fx.Vector(raw)[i].ir_value() for i in range(VEC)]
                else:
                    # VEC > 4: AMD HW max is dwordx4 -> ceil(VEC/4) loads.
                    quarter = 4
                    n_chunks = VEC // quarter
                    result = []
                    for q in range_constexpr(n_chunks):
                        r = fx.Vector(
                            buffer_ops.buffer_load(
                                rsrc,
                                fx.Int32(base_off_elems_i32) + q * quarter,
                                vec_width=quarter,
                                dtype=f32,
                            )
                        )
                        result.extend(r[j].ir_value() for j in range(quarter))
                    return result

            def _issue_phase2_loads(k_i32):
                """Phase 2 (ragged input) loads. Returns (kv_list, sc_list,
                ape_list) each of length VEC."""
                k = fx.Int32(k_i32)
                ape_row = fx.Int32((fx.Uint32(k_i32) % ratio).ir_value())
                in_row_raw = fx.Int32(ragged_id) - (fx.Int32(K - 1) - k)
                in_row = fx.max(in_row_raw, fx.Int32(0))
                base_in_off = in_row * fx.Int32(kv_in_row_stride) + col_off_base
                kv = _load_bf16_vec_to_f32(kv_in_rsrc, base_in_off)
                base_sc_off = in_row * fx.Int32(score_in_row_stride) + col_off_base
                sc = _load_bf16_vec_to_f32(score_in_rsrc, base_sc_off)
                base_ape_off = ape_row * DIM_FULL + col_off_base
                ape_v = _load_f32_vec(ape_rsrc, base_ape_off)
                return kv, sc, ape_v

            def _issue_score_ape_loads(k_i32):
                """Phase 2 score + APE loads only (kv comes from LDS in TDM
                mode). Returns (sc_list, ape_list) each of length VEC."""
                k = fx.Int32(k_i32)
                in_row_raw = fx.Int32(ragged_id) - (fx.Int32(K - 1) - k)
                in_row = fx.max(in_row_raw, fx.Int32(0))
                base_sc_off = in_row * fx.Int32(score_in_row_stride) + col_off_base
                sc = _load_bf16_vec_to_f32(score_in_rsrc, base_sc_off)
                ape_row = fx.Int32((fx.Uint32(k_i32) % ratio).ir_value())
                base_ape_off = ape_row * DIM_FULL + col_off_base
                ape_v = _load_f32_vec(ape_rsrc, base_ape_off)
                return sc, ape_v

            def _read_kv_from_lds(k_i32, q_lo, buf_base):
                """Read VEC bf16 kv values from TDM double-buffered LDS."""
                k = fx.Int32(k_i32)
                lds_row = k - fx.Int32(q_lo)
                lds_off = fx.Int32(buf_base) + lds_row * SLICE_SZ + lid * VEC
                if const_expr(VEC >= 2):
                    lds_off_dw = fx.Int32((fx.Uint32(lds_off) >> 1).ir_value())
                    kv_lds_i32 = fx.recast_iter(fx.Int32, kv_db_ptr)
                    raw_vec = fx.Vector(
                        fx.ptr_load(
                            fx.add_offset(kv_lds_i32, lds_off_dw),
                            T.vec(VEC // 2, T.i32),
                        )
                    )
                    vec_bf16 = raw_vec.bitcast(fx.BFloat16)
                    return [vec_bf16[i].to(fx.Float32).ir_value() for i in range(VEC)]
                else:
                    result = []
                    for _vi in range_constexpr(VEC):
                        bf16_v = fx.ptr_load(kv_db_ptr + (lds_off + _vi))
                        result.append(bf16_v.to(fx.Float32).ir_value())
                    return result

            def _issue_phase1_loads(k_i32):
                """Phase 1 (state cache) loads. Returns (kv_list, sc_padded_list)
                each of length VEC. Score is -inf when s < 0."""
                s = (fx.Int32(position) - fx.Int32(K - 1) + fx.Int32(k_i32)).ir_value()
                is_pad = arith.cmpi(CmpIPredicate.slt, s, c_zero_i32)
                s_safe = fx.Int32(arith.select(is_pad, c_zero_i32, s))
                ring = fx.Int32((fx.Uint32(s_safe.ir_value()) % state_size).ir_value())
                # Slot term already folded into the descriptor base.
                base_kv_off = ring * fx.Int32(kv_state_pos_stride) + col_off_base
                base_sc_off = ring * fx.Int32(score_state_pos_stride) + col_off_base
                kv_list = _load_f32_vec(kv_state_rsrc, base_kv_off)
                sc_list = _load_f32_vec(score_state_rsrc, base_sc_off)
                sc_padded = [
                    arith.select(is_pad, c_neg_inf, sc_list[i]) for i in range(VEC)
                ]
                return kv_list, sc_padded

            def _softmax_step_padded(
                m_old_list, kv_old_list, w_old_list, score_k_list, kv_k_list
            ):
                """Padding-aware vector softmax step over VEC lanes. When
                score_k == -inf, w_k is forced to 0 (avoids NaN when m_old
                is also -inf). Safe in both Phase 1 (padding can occur) and
                Phase 2 (score finite -> pad-select branch is dead code).
                """
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = m_old_list[i]
                    kv_old = kv_old_list[i]
                    w_old = w_old_list[i]
                    score_k = score_k_list[i]
                    kv_k = kv_k_list[i]
                    m_new = fx.max(fx.Float32(m_old), fx.Float32(score_k)).ir_value()
                    is_first = arith.cmpf(CmpFPredicate.OEQ, m_old, c_neg_inf)
                    scale_active = fexp_f32(arith.subf(m_old, m_new))
                    scale_v = arith.select(is_first, c_zero_f32, scale_active)
                    wk_active = fexp_f32(arith.subf(score_k, m_new))
                    is_pad_score = arith.cmpf(CmpFPredicate.OEQ, score_k, c_neg_inf)
                    w_k = arith.select(is_pad_score, c_zero_f32, wk_active)
                    # Explicit fastmath float layer: fx `+`/`*` drop fastmath<fast>
                    # here (the rocdl-fastmath pass does not re-add it on gfx1250)
                    # -> ISA drift (fmac vs split add/mul). Kept raw arith.*FOp.
                    new_kv.append(
                        arith.AddFOp(
                            arith.MulFOp(kv_old, scale_v, fastmath=fm_fast).result,
                            arith.MulFOp(w_k, kv_k, fastmath=fm_fast).result,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_w.append(
                        arith.AddFOp(
                            arith.MulFOp(w_old, scale_v, fastmath=fm_fast).result,
                            w_k,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_m.append(m_new)
                return new_m, new_kv, new_w

            # -- TDM: pipelined kv_in double-buffer (Q0 issued now) --------
            if const_expr(enable_tdm):
                tdm_first_row = fx.max(
                    fx.Int32(ragged_id) - fx.Int32(K - 1), fx.Int32(0)
                )
                kv_db_ptr = lds.lds_kv_db.ptr

                col_start_idx = arith.index_cast(
                    T.index, (fx.Int32(sid) * SLICE_SZ).ir_value()
                )

                kv_db_view = fx.Tensor(
                    fx.make_view(
                        kv_db_ptr,
                        fx.make_layout((K, SLICE_SZ), (SLICE_SZ, 1)),
                    )
                )

                q0_row_idx = arith.index_cast(T.index, tdm_first_row.ir_value())
                tdm_desc = tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=kv_in,
                    lds_memref=kv_db_view,
                    global_offset=(q0_row_idx, col_start_idx),
                    tensor_shape=(K, D),
                    strides=(kv_in_row_stride, 1),
                    tile_shape=(K, SLICE_SZ),
                    elem_bytes=2,
                    num_warps=NW,
                )
                tdm_ops.tensor_load_2d(tdm_desc)

            # -- Wave's K range: [wid * K_PER_WAVE, (wid+1) * K_PER_WAVE) --
            k_start_i32 = wid * K_PER_WAVE
            k_end_i32 = k_start_i32 + K_PER_WAVE

            # Split point inside this wave's K range. Each wave sees a
            # window_len-dependent slice of Phase 1 followed by Phase 2.
            # Cases (`wl = window_len`):
            #   wl <= k_start:  pure Phase 2 (entire wave is input)
            #   wl >= k_end:    pure Phase 1 (entire wave is state cache)
            #   else:          mixed (Phase 1 in [k_start, wl), Phase 2 in [wl, k_end))
            # ``split`` = clamp(wl, k_start, k_end) gives the boundary;
            # both sub-loops are empty when their bound collapses, so any
            # of the three cases naturally falls out.
            # split = clamp(window_len, k_start, k_end) -> signed int max/min.
            split_i32 = fx.min(fx.max(window_len, k_start_i32), k_end_i32)

            # State is 3*VEC scalars: m_lane[VEC] + kv_lane[VEC] + w_lane[VEC].
            init_m = [c_neg_inf for _ in range(VEC)]
            init_kv = [c_zero_f32 for _ in range(VEC)]
            init_w = [c_zero_f32 for _ in range(VEC)]
            init_state = init_m + init_kv + init_w

            # Sub-loop 1: Phase 1 sub-range [k_start, split). Reads state
            # cache; padded softmax (score can be -inf).
            phase1_local = init_state
            for k_static, state in range(
                k_start_i32.ir_value(), split_i32.ir_value(), 1, init=init_state
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = fx.Int32(k_static)
                kv_v, sc_v = _issue_phase1_loads(k_i32)
                new_m, new_kv, new_w = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, sc_v, kv_v
                )
                phase1_local = yield list(new_m) + list(new_kv) + list(new_w)

            # TDM: wait for kv_in tile to land in LDS before Phase 2 reads.
            if const_expr(enable_tdm):
                tdm_ops.tensor_wait(0)
                fx.rocdl.s_wait_dscnt(0)
                gpu.barrier()

            # Sub-loop 2: Phase 2 sub-range [split, k_end). Reads input.
            # Carry Phase 1's accumulator through as init.
            if const_expr(enable_tdm):
                _p2_count = fx.max(k_end_i32 - split_i32, fx.Int32(0))
                _p2_even = fx.Int32((fx.Uint32(_p2_count.ir_value()) & ~1).ir_value())
                _k_end_u2 = split_i32 + _p2_even

                _k_pro0 = split_i32
                _k_pro1 = split_i32 + 1
                _pf0_sc, _pf0_ape = _issue_score_ape_loads(_k_pro0)
                _pf1_sc, _pf1_ape = _issue_score_ape_loads(_k_pro1)
                _pf_init = (
                    list(phase1_local)
                    + list(_pf0_sc)
                    + list(_pf0_ape)
                    + list(_pf1_sc)
                    + list(_pf1_ape)
                )

                _loop_final = _pf_init
                for k_static, state in range(
                    split_i32.ir_value(),
                    _k_end_u2.ir_value(),
                    2,
                    init=_pf_init,
                ):
                    m_lane = list(state[0:VEC])
                    kv_lane = list(state[VEC : 2 * VEC])
                    w_lane = list(state[2 * VEC : 3 * VEC])
                    pf0_sc = list(state[3 * VEC : 4 * VEC])
                    pf0_ape = list(state[4 * VEC : 5 * VEC])
                    pf1_sc = list(state[5 * VEC : 6 * VEC])
                    pf1_ape = list(state[6 * VEC : 7 * VEC])

                    k_i32 = fx.Int32(k_static)
                    nxt0_sc, nxt0_ape = _issue_score_ape_loads(k_i32 + 2)
                    nxt1_sc, nxt1_ape = _issue_score_ape_loads(k_i32 + 3)

                    kv_a = _read_kv_from_lds(k_i32, 0, 0)
                    sc_a = [
                        arith.AddFOp(pf0_sc[i], pf0_ape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    m_a, kv_a_acc, w_a = _softmax_step_padded(
                        m_lane, kv_lane, w_lane, sc_a, kv_a
                    )

                    kv_b = _read_kv_from_lds(k_i32 + 1, 0, 0)
                    sc_b = [
                        arith.AddFOp(pf1_sc[i], pf1_ape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    m_b, kv_b_acc, w_b = _softmax_step_padded(
                        m_a, kv_a_acc, w_a, sc_b, kv_b
                    )
                    _loop_final = yield (
                        list(m_b)
                        + list(kv_b_acc)
                        + list(w_b)
                        + list(nxt0_sc)
                        + list(nxt0_ape)
                        + list(nxt1_sc)
                        + list(nxt1_ape)
                    )

                _tail_carry = list(_loop_final[0 : 3 * VEC])
                _tail_final = _tail_carry
                for _tk_static, _tstate in range(
                    _k_end_u2.ir_value(),
                    k_end_i32.ir_value(),
                    1,
                    init=_tail_carry,
                ):
                    _tm = list(_tstate[0:VEC])
                    _tkv = list(_tstate[VEC : 2 * VEC])
                    _tw = list(_tstate[2 * VEC : 3 * VEC])
                    _tk = fx.Int32(_tk_static)
                    _tkv_lds = _read_kv_from_lds(_tk, 0, 0)
                    _tsc, _tape = _issue_score_ape_loads(_tk)
                    _tscore = [
                        arith.AddFOp(_tsc[i], _tape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    _tnm, _tnkv, _tnw = _softmax_step_padded(
                        _tm, _tkv, _tw, _tscore, _tkv_lds
                    )
                    _tail_final = yield (list(_tnm) + list(_tnkv) + list(_tnw))
                final = list(_tail_final)
            elif const_expr(not enable_prefetch_input):
                final = phase1_local
                for k_static, state in range(
                    split_i32.ir_value(), k_end_i32.ir_value(), 1, init=phase1_local
                ):
                    m_lane = list(state[0:VEC])
                    kv_lane = list(state[VEC : 2 * VEC])
                    w_lane = list(state[2 * VEC : 3 * VEC])
                    k_i32 = fx.Int32(k_static)
                    p2_kv, p2_sc, p2_ape = _issue_phase2_loads(k_i32)
                    p2_score = [
                        arith.AddFOp(p2_sc[i], p2_ape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    new_m, new_kv, new_w = _softmax_step_padded(
                        m_lane, kv_lane, w_lane, p2_score, p2_kv
                    )
                    final = yield list(new_m) + list(new_kv) + list(new_w)
            else:
                # Phase 2 with 2x-unrolled prefetch: prologue issues
                # loads for the first 2 iterations, each loop body
                # processes a pair of iterations and issues the next
                # pair, halving loop overhead and doubling outstanding
                # memory requests.
                c_k_end_m1 = k_end_i32 - 1

                k_pro0 = fx.min(split_i32, c_k_end_m1)
                k_pro1 = fx.min(split_i32 + 1, c_k_end_m1)
                p0_kv, p0_sc, p0_ape = _issue_phase2_loads(k_pro0)
                p1_kv, p1_sc, p1_ape = _issue_phase2_loads(k_pro1)
                init_pf = (
                    list(phase1_local)
                    + list(p0_kv)
                    + list(p0_sc)
                    + list(p0_ape)
                    + list(p1_kv)
                    + list(p1_sc)
                    + list(p1_ape)
                )

                p2_count = fx.max(c_k_end_m1 - split_i32, fx.Int32(0))
                p2_even = fx.Int32((fx.Uint32(p2_count.ir_value()) & ~1).ir_value())
                k_end_u2 = split_i32 + p2_even

                loop_final = init_pf
                for k_static, state in range(
                    split_i32.ir_value(), k_end_u2.ir_value(), 2, init=init_pf
                ):
                    m_lane = list(state[0:VEC])
                    kv_lane = list(state[VEC : 2 * VEC])
                    w_lane = list(state[2 * VEC : 3 * VEC])
                    pf0_kv = list(state[3 * VEC : 4 * VEC])
                    pf0_sc = list(state[4 * VEC : 5 * VEC])
                    pf0_ape = list(state[5 * VEC : 6 * VEC])
                    pf1_kv = list(state[6 * VEC : 7 * VEC])
                    pf1_sc = list(state[7 * VEC : 8 * VEC])
                    pf1_ape = list(state[8 * VEC : 9 * VEC])

                    k_i32 = fx.Int32(k_static)
                    nxt0_kv, nxt0_sc, nxt0_ape = _issue_phase2_loads(k_i32 + 2)
                    nxt1_kv, nxt1_sc, nxt1_ape = _issue_phase2_loads(
                        fx.min(k_i32 + 3, c_k_end_m1)
                    )

                    sc_a = [
                        arith.AddFOp(pf0_sc[i], pf0_ape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    m_a, kv_a, w_a = _softmax_step_padded(
                        m_lane, kv_lane, w_lane, sc_a, pf0_kv
                    )
                    sc_b = [
                        arith.AddFOp(pf1_sc[i], pf1_ape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    m_b, kv_b, w_b = _softmax_step_padded(m_a, kv_a, w_a, sc_b, pf1_kv)
                    loop_final = yield (
                        list(m_b)
                        + list(kv_b)
                        + list(w_b)
                        + list(nxt0_kv)
                        + list(nxt0_sc)
                        + list(nxt0_ape)
                        + list(nxt1_kv)
                        + list(nxt1_sc)
                        + list(nxt1_ape)
                    )

                is_p2 = arith.cmpi(
                    CmpIPredicate.slt,
                    split_i32.ir_value(),
                    k_end_i32.ir_value(),
                )
                is_p2_ge2 = arith.cmpi(
                    CmpIPredicate.slt,
                    (split_i32 + 1).ir_value(),
                    k_end_i32.ir_value(),
                )
                is_odd = arith.cmpi(
                    CmpIPredicate.ne,
                    (p2_count & 1).ir_value(),
                    arith.constant(0, type=i32),
                )

                m_t = list(loop_final[0:VEC])
                kv_t = list(loop_final[VEC : 2 * VEC])
                w_t = list(loop_final[2 * VEC : 3 * VEC])
                t0_kv = list(loop_final[3 * VEC : 4 * VEC])
                t0_sc = list(loop_final[4 * VEC : 5 * VEC])
                t0_ape = list(loop_final[5 * VEC : 6 * VEC])
                t1_kv = list(loop_final[6 * VEC : 7 * VEC])
                t1_sc = list(loop_final[7 * VEC : 8 * VEC])
                t1_ape = list(loop_final[8 * VEC : 9 * VEC])

                tail0_score = [
                    arith.select(
                        is_p2,
                        arith.AddFOp(t0_sc[i], t0_ape[i], fastmath=fm_fast).result,
                        c_neg_inf,
                    )
                    for i in range(VEC)
                ]
                r0_m, r0_kv, r0_w = _softmax_step_padded(
                    m_t, kv_t, w_t, tail0_score, t0_kv
                )

                tail1_gate = arith.andi(is_p2_ge2, is_odd)
                tail1_score = [
                    arith.select(
                        tail1_gate,
                        arith.AddFOp(t1_sc[i], t1_ape[i], fastmath=fm_fast).result,
                        c_neg_inf,
                    )
                    for i in range(VEC)
                ]
                r1_m, r1_kv, r1_w = _softmax_step_padded(
                    r0_m, r0_kv, r0_w, tail1_score, t1_kv
                )
                final = list(r1_m) + list(r1_kv) + list(r1_w)

            m_local = list(final[0:VEC])
            kv_local = list(final[VEC : 2 * VEC])
            w_local = list(final[2 * VEC : 3 * VEC])

            # -- LDS write: each thread writes VEC entries per array --
            # Layout: per array, NW * SLICE_SZ fp32 entries; per-thread
            # base = wid * SLICE_SZ + lid * VEC; thread writes VEC values
            # at base+0, base+1, ..., base+VEC-1.
            lds_m_ptr = lds.lds_m.ptr
            lds_kv_ptr = lds.lds_kv.ptr
            lds_w_ptr = lds.lds_w.ptr
            lds_thread_base = wid * SLICE_SZ + lid * VEC
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + i
                fx.ptr_store(m_local[i], lds_m_ptr + idx_i)
                fx.ptr_store(kv_local[i], lds_kv_ptr + idx_i)
                fx.ptr_store(w_local[i], lds_w_ptr + idx_i)

            gpu.barrier()

            # -- Cross-wave reduction: only wave 0 reads and reduces --
            # Wave 0's 32 threads cover SLICE_SZ head_dim elements (VEC elements
            # per thread). For each owned element, the thread reads NW values
            # from LDS (one per K-split wave) and computes the global softmax.
            def _wave0():
                comp_list = []
                for i in range_constexpr(VEC):
                    lane_off = lid * VEC + i
                    # Global max across NW waves for this element.
                    m_g = fx.Float32(c_neg_inf)
                    m_arr = []
                    for w in range_constexpr(NW):
                        m_w = fx.ptr_load(lds_m_ptr + (lane_off + w * SLICE_SZ))
                        m_arr.append(m_w)
                        m_g = m_g.maximumf(m_w)

                    # Weighted sums (kv * scale_w) and (w * scale_w).
                    kv_sum = fx.Float32(0.0)
                    w_sum = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        idx_w = lane_off + w * SLICE_SZ
                        kv_w = fx.ptr_load(lds_kv_ptr + idx_w)
                        w_w = fx.ptr_load(lds_w_ptr + idx_w)
                        m_w = m_arr[w]
                        scale_w = fx.Float32(fexp_f32((m_w - m_g).ir_value()))
                        kv_sum = kv_sum + kv_w * scale_w
                        w_sum = w_sum + w_w * scale_w
                    rcp_w = fx.Float32(
                        llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [w_sum.ir_value()], [], []
                        )
                    )
                    comp_list.append(kv_sum * rcp_w)

                # -- Vectorized write of VEC f32 comp values --
                out_rsrc = buffer_ops.create_buffer_resource(
                    kv_compressed, max_size=True
                )
                out_off = (
                    fx.Int32(pid) * fx.Int32(kv_compressed_row_stride) + col_off_base
                )
                if const_expr(VEC == 1):
                    buffer_ops.buffer_store(comp_list[0].ir_value(), out_rsrc, out_off)
                elif const_expr(VEC <= 4):
                    out_vec = fx.Vector.from_elements(comp_list, dtype=fx.Float32)
                    buffer_ops.buffer_store(out_vec.ir_value(), out_rsrc, out_off)
                else:
                    # VEC > 4: AMD HW max is dwordx4 -> split into Nx dwordx4 stores.
                    quarter = 4
                    n_chunks = VEC // quarter
                    for q in range_constexpr(n_chunks):
                        base = q * quarter
                        sv = fx.Vector.from_elements(
                            comp_list[base : base + quarter], dtype=fx.Float32
                        )
                        buffer_ops.buffer_store(sv.ir_value(), out_rsrc, out_off + base)

            if wid == 0:
                _wave0()

        if fx.Int32(position) >= 0:
            _body()

    @flyc.jit
    def launch_hca_compress_forward(
        kv_in: fx.Tensor,
        kv_in_row_stride: fx.Int32,
        score_in: fx.Tensor,
        score_in_row_stride: fx.Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,
        kv_state_slot_stride: fx.Int32,
        kv_state_pos_stride: fx.Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: fx.Int32,
        score_state_pos_stride: fx.Int32,
        state_slot_mapping: fx.Tensor,
        ape: fx.Tensor,
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream,
    ):
        idx_p = fx.Int64(plan_capacity)
        idx_s = fx.Int64(NUM_SPLIT)
        k = kernel(
            kv_in,
            kv_in_row_stride,
            score_in,
            score_in_row_stride,
            plan,
            kv_state,
            kv_state_slot_stride,
            kv_state_pos_stride,
            score_state,
            score_state_slot_stride,
            score_state_pos_stride,
            state_slot_mapping,
            ape,
            kv_compressed,
            kv_compressed_row_stride,
        )
        k.launch(
            grid=(idx_p, idx_s, 1),
            block=(BLOCK_TH, 1, 1),
            stream=stream,
        )

    return launch_hca_compress_forward


# ============================================================================
# Kernel B: norm + rope + scatter (BF16, per-row)
# ============================================================================


def _build_norm_rope_scatter_kernel(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    k_per_block: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    quant: bool = False,
    quant_group_size: int = 64,
):
    """Build per-row RMSNorm + GPT-J RoPE + paged scatter for HCA (wave32).

    Reads kv_compressed[num_compress, head_dim] fp32 and the plan; for each
    boundary, normalizes / rotates / scatters into kv_cache.

    quant=False: BF16 single-buffer scatter (nope + rope in one kv_cache row).
    quant=True : FP8 nope (1xG e8m0 group-quant) + inline duplicated e8m0 scale into
                 kv_cache (V4 nm asm layout), rotated PE bf16 into a SEPARATE k_rope_buff
                 -- byte-identical to the C++ k_wave / fused_kv_compress_scatter output.
    """
    D = head_dim
    RD = rope_head_dim
    NOPE = D - RD
    VEC = D // BLOCK_THREADS  # 16 for D=512 (wave32)
    ROPE_THREAD_LO = NOPE // VEC
    PAIRS_PER_THREAD = VEC // 2

    assert D % BLOCK_THREADS == 0
    assert RD > 0 and RD % 2 == 0 and RD % VEC == 0

    # FP8 1xG e8m0 group-quant geometry (nope region only). GROUP_SIZE must divide
    # NOPE and be a multiple of VEC (a lane's VEC slice never crosses a group).
    GROUP_SIZE_Q = quant_group_size
    assert (not quant) or (
        NOPE % GROUP_SIZE_Q == 0 and GROUP_SIZE_Q % VEC == 0
    ), f"quant: NOPE={NOPE} must be divisible by group={GROUP_SIZE_Q}, group%VEC==0"
    assert (not quant) or VEC % 4 == 0, f"quant: VEC={VEC} must be a multiple of 4"
    RTS = GROUP_SIZE_Q // VEC if quant else 1  # threads per group (=4 for G=64,VEC=16)
    log2_rts = int(math.log2(RTS)) if quant else 0

    _kname = (
        f"hca_norm_rope_scatter_w32_D{D}_RD{RD}_R{ratio}_KB{k_per_block}"
        f"{'_rmsbf16' if rms_weight_is_bf16 else ''}{'_fp8' if quant else ''}_flydsl"
    )
    fm_fast = arith.FastMathFlags.fast
    log2_block = int(math.log2(BLOCK_THREADS))

    @flyc.kernel(name=_kname)
    def kernel(
        kv_compressed: fx.Tensor,  # [num_compress, head_dim] f32
        kv_compressed_row_stride: Int32,
        plan: fx.Tensor,  # [num_compress, 4] i32
        rms_weight: fx.Tensor,  # [head_dim] bf16 or f32
        cos_cache: fx.Tensor,  # [max_pos, RD/2] bf16
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,  # bf16: [NB,k_per_block,D]; fp8: [NB,k_per_block,entry] nope+scale
        kv_cache_block_stride: Int32,  # elements (bf16 or fp8/byte)
        kv_cache_token_stride: Int32,
        block_table: fx.Tensor,  # [bs, max_blocks_per_seq] i32
        block_table_seq_stride: Int32,
        k_rope_buff: fx.Tensor,  # fp8 only: paged [NB,k_per_block,RD] bf16 rope (dummy if !quant)
        krope_block_stride: Int32,
        krope_token_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32

        pid = fx.block_idx.x
        tid = fx.thread_idx.x

        c_eps = arith.constant(rms_eps, type=f32)
        c_inv_D = arith.constant(1.0 / D, type=f32)

        def wave_reduce_add(w):
            # w is a raw f32 ir.Value; keep the explicit-fastmath add (fx `+`
            # drops fastmath<fast> -> ISA drift on gfx1250).
            for sh_exp in range_constexpr(log2_block):
                off = BLOCK_THREADS // (2 << sh_exp)
                peer = fx.Float32(w).shuffle_xor(off, BLOCK_THREADS).ir_value()
                w = arith.AddFOp(w, peer, fastmath=fm_fast).result
            return w

        # -- Load plan row --
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_vec = fx.Vector(
            buffer_ops.buffer_load(plan_rsrc, fx.Int32(pid) * 4, vec_width=4, dtype=i32)
        )
        batch_id = plan_vec[1]
        position = plan_vec[2]

        # Sentinel-skip: run the whole body only for position >= 0, as a closure
        # under a runtime `if` (rewriter sees an opaque call -> scf.if).
        def _body():
            tid_x_vec = fx.Int32(tid) * VEC

            # -- Load kv_compressed[pid, tid*VEC : tid*VEC + VEC] --
            kvc_rsrc = buffer_ops.create_buffer_resource(kv_compressed, max_size=True)
            base_off = fx.Int32(pid) * fx.Int32(kv_compressed_row_stride) + tid_x_vec
            # VEC ? {2, 4, 8, 16}: VEC <= 4 -> single dwordx{VEC}; VEC>4 -> Nx dwordx4.
            # comp_lane held as raw f32 ir.Values for the explicit-fastmath layer.
            if const_expr(VEC <= 4):
                raw = fx.Vector(
                    buffer_ops.buffer_load(kvc_rsrc, base_off, vec_width=VEC, dtype=f32)
                )
                comp_lane = [raw[i].ir_value() for i in range(VEC)]
            else:
                quarter = 4
                n_chunks = VEC // quarter
                comp_lane = []
                for q in range_constexpr(n_chunks):
                    r = fx.Vector(
                        buffer_ops.buffer_load(
                            kvc_rsrc,
                            base_off + q * quarter,
                            vec_width=quarter,
                            dtype=f32,
                        )
                    )
                    comp_lane += [r[i].ir_value() for i in range_constexpr(quarter)]

            # -- RMSNorm (wave reduce-add of squares / D + eps; rsqrt) --
            sq_local = arith.constant(0.0, type=f32)
            for i in range_constexpr(VEC):
                sq_local = arith.AddFOp(
                    sq_local,
                    arith.MulFOp(comp_lane[i], comp_lane[i], fastmath=fm_fast).result,
                    fastmath=fm_fast,
                ).result
            sq_full = wave_reduce_add(sq_local)
            var = arith.MulFOp(sq_full, c_inv_D, fastmath=fm_fast).result
            rrms = fmath.rsqrt(
                arith.AddFOp(var, c_eps, fastmath=fm_fast).result, fastmath=fm_fast
            )

            # rms_weight load
            rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
            if const_expr(rms_weight_is_bf16):
                dwords = (VEC + 1) // 2
                # logical shift (tid_x_vec >= 0); fx Int32 >> is arithmetic.
                off_dw = fx.Int32((fx.Uint32(tid_x_vec.ir_value()) >> 1).ir_value())
                if const_expr(dwords == 1):
                    raw_s = buffer_ops.buffer_load(
                        rmsw_rsrc, off_dw, vec_width=1, dtype=i32
                    )
                    raw = fx.Vector.from_elements([raw_s], dtype=fx.Int32)
                    vec_bf16 = raw.bitcast(fx.BFloat16)
                    rmsw_lane = [
                        vec_bf16[i].to(fx.Float32).ir_value()
                        for i in range_constexpr(VEC)
                    ]
                elif const_expr(dwords <= 4):
                    raw = fx.Vector(
                        buffer_ops.buffer_load(
                            rmsw_rsrc, off_dw, vec_width=dwords, dtype=i32
                        )
                    )
                    vec_bf16 = raw.bitcast(fx.BFloat16)
                    rmsw_lane = [
                        vec_bf16[i].to(fx.Float32).ir_value()
                        for i in range_constexpr(VEC)
                    ]
                else:
                    # dwords > 4 (VEC=16 -> dwords=8): split into 2x dwordx4
                    half_dw = 4
                    half_bf16 = half_dw * 2
                    rmsw_lane = []
                    for chunk in range_constexpr(dwords // half_dw):
                        r = buffer_ops.buffer_load(
                            rmsw_rsrc,
                            off_dw + chunk * half_dw,
                            vec_width=half_dw,
                            dtype=i32,
                        )
                        vbf16 = fx.Vector(r).bitcast(fx.BFloat16)
                        rmsw_lane += [
                            vbf16[i].to(fx.Float32).ir_value()
                            for i in range_constexpr(half_bf16)
                        ]
            else:
                if const_expr(VEC <= 4):
                    raw = fx.Vector(
                        buffer_ops.buffer_load(
                            rmsw_rsrc, tid_x_vec, vec_width=VEC, dtype=f32
                        )
                    )
                    rmsw_lane = [raw[i].ir_value() for i in range(VEC)]
                else:
                    quarter = 4
                    n_chunks = VEC // quarter
                    rmsw_lane = []
                    for q in range_constexpr(n_chunks):
                        r = fx.Vector(
                            buffer_ops.buffer_load(
                                rmsw_rsrc,
                                tid_x_vec + q * quarter,
                                vec_width=quarter,
                                dtype=f32,
                            )
                        )
                        rmsw_lane += [r[i].ir_value() for i in range_constexpr(quarter)]

            normed_lane = [
                arith.MulFOp(
                    arith.MulFOp(comp_lane[i], rrms, fastmath=fm_fast).result,
                    rmsw_lane[i],
                    fastmath=fm_fast,
                ).result
                for i in range(VEC)
            ]

            # -- GPT-J RoPE on RD tail -- (position >= 0 -> unsigned div)
            comp_pos_i32 = fx.Int32((fx.Uint32(position) // ratio).ir_value()) * ratio
            cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
            sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
            cos_row_base = comp_pos_i32 * (RD // 2)

            is_rope_t = arith.cmpi(
                CmpIPredicate.sge,
                tid.ir_value(),
                arith.constant(ROPE_THREAD_LO, type=i32),
            )
            rope_rel_raw = fx.Int32(tid) - ROPE_THREAD_LO
            rope_rel = fx.max(rope_rel_raw, fx.Int32(0))
            cs_lo = rope_rel * PAIRS_PER_THREAD

            if const_expr(PAIRS_PER_THREAD == 1):
                cos_b = buffer_ops.buffer_load(
                    cos_rsrc, cos_row_base + cs_lo, vec_width=1, dtype=T.bf16
                )
                sin_b = buffer_ops.buffer_load(
                    sin_rsrc, cos_row_base + cs_lo, vec_width=1, dtype=T.bf16
                )
                cos_vals = [fx.BFloat16(cos_b).to(fx.Float32).ir_value()]
                sin_vals = [fx.BFloat16(sin_b).to(fx.Float32).ir_value()]
            else:
                cos_vec = fx.Vector(
                    buffer_ops.buffer_load(
                        cos_rsrc,
                        cos_row_base + cs_lo,
                        vec_width=PAIRS_PER_THREAD,
                        dtype=T.bf16,
                    )
                )
                sin_vec = fx.Vector(
                    buffer_ops.buffer_load(
                        sin_rsrc,
                        cos_row_base + cs_lo,
                        vec_width=PAIRS_PER_THREAD,
                        dtype=T.bf16,
                    )
                )
                cos_vals = [
                    cos_vec[i].to(fx.Float32).ir_value()
                    for i in range(PAIRS_PER_THREAD)
                ]
                sin_vals = [
                    sin_vec[i].to(fx.Float32).ir_value()
                    for i in range(PAIRS_PER_THREAD)
                ]

            rotated_lane = list(normed_lane)
            for k in range_constexpr(PAIRS_PER_THREAD):
                e = normed_lane[2 * k]
                o = normed_lane[2 * k + 1]
                c = cos_vals[k]
                s = sin_vals[k]
                # NOTE: real part uses a non-fastmath subtract (default flags);
                # explicit-fastmath MulFOp/AddFOp for the rest (fx ops drop
                # fastmath<fast> on gfx1250 -> ISA drift).
                new_e = arith.subf(
                    arith.MulFOp(e, c, fastmath=fm_fast).result,
                    arith.MulFOp(o, s, fastmath=fm_fast).result,
                )
                new_o = arith.AddFOp(
                    arith.MulFOp(e, s, fastmath=fm_fast).result,
                    arith.MulFOp(o, c, fastmath=fm_fast).result,
                    fastmath=fm_fast,
                ).result
                rotated_lane[2 * k] = new_e
                rotated_lane[2 * k + 1] = new_o

            # -- Paged scatter dest (shared by bf16 / fp8) --
            # position >= 0 (active guard) -> unsigned div/rem (divui/remui).
            ci = fx.Int32((fx.Uint32(position) // ratio).ir_value())
            block_in_seq = fx.Int32(
                (fx.Uint32(ci.ir_value()) // k_per_block).ir_value()
            )
            slot_in_block = fx.Int32(
                (fx.Uint32(ci.ir_value()) % k_per_block).ir_value()
            )
            bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
            bt_off = (
                fx.Int32(batch_id) * fx.Int32(block_table_seq_stride) + block_in_seq
            )
            physical_block = buffer_ops.buffer_load(
                bt_rsrc, bt_off, vec_width=1, dtype=i32
            )
            # The block term rides on the descriptor's base, not on the
            # 32-bit offset -- see `block_base_bytes_i64`.
            cache_base = slot_in_block * fx.Int32(kv_cache_token_stride)
            out_rsrc = buffer_ops.create_buffer_resource(
                kv_cache,
                max_size=True,
                base_byte_offset=block_base_bytes_i64(
                    physical_block, kv_cache_block_stride, 1 if quant else 2
                ),
            )

            if const_expr(quant):
                # -- group_fp8 (V4 nm-asm) via shared emitter (wave32; same layout
                # as wave64 CSA/HCA -- single source of truth). The emitter lives
                # in _common and consumes raw ir.Values. --
                _krope_base = slot_in_block * fx.Int32(krope_token_stride)
                emit_group_fp8_nm_asm_scatter(
                    normed_lane=normed_lane,
                    rotated_lane=rotated_lane,
                    lane=tid,
                    is_rope_t=is_rope_t,
                    cache_base=cache_base.ir_value(),
                    out_base_i64=fx.Int64(fx.ptrtoint(fx.get_iter(kv_cache)))
                    + fx.Int64(
                        block_base_bytes_i64(physical_block, kv_cache_block_stride, 1)
                    ),
                    krope_base=_krope_base.ir_value(),
                    krope_base_i64=fx.Int64(fx.ptrtoint(fx.get_iter(k_rope_buff)))
                    + fx.Int64(
                        block_base_bytes_i64(physical_block, krope_block_stride, 2)
                    ),
                    VEC=VEC,
                    NOPE=NOPE,
                    RTS=RTS,
                    log2_rts=log2_rts,
                    ROPE_THREAD_LO=ROPE_THREAD_LO,
                    wave_width=BLOCK_THREADS,
                )
            else:
                # ---- BF16 single-buffer scatter (nope + rope contiguous) ----
                out_lane = [
                    arith.select(is_rope_t, rotated_lane[i], normed_lane[i])
                    for i in range_constexpr(VEC)
                ]
                cache_off = cache_base + tid_x_vec
                out_vec_t = T.vec(VEC, T.bf16)
                raw_vec = fx.Vector.from_elements(out_lane, dtype=fx.Float32)
                bf16_vec = raw_vec.truncf(out_vec_t)
                # logical shift (cache_off >= 0); fx Int32 >> is arithmetic.
                cache_off_dw = fx.Int32(
                    (fx.Uint32(cache_off.ir_value()) >> 1).ir_value()
                )
                dwords = (VEC + 1) // 2
                bf16_as_i32 = bf16_vec.bitcast(fx.Int32)
                if const_expr(dwords == 1):
                    buffer_ops.buffer_store(
                        bf16_as_i32[0].ir_value(), out_rsrc, cache_off_dw
                    )
                elif const_expr(dwords <= 4):
                    buffer_ops.buffer_store(
                        bf16_as_i32.ir_value(), out_rsrc, cache_off_dw
                    )
                else:
                    # dwords > 4 (VEC=16 -> dwords=8): split into 2x dwordx4.
                    lo = fx.Vector.from_elements(
                        [bf16_as_i32[i] for i in range(4)], dtype=fx.Int32
                    )
                    hi = fx.Vector.from_elements(
                        [bf16_as_i32[i] for i in range(4, 8)], dtype=fx.Int32
                    )
                    buffer_ops.buffer_store(lo.ir_value(), out_rsrc, cache_off_dw)
                    buffer_ops.buffer_store(hi.ir_value(), out_rsrc, cache_off_dw + 4)

        if fx.Int32(position) >= 0:
            _body()

    @flyc.jit
    def launch_hca_norm_rope_scatter(
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: fx.Int32,
        plan: fx.Tensor,
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: fx.Int32,
        kv_cache_token_stride: fx.Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: fx.Int32,
        k_rope_buff: fx.Tensor,
        krope_block_stride: fx.Int32,
        krope_token_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream,
    ):
        idx_p = fx.Int64(plan_capacity)
        k = kernel(
            kv_compressed,
            kv_compressed_row_stride,
            plan,
            rms_weight,
            cos_cache,
            sin_cache,
            kv_cache,
            kv_cache_block_stride,
            kv_cache_token_stride,
            block_table,
            block_table_seq_stride,
            k_rope_buff,
            krope_block_stride,
            krope_token_stride,
        )
        k.launch(
            grid=(idx_p, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_hca_norm_rope_scatter


# ============================================================================
# Kernel C: fused compress + norm + rope + scatter (single launch, SL=512)
# ============================================================================


def _build_fused_compress_kernel(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    k_per_block: int = 64,
    rms_weight_is_bf16: bool = False,
    rms_eps: float = 1e-6,
):
    """Fused single-launch HCA kernel: K-split pool + softmax + RMSNorm + RoPE +
    BF16 scatter to paged cache.

    Combines Kernel A (compress_forward) and Kernel B (norm_rope_scatter) into
    one kernel launch by hardcoding slice_size=512 so NUM_SPLIT=1 (one block
    per boundary). After the LDS cross-wave reduction, wave 0 holds the full
    D=512 vector and can inline the norm+rope+scatter tail without cross-block
    synchronization, eliminating one kernel launch and the fp32 scratch buffer.

    BF16 non-quant path only. Falls back to the 2-kernel path for quant.
    """
    SLICE_SZ = 512
    D = head_dim
    RD = rope_head_dim
    K = ratio
    DIM_FULL = D
    VEC = SLICE_SZ // BLOCK_THREADS
    NW = k_split_num_waves
    BLOCK_TH = BLOCK_THREADS * NW
    K_PER_WAVE = K // NW
    NOPE = D - RD
    ROPE_THREAD_LO = NOPE // VEC
    PAIRS_PER_THREAD = VEC // 2
    log2_block = int(math.log2(BLOCK_THREADS))

    assert D % SLICE_SZ == 0
    assert SLICE_SZ == D, "fused kernel requires SL=D (one block per boundary)"
    assert K % NW == 0, f"K={K} must divide evenly across {NW} waves"
    assert state_size >= K
    assert NOPE % VEC == 0, f"NOPE={NOPE} must be divisible by VEC={VEC}"

    LDS_M_ELEMS = NW * SLICE_SZ
    LDS_KV_ELEMS = NW * SLICE_SZ
    LDS_W_ELEMS = NW * SLICE_SZ

    @fx.struct
    class SharedStorage:
        lds_m: fx.Array[fx.Float32, LDS_M_ELEMS, 16]
        lds_kv: fx.Array[fx.Float32, LDS_KV_ELEMS, 16]
        lds_w: fx.Array[fx.Float32, LDS_W_ELEMS, 16]

    _kname = f"hca_compress_fused_w32_D{D}_R{K}_NW{NW}_S{state_size}_flydsl"
    fm_fast = arith.FastMathFlags.fast

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_TH, 1, 1])
    def kernel(
        kv_in: fx.Tensor,
        kv_in_row_stride: Int32,
        score_in: fx.Tensor,
        score_in_row_stride: Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,
        kv_state_slot_stride: Int32,
        kv_state_pos_stride: Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: Int32,
        score_state_pos_stride: Int32,
        state_slot_mapping: fx.Tensor,
        ape: fx.Tensor,
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: Int32,
        kv_cache_token_stride: Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32

        pid = fx.block_idx.x
        tid = fx.thread_idx.x

        c_zero_i32 = arith.constant(0, type=i32)
        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        c_eps = arith.constant(rms_eps, type=f32)
        c_inv_D = arith.constant(1.0 / D, type=f32)

        def fexp_f32(x):
            return llvm.call_intrinsic(
                f32, "llvm.amdgcn.exp2.f32", [x * c_log2e], [], []
            )

        def wave_reduce_add(w):
            for sh_exp in range_constexpr(log2_block):
                off = BLOCK_THREADS // (2 << sh_exp)
                peer = fx.Float32(w).shuffle_xor(off, BLOCK_THREADS).ir_value()
                w = arith.AddFOp(w, peer, fastmath=fm_fast).result
            return w

        wid = fx.Int32((fx.Uint32(tid) // BLOCK_THREADS).ir_value())
        lid = fx.Int32((fx.Uint32(tid) % BLOCK_THREADS).ir_value())

        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_vec = fx.Vector(
            buffer_ops.buffer_load(plan_rsrc, fx.Int32(pid) * 4, vec_width=4, dtype=i32)
        )
        ragged_id = plan_vec[0]
        batch_id = plan_vec[1]
        position = plan_vec[2]
        window_len = plan_vec[3]

        def _body():
            col_off_base = lid * VEC

            slot_map_rsrc = buffer_ops.create_buffer_resource(
                state_slot_mapping, max_size=True
            )
            slot = buffer_ops.buffer_load(
                slot_map_rsrc, batch_id, vec_width=1, dtype=i32
            )

            kv_in_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            score_in_rsrc = buffer_ops.create_buffer_resource(score_in, max_size=True)
            kv_state_rsrc = buffer_ops.create_buffer_resource(
                kv_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, kv_state_slot_stride),
            )
            score_state_rsrc = buffer_ops.create_buffer_resource(
                score_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, score_state_slot_stride),
            )
            ape_rsrc = buffer_ops.create_buffer_resource(ape, max_size=True)

            # -- Load helpers (VEC=16) --

            def _load_bf16_vec_to_f32(rsrc, base_off_elems_i32):
                base_off = fx.Int32(base_off_elems_i32)
                off_dw = fx.Int32((fx.Uint32(base_off_elems_i32) >> 1).ir_value())
                dwords = VEC // 2
                if const_expr(dwords <= 4):
                    raw = fx.Vector(
                        buffer_ops.buffer_load(
                            rsrc, off_dw, vec_width=dwords, dtype=i32
                        )
                    )
                else:
                    half_dw = dwords // 2
                    r0 = fx.Vector(
                        buffer_ops.buffer_load(
                            rsrc, off_dw, vec_width=half_dw, dtype=i32
                        )
                    )
                    r1 = fx.Vector(
                        buffer_ops.buffer_load(
                            rsrc, off_dw + half_dw, vec_width=half_dw, dtype=i32
                        )
                    )
                    raw = fx.Vector.from_elements(
                        [r0[i] for i in range(half_dw)]
                        + [r1[i] for i in range(half_dw)],
                        dtype=fx.Int32,
                    )
                vec_bf16 = raw.bitcast(fx.BFloat16)
                return [vec_bf16[i].to(fx.Float32).ir_value() for i in range(VEC)]

            def _load_f32_vec(rsrc, base_off_elems_i32):
                quarter = 4
                n_chunks = VEC // quarter
                result = []
                for q in range_constexpr(n_chunks):
                    r = fx.Vector(
                        buffer_ops.buffer_load(
                            rsrc,
                            fx.Int32(base_off_elems_i32) + q * quarter,
                            vec_width=quarter,
                            dtype=f32,
                        )
                    )
                    result.extend(r[j].ir_value() for j in range(quarter))
                return result

            def _issue_phase2_loads(k_i32):
                k = fx.Int32(k_i32)
                ape_row = fx.Int32((fx.Uint32(k_i32) % ratio).ir_value())
                in_row_raw = fx.Int32(ragged_id) - (fx.Int32(K - 1) - k)
                in_row = fx.max(in_row_raw, fx.Int32(0))
                base_in_off = in_row * fx.Int32(kv_in_row_stride) + col_off_base
                base_sc_off = in_row * fx.Int32(score_in_row_stride) + col_off_base
                base_ape_off = ape_row * DIM_FULL + col_off_base
                kv = _load_bf16_vec_to_f32(kv_in_rsrc, base_in_off)
                sc = _load_bf16_vec_to_f32(score_in_rsrc, base_sc_off)
                ape_v = _load_f32_vec(ape_rsrc, base_ape_off)
                return kv, sc, ape_v

            def _issue_phase1_loads(k_i32):
                s = (fx.Int32(position) - fx.Int32(K - 1) + fx.Int32(k_i32)).ir_value()
                is_pad = arith.cmpi(CmpIPredicate.slt, s, c_zero_i32)
                s_safe = fx.Int32(arith.select(is_pad, c_zero_i32, s))
                ring = fx.Int32((fx.Uint32(s_safe.ir_value()) % state_size).ir_value())
                base_kv_off = ring * fx.Int32(kv_state_pos_stride) + col_off_base
                base_sc_off = ring * fx.Int32(score_state_pos_stride) + col_off_base
                kv_list = _load_f32_vec(kv_state_rsrc, base_kv_off)
                sc_list = _load_f32_vec(score_state_rsrc, base_sc_off)
                sc_padded = [
                    arith.select(is_pad, c_neg_inf, sc_list[i]) for i in range(VEC)
                ]
                return kv_list, sc_padded

            def _softmax_step_padded(
                m_old_list, kv_old_list, w_old_list, score_k_list, kv_k_list
            ):
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = m_old_list[i]
                    kv_old = kv_old_list[i]
                    w_old = w_old_list[i]
                    score_k = score_k_list[i]
                    kv_k = kv_k_list[i]
                    m_new = fx.max(fx.Float32(m_old), fx.Float32(score_k)).ir_value()
                    is_first = arith.cmpf(CmpFPredicate.OEQ, m_old, c_neg_inf)
                    scale_active = fexp_f32(arith.subf(m_old, m_new))
                    scale_v = arith.select(is_first, c_zero_f32, scale_active)
                    wk_active = fexp_f32(arith.subf(score_k, m_new))
                    is_pad_score = arith.cmpf(CmpFPredicate.OEQ, score_k, c_neg_inf)
                    w_k = arith.select(is_pad_score, c_zero_f32, wk_active)
                    new_kv.append(
                        arith.AddFOp(
                            arith.MulFOp(kv_old, scale_v, fastmath=fm_fast).result,
                            arith.MulFOp(w_k, kv_k, fastmath=fm_fast).result,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_w.append(
                        arith.AddFOp(
                            arith.MulFOp(w_old, scale_v, fastmath=fm_fast).result,
                            w_k,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_m.append(m_new)
                return new_m, new_kv, new_w

            # -- K-split pool + online softmax (identical to Kernel A) --
            k_start_i32 = wid * K_PER_WAVE
            k_end_i32 = k_start_i32 + K_PER_WAVE
            split_i32 = fx.min(fx.max(window_len, k_start_i32), k_end_i32)

            init_m = [c_neg_inf for _ in range(VEC)]
            init_kv = [c_zero_f32 for _ in range(VEC)]
            init_w = [c_zero_f32 for _ in range(VEC)]
            init_state = init_m + init_kv + init_w

            phase1_local = init_state
            for k_static, state in range(
                k_start_i32.ir_value(), split_i32.ir_value(), 1, init=init_state
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = fx.Int32(k_static)
                kv_v, sc_v = _issue_phase1_loads(k_i32)
                new_m, new_kv, new_w = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, sc_v, kv_v
                )
                phase1_local = yield list(new_m) + list(new_kv) + list(new_w)

            final = phase1_local
            for k_static, state in range(
                split_i32.ir_value(), k_end_i32.ir_value(), 1, init=phase1_local
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = fx.Int32(k_static)
                p2_kv, p2_sc, p2_ape = _issue_phase2_loads(k_i32)
                p2_score = [
                    arith.AddFOp(p2_sc[i], p2_ape[i], fastmath=fm_fast).result
                    for i in range(VEC)
                ]
                new_m, new_kv, new_w = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, p2_score, p2_kv
                )
                final = yield list(new_m) + list(new_kv) + list(new_w)

            m_local = list(final[0:VEC])
            kv_local = list(final[VEC : 2 * VEC])
            w_local = list(final[2 * VEC : 3 * VEC])

            # -- LDS write + barrier --
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            lds_m_ptr = lds.lds_m.ptr
            lds_kv_ptr = lds.lds_kv.ptr
            lds_w_ptr = lds.lds_w.ptr
            lds_thread_base = wid * SLICE_SZ + lid * VEC
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + i
                fx.ptr_store(m_local[i], lds_m_ptr + idx_i)
                fx.ptr_store(kv_local[i], lds_kv_ptr + idx_i)
                fx.ptr_store(w_local[i], lds_w_ptr + idx_i)

            gpu.barrier()

            # -- Wave 0: cross-wave reduce + norm + rope + scatter --
            def _wave0():
                comp_list = []
                for i in range_constexpr(VEC):
                    lane_off = lid * VEC + i
                    m_g = fx.Float32(c_neg_inf)
                    m_arr = []
                    for w in range_constexpr(NW):
                        m_w = fx.ptr_load(lds_m_ptr + (lane_off + w * SLICE_SZ))
                        m_arr.append(m_w)
                        m_g = m_g.maximumf(m_w)

                    kv_sum = fx.Float32(0.0)
                    w_sum = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        idx_w = lane_off + w * SLICE_SZ
                        kv_w = fx.ptr_load(lds_kv_ptr + idx_w)
                        w_w = fx.ptr_load(lds_w_ptr + idx_w)
                        m_w = m_arr[w]
                        scale_w = fx.Float32(fexp_f32((m_w - m_g).ir_value()))
                        kv_sum = kv_sum + kv_w * scale_w
                        w_sum = w_sum + w_w * scale_w
                    rcp_w = fx.Float32(
                        llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [w_sum.ir_value()], [], []
                        )
                    )
                    comp_list.append(kv_sum * rcp_w)

                # ---- RMSNorm ----
                sq_local = arith.constant(0.0, type=f32)
                for i in range_constexpr(VEC):
                    sq_local = arith.AddFOp(
                        sq_local,
                        arith.MulFOp(
                            comp_list[i].ir_value(),
                            comp_list[i].ir_value(),
                            fastmath=fm_fast,
                        ).result,
                        fastmath=fm_fast,
                    ).result
                sq_full = wave_reduce_add(sq_local)
                var = arith.MulFOp(sq_full, c_inv_D, fastmath=fm_fast).result
                rrms = fmath.rsqrt(
                    arith.AddFOp(var, c_eps, fastmath=fm_fast).result,
                    fastmath=fm_fast,
                )

                # rms_weight load (VEC=16 elements at lid*VEC)
                rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
                tid_x_vec = lid * VEC
                if const_expr(rms_weight_is_bf16):
                    dwords = (VEC + 1) // 2
                    off_dw = fx.Int32((fx.Uint32(tid_x_vec.ir_value()) >> 1).ir_value())
                    if const_expr(dwords <= 4):
                        raw = fx.Vector(
                            buffer_ops.buffer_load(
                                rmsw_rsrc, off_dw, vec_width=dwords, dtype=i32
                            )
                        )
                        vec_bf16 = raw.bitcast(fx.BFloat16)
                        rmsw_lane = [
                            vec_bf16[i].to(fx.Float32).ir_value()
                            for i in range_constexpr(VEC)
                        ]
                    else:
                        half_dw = 4
                        half_bf16 = half_dw * 2
                        rmsw_lane = []
                        for chunk in range_constexpr(dwords // half_dw):
                            r = buffer_ops.buffer_load(
                                rmsw_rsrc,
                                off_dw + chunk * half_dw,
                                vec_width=half_dw,
                                dtype=i32,
                            )
                            vbf16 = fx.Vector(r).bitcast(fx.BFloat16)
                            rmsw_lane += [
                                vbf16[i].to(fx.Float32).ir_value()
                                for i in range_constexpr(half_bf16)
                            ]
                else:
                    quarter = 4
                    n_chunks = VEC // quarter
                    rmsw_lane = []
                    for q in range_constexpr(n_chunks):
                        r = fx.Vector(
                            buffer_ops.buffer_load(
                                rmsw_rsrc,
                                tid_x_vec + q * quarter,
                                vec_width=quarter,
                                dtype=f32,
                            )
                        )
                        rmsw_lane += [r[i].ir_value() for i in range_constexpr(quarter)]

                normed_lane = [
                    arith.MulFOp(
                        arith.MulFOp(
                            comp_list[i].ir_value(), rrms, fastmath=fm_fast
                        ).result,
                        rmsw_lane[i],
                        fastmath=fm_fast,
                    ).result
                    for i in range(VEC)
                ]

                # ---- GPT-J RoPE on last RD dims ----
                comp_pos_i32 = (
                    fx.Int32((fx.Uint32(position) // ratio).ir_value()) * ratio
                )
                cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
                sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
                cos_row_base = comp_pos_i32 * (RD // 2)

                is_rope_t = arith.cmpi(
                    CmpIPredicate.sge,
                    lid.ir_value(),
                    arith.constant(ROPE_THREAD_LO, type=i32),
                )
                rope_rel_raw = lid - ROPE_THREAD_LO
                rope_rel = fx.max(rope_rel_raw, fx.Int32(0))
                cs_lo = rope_rel * PAIRS_PER_THREAD

                if const_expr(PAIRS_PER_THREAD == 1):
                    cos_b = buffer_ops.buffer_load(
                        cos_rsrc,
                        cos_row_base + cs_lo,
                        vec_width=1,
                        dtype=T.bf16,
                    )
                    sin_b = buffer_ops.buffer_load(
                        sin_rsrc,
                        cos_row_base + cs_lo,
                        vec_width=1,
                        dtype=T.bf16,
                    )
                    cos_vals = [fx.BFloat16(cos_b).to(fx.Float32).ir_value()]
                    sin_vals = [fx.BFloat16(sin_b).to(fx.Float32).ir_value()]
                else:
                    cos_vec = fx.Vector(
                        buffer_ops.buffer_load(
                            cos_rsrc,
                            cos_row_base + cs_lo,
                            vec_width=PAIRS_PER_THREAD,
                            dtype=T.bf16,
                        )
                    )
                    sin_vec = fx.Vector(
                        buffer_ops.buffer_load(
                            sin_rsrc,
                            cos_row_base + cs_lo,
                            vec_width=PAIRS_PER_THREAD,
                            dtype=T.bf16,
                        )
                    )
                    cos_vals = [
                        cos_vec[i].to(fx.Float32).ir_value()
                        for i in range(PAIRS_PER_THREAD)
                    ]
                    sin_vals = [
                        sin_vec[i].to(fx.Float32).ir_value()
                        for i in range(PAIRS_PER_THREAD)
                    ]

                rotated_lane = list(normed_lane)
                for k in range_constexpr(PAIRS_PER_THREAD):
                    e = normed_lane[2 * k]
                    o = normed_lane[2 * k + 1]
                    c = cos_vals[k]
                    s = sin_vals[k]
                    new_e = arith.subf(
                        arith.MulFOp(e, c, fastmath=fm_fast).result,
                        arith.MulFOp(o, s, fastmath=fm_fast).result,
                    )
                    new_o = arith.AddFOp(
                        arith.MulFOp(e, s, fastmath=fm_fast).result,
                        arith.MulFOp(o, c, fastmath=fm_fast).result,
                        fastmath=fm_fast,
                    ).result
                    rotated_lane[2 * k] = new_e
                    rotated_lane[2 * k + 1] = new_o

                # ---- BF16 paged scatter ----
                ci = fx.Int32((fx.Uint32(position) // ratio).ir_value())
                block_in_seq = fx.Int32(
                    (fx.Uint32(ci.ir_value()) // k_per_block).ir_value()
                )
                slot_in_block = fx.Int32(
                    (fx.Uint32(ci.ir_value()) % k_per_block).ir_value()
                )
                bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
                bt_off = (
                    fx.Int32(batch_id) * fx.Int32(block_table_seq_stride) + block_in_seq
                )
                physical_block = buffer_ops.buffer_load(
                    bt_rsrc, bt_off, vec_width=1, dtype=i32
                )
                cache_base = slot_in_block * fx.Int32(kv_cache_token_stride)
                out_rsrc = buffer_ops.create_buffer_resource(
                    kv_cache,
                    max_size=True,
                    base_byte_offset=block_base_bytes_i64(
                        physical_block, kv_cache_block_stride, 2
                    ),
                )
                out_lane = [
                    arith.select(is_rope_t, rotated_lane[i], normed_lane[i])
                    for i in range_constexpr(VEC)
                ]
                cache_off = cache_base + tid_x_vec
                out_vec_t = T.vec(VEC, T.bf16)
                raw_vec = fx.Vector.from_elements(out_lane, dtype=fx.Float32)
                bf16_vec = raw_vec.truncf(out_vec_t)
                cache_off_dw = fx.Int32(
                    (fx.Uint32(cache_off.ir_value()) >> 1).ir_value()
                )
                bf16_as_i32 = bf16_vec.bitcast(fx.Int32)
                dwords = (VEC + 1) // 2
                if const_expr(dwords <= 4):
                    buffer_ops.buffer_store(
                        bf16_as_i32.ir_value(), out_rsrc, cache_off_dw
                    )
                else:
                    lo = fx.Vector.from_elements(
                        [bf16_as_i32[i] for i in range(4)], dtype=fx.Int32
                    )
                    hi = fx.Vector.from_elements(
                        [bf16_as_i32[i] for i in range(4, 8)], dtype=fx.Int32
                    )
                    buffer_ops.buffer_store(lo.ir_value(), out_rsrc, cache_off_dw)
                    buffer_ops.buffer_store(hi.ir_value(), out_rsrc, cache_off_dw + 4)

            if wid == 0:
                _wave0()

        if fx.Int32(position) >= 0:
            _body()

    @flyc.jit
    def launch_hca_compress_fused(
        kv_in: fx.Tensor,
        kv_in_row_stride: fx.Int32,
        score_in: fx.Tensor,
        score_in_row_stride: fx.Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,
        kv_state_slot_stride: fx.Int32,
        kv_state_pos_stride: fx.Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: fx.Int32,
        score_state_pos_stride: fx.Int32,
        state_slot_mapping: fx.Tensor,
        ape: fx.Tensor,
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: fx.Int32,
        kv_cache_token_stride: fx.Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream,
    ):
        idx_p = fx.Int64(plan_capacity)
        k = kernel(
            kv_in,
            kv_in_row_stride,
            score_in,
            score_in_row_stride,
            plan,
            kv_state,
            kv_state_slot_stride,
            kv_state_pos_stride,
            score_state,
            score_state_slot_stride,
            score_state_pos_stride,
            state_slot_mapping,
            ape,
            rms_weight,
            cos_cache,
            sin_cache,
            kv_cache,
            kv_cache_block_stride,
            kv_cache_token_stride,
            block_table,
            block_table_seq_stride,
        )
        k.launch(
            grid=(idx_p, 1, 1),
            block=(BLOCK_TH, 1, 1),
            stream=stream,
        )

    return launch_hca_compress_fused


_FUSED_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


@lru_cache(maxsize=16)
def compile_hca_fused_compress_gfx1250(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    k_per_block: int = 64,
    rms_weight_is_bf16: bool = False,
    rms_eps: float = 1e-6,
):
    """Compile the fused single-launch HCA kernel (pool+norm+rope+scatter)."""
    launcher = _build_fused_compress_kernel(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        state_size=state_size,
        k_split_num_waves=k_split_num_waves,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
    )
    launcher.compile_hints = dict(_FUSED_COMPILE_HINTS)
    return launcher


# ============================================================================
# Kernel D: atomic-fused compress + norm + rope + scatter (SL=128, single launch)
# ============================================================================


def _build_atomic_fused_compress_kernel(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    slice_size: int = 128,
    k_per_block: int = 64,
    rms_weight_is_bf16: bool = False,
    rms_eps: float = 1e-6,
    enable_prefetch_input: bool = True,
):
    """Atomic-fused single-launch HCA kernel: SL=128 pool+softmax + atomic tile
    counter + norm+rope+scatter tail.

    Unlike Kernel C (SL=512, poor CU utilization), this kernel keeps SL=128
    (optimal occupancy) and uses an atomic completion counter per boundary.
    After each block writes its SL=128 slice to the scratch buffer, it atomically
    increments a per-boundary counter.  The last block to arrive (counter ==
    NUM_SPLIT - 1) runs the norm+rope+scatter tail, reading the full D=512 vector
    from scratch.

    BF16 non-quant path only.
    """
    assert head_dim % slice_size == 0
    assert slice_size % 32 == 0
    assert ratio % k_split_num_waves == 0
    assert state_size >= ratio

    D = head_dim
    K = ratio
    DIM_FULL = D
    SLICE_SZ = slice_size
    VEC = SLICE_SZ // BLOCK_THREADS
    NUM_SPLIT = D // SLICE_SZ
    NW = k_split_num_waves
    BLOCK_TH = BLOCK_THREADS * NW
    K_PER_WAVE = K // NW

    # Tail constants (full-D norm+rope+scatter, wave 0 only = 32 threads)
    RD = rope_head_dim
    NOPE = D - RD
    VEC_FULL = D // BLOCK_THREADS  # 16 for D=512
    ROPE_THREAD_LO_FULL = NOPE // VEC_FULL
    PAIRS_PER_THREAD_FULL = VEC_FULL // 2
    log2_block = int(math.log2(BLOCK_THREADS))

    assert NOPE % VEC_FULL == 0
    assert RD > 0 and RD % 2 == 0 and RD % VEC_FULL == 0

    LDS_M_ELEMS = NW * SLICE_SZ
    LDS_KV_ELEMS = NW * SLICE_SZ
    LDS_W_ELEMS = NW * SLICE_SZ

    @fx.struct
    class SharedStorage:
        lds_m: fx.Array[fx.Float32, LDS_M_ELEMS, 16]
        lds_kv: fx.Array[fx.Float32, LDS_KV_ELEMS, 16]
        lds_w: fx.Array[fx.Float32, LDS_W_ELEMS, 16]

    _kname = (
        f"hca_compress_atomic_fused_w32_D{D}_R{K}_NW{NW}"
        f"_SL{SLICE_SZ}_S{state_size}_flydsl"
    )
    fm_fast = arith.FastMathFlags.fast

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_TH, 1, 1])
    def kernel(
        kv_in: fx.Tensor,
        kv_in_row_stride: Int32,
        score_in: fx.Tensor,
        score_in_row_stride: Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,
        kv_state_slot_stride: Int32,
        kv_state_pos_stride: Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: Int32,
        score_state_pos_stride: Int32,
        state_slot_mapping: fx.Tensor,
        ape: fx.Tensor,
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: Int32,
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: Int32,
        kv_cache_token_stride: Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: Int32,
        tile_done: fx.Tensor,
    ):
        f32 = T.f32
        i32 = T.i32

        pid = fx.block_idx.x  # boundary index
        sid = fx.block_idx.y  # slice index [0, NUM_SPLIT)
        tid = fx.thread_idx.x

        c_zero_i32 = arith.constant(0, type=i32)
        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)

        def fexp_f32(x):
            return llvm.call_intrinsic(
                f32, "llvm.amdgcn.exp2.f32", [x * c_log2e], [], []
            )

        def wave_reduce_add(w):
            for sh_exp in range_constexpr(log2_block):
                off = BLOCK_THREADS // (2 << sh_exp)
                peer = fx.Float32(w).shuffle_xor(off, BLOCK_THREADS).ir_value()
                w = arith.AddFOp(w, peer, fastmath=fm_fast).result
            return w

        wid = fx.Int32((fx.Uint32(tid) // BLOCK_THREADS).ir_value())
        lid = fx.Int32((fx.Uint32(tid) % BLOCK_THREADS).ir_value())

        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_vec = fx.Vector(
            buffer_ops.buffer_load(plan_rsrc, fx.Int32(pid) * 4, vec_width=4, dtype=i32)
        )
        ragged_id = plan_vec[0]
        batch_id = plan_vec[1]
        position = plan_vec[2]
        window_len = plan_vec[3]

        def _body():
            col_off_base = fx.Int32(sid) * SLICE_SZ + lid * VEC

            slot_map_rsrc = buffer_ops.create_buffer_resource(
                state_slot_mapping, max_size=True
            )
            slot = buffer_ops.buffer_load(
                slot_map_rsrc, batch_id, vec_width=1, dtype=i32
            )

            kv_in_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            score_in_rsrc = buffer_ops.create_buffer_resource(score_in, max_size=True)
            kv_state_rsrc = buffer_ops.create_buffer_resource(
                kv_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, kv_state_slot_stride),
            )
            score_state_rsrc = buffer_ops.create_buffer_resource(
                score_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, score_state_slot_stride),
            )
            ape_rsrc = buffer_ops.create_buffer_resource(ape, max_size=True)

            def _load_bf16_vec_to_f32(rsrc, base_off_elems_i32):
                base_off = fx.Int32(base_off_elems_i32)
                off_dw = fx.Int32((fx.Uint32(base_off_elems_i32) >> 1).ir_value())
                if const_expr(VEC == 1):
                    lane_in_dw = base_off & 1
                    raw_s = buffer_ops.buffer_load(rsrc, off_dw, vec_width=1, dtype=i32)
                    hi = fx.Int32((fx.Uint32(raw_s) >> 16).ir_value())
                    lo_or_hi = arith.select(
                        arith.cmpi(
                            CmpIPredicate.eq,
                            lane_in_dw.ir_value(),
                            c_zero_i32,
                        ),
                        raw_s,
                        hi.ir_value(),
                    )
                    lo16 = arith.andi(lo_or_hi, arith.constant(0xFFFF, type=i32))
                    lo16_v = fx.Vector.from_elements([lo16], dtype=fx.Int32)
                    bf16_pair = lo16_v.bitcast(fx.BFloat16)
                    return [bf16_pair[0].to(fx.Float32).ir_value()]
                else:
                    dwords = VEC // 2
                    if const_expr(dwords == 1):
                        raw_s = buffer_ops.buffer_load(
                            rsrc, off_dw, vec_width=1, dtype=i32
                        )
                        raw = fx.Vector.from_elements([raw_s], dtype=fx.Int32)
                    elif const_expr(dwords <= 4):
                        raw = fx.Vector(
                            buffer_ops.buffer_load(
                                rsrc, off_dw, vec_width=dwords, dtype=i32
                            )
                        )
                    else:
                        half_dw = dwords // 2
                        r0 = fx.Vector(
                            buffer_ops.buffer_load(
                                rsrc, off_dw, vec_width=half_dw, dtype=i32
                            )
                        )
                        r1 = fx.Vector(
                            buffer_ops.buffer_load(
                                rsrc,
                                off_dw + half_dw,
                                vec_width=half_dw,
                                dtype=i32,
                            )
                        )
                        raw = fx.Vector.from_elements(
                            [r0[i] for i in range(half_dw)]
                            + [r1[i] for i in range(half_dw)],
                            dtype=fx.Int32,
                        )
                    vec_bf16 = raw.bitcast(fx.BFloat16)
                    return [vec_bf16[i].to(fx.Float32).ir_value() for i in range(VEC)]

            def _load_f32_vec(rsrc, base_off_elems_i32):
                if const_expr(VEC <= 4):
                    raw = buffer_ops.buffer_load(
                        rsrc, base_off_elems_i32, vec_width=VEC, dtype=f32
                    )
                    if const_expr(VEC == 1):
                        return [raw]
                    return [fx.Vector(raw)[i].ir_value() for i in range(VEC)]
                else:
                    quarter = 4
                    n_chunks = VEC // quarter
                    result = []
                    for q in range_constexpr(n_chunks):
                        r = fx.Vector(
                            buffer_ops.buffer_load(
                                rsrc,
                                fx.Int32(base_off_elems_i32) + q * quarter,
                                vec_width=quarter,
                                dtype=f32,
                            )
                        )
                        result.extend(r[j].ir_value() for j in range(quarter))
                    return result

            def _issue_phase2_loads(k_i32):
                k = fx.Int32(k_i32)
                ape_row = fx.Int32((fx.Uint32(k_i32) % ratio).ir_value())
                in_row_raw = fx.Int32(ragged_id) - (fx.Int32(K - 1) - k)
                in_row = fx.max(in_row_raw, fx.Int32(0))
                base_in_off = in_row * fx.Int32(kv_in_row_stride) + col_off_base
                base_sc_off = in_row * fx.Int32(score_in_row_stride) + col_off_base
                base_ape_off = ape_row * DIM_FULL + col_off_base
                kv = _load_bf16_vec_to_f32(kv_in_rsrc, base_in_off)
                sc = _load_bf16_vec_to_f32(score_in_rsrc, base_sc_off)
                ape_v = _load_f32_vec(ape_rsrc, base_ape_off)
                return kv, sc, ape_v

            def _issue_phase1_loads(k_i32):
                s = (fx.Int32(position) - fx.Int32(K - 1) + fx.Int32(k_i32)).ir_value()
                is_pad = arith.cmpi(CmpIPredicate.slt, s, c_zero_i32)
                s_safe = fx.Int32(arith.select(is_pad, c_zero_i32, s))
                ring = fx.Int32((fx.Uint32(s_safe.ir_value()) % state_size).ir_value())
                base_kv_off = ring * fx.Int32(kv_state_pos_stride) + col_off_base
                base_sc_off = ring * fx.Int32(score_state_pos_stride) + col_off_base
                kv_list = _load_f32_vec(kv_state_rsrc, base_kv_off)
                sc_list = _load_f32_vec(score_state_rsrc, base_sc_off)
                sc_padded = [
                    arith.select(is_pad, c_neg_inf, sc_list[i]) for i in range(VEC)
                ]
                return kv_list, sc_padded

            def _softmax_step_padded(
                m_old_list, kv_old_list, w_old_list, score_k_list, kv_k_list
            ):
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = m_old_list[i]
                    kv_old = kv_old_list[i]
                    w_old = w_old_list[i]
                    score_k = score_k_list[i]
                    kv_k = kv_k_list[i]
                    m_new = fx.max(fx.Float32(m_old), fx.Float32(score_k)).ir_value()
                    is_first = arith.cmpf(CmpFPredicate.OEQ, m_old, c_neg_inf)
                    scale_active = fexp_f32(arith.subf(m_old, m_new))
                    scale_v = arith.select(is_first, c_zero_f32, scale_active)
                    wk_active = fexp_f32(arith.subf(score_k, m_new))
                    is_pad_score = arith.cmpf(CmpFPredicate.OEQ, score_k, c_neg_inf)
                    w_k = arith.select(is_pad_score, c_zero_f32, wk_active)
                    new_kv.append(
                        arith.AddFOp(
                            arith.MulFOp(kv_old, scale_v, fastmath=fm_fast).result,
                            arith.MulFOp(w_k, kv_k, fastmath=fm_fast).result,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_w.append(
                        arith.AddFOp(
                            arith.MulFOp(w_old, scale_v, fastmath=fm_fast).result,
                            w_k,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_m.append(m_new)
                return new_m, new_kv, new_w

            # -- K-split pool + online softmax (same as Kernel A) --
            k_start_i32 = wid * K_PER_WAVE
            k_end_i32 = k_start_i32 + K_PER_WAVE
            split_i32 = fx.min(fx.max(window_len, k_start_i32), k_end_i32)

            init_m = [c_neg_inf for _ in range(VEC)]
            init_kv = [c_zero_f32 for _ in range(VEC)]
            init_w = [c_zero_f32 for _ in range(VEC)]
            init_state = init_m + init_kv + init_w

            phase1_local = init_state
            for k_static, state in range(
                k_start_i32.ir_value(),
                split_i32.ir_value(),
                1,
                init=init_state,
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = fx.Int32(k_static)
                kv_v, sc_v = _issue_phase1_loads(k_i32)
                new_m, new_kv, new_w = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, sc_v, kv_v
                )
                phase1_local = yield list(new_m) + list(new_kv) + list(new_w)

            if const_expr(not enable_prefetch_input):
                final = phase1_local
                for k_static, state in range(
                    split_i32.ir_value(),
                    k_end_i32.ir_value(),
                    1,
                    init=phase1_local,
                ):
                    m_lane = list(state[0:VEC])
                    kv_lane = list(state[VEC : 2 * VEC])
                    w_lane = list(state[2 * VEC : 3 * VEC])
                    k_i32 = fx.Int32(k_static)
                    p2_kv, p2_sc, p2_ape = _issue_phase2_loads(k_i32)
                    p2_score = [
                        arith.AddFOp(p2_sc[i], p2_ape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    new_m, new_kv, new_w = _softmax_step_padded(
                        m_lane, kv_lane, w_lane, p2_score, p2_kv
                    )
                    final = yield (list(new_m) + list(new_kv) + list(new_w))
            else:
                c_k_end_m1 = k_end_i32 - 1

                # Prologue: prefetch data for the first 2 iterations.
                k_pro0 = fx.min(split_i32, c_k_end_m1)
                k_pro1 = fx.min(split_i32 + 1, c_k_end_m1)
                p0_kv, p0_sc, p0_ape = _issue_phase2_loads(k_pro0)
                p1_kv, p1_sc, p1_ape = _issue_phase2_loads(k_pro1)
                init_pf = (
                    list(phase1_local)
                    + list(p0_kv)
                    + list(p0_sc)
                    + list(p0_ape)
                    + list(p1_kv)
                    + list(p1_sc)
                    + list(p1_ape)
                )

                # 2x-unrolled loop: step by 2, each body processes
                # iter k (from pre0) and iter k+1 (from pre1), then
                # issues loads for the next pair (k+2, k+3).
                # Upper bound: round down the Phase 2 range to even
                # count, leaving 1-2 tail iterations.
                p2_count = fx.max(c_k_end_m1 - split_i32, fx.Int32(0))
                p2_even = fx.Int32((fx.Uint32(p2_count.ir_value()) & ~1).ir_value())
                k_end_u2 = split_i32 + p2_even

                loop_final = init_pf
                for k_static, state in range(
                    split_i32.ir_value(),
                    k_end_u2.ir_value(),
                    2,
                    init=init_pf,
                ):
                    m_lane = list(state[0:VEC])
                    kv_lane = list(state[VEC : 2 * VEC])
                    w_lane = list(state[2 * VEC : 3 * VEC])
                    pf0_kv = list(state[3 * VEC : 4 * VEC])
                    pf0_sc = list(state[4 * VEC : 5 * VEC])
                    pf0_ape = list(state[5 * VEC : 6 * VEC])
                    pf1_kv = list(state[6 * VEC : 7 * VEC])
                    pf1_sc = list(state[7 * VEC : 8 * VEC])
                    pf1_ape = list(state[8 * VEC : 9 * VEC])

                    k_i32 = fx.Int32(k_static)
                    # Issue next pair's loads early.
                    nxt0_kv, nxt0_sc, nxt0_ape = _issue_phase2_loads(k_i32 + 2)
                    nxt1_kv, nxt1_sc, nxt1_ape = _issue_phase2_loads(
                        fx.min(k_i32 + 3, c_k_end_m1)
                    )

                    # Compute iter k (consume pre0).
                    sc_a = [
                        arith.AddFOp(pf0_sc[i], pf0_ape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    m_a, kv_a, w_a = _softmax_step_padded(
                        m_lane, kv_lane, w_lane, sc_a, pf0_kv
                    )
                    # Compute iter k+1 (consume pre1).
                    sc_b = [
                        arith.AddFOp(pf1_sc[i], pf1_ape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    m_b, kv_b, w_b = _softmax_step_padded(m_a, kv_a, w_a, sc_b, pf1_kv)
                    loop_final = yield (
                        list(m_b)
                        + list(kv_b)
                        + list(w_b)
                        + list(nxt0_kv)
                        + list(nxt0_sc)
                        + list(nxt0_ape)
                        + list(nxt1_kv)
                        + list(nxt1_sc)
                        + list(nxt1_ape)
                    )

                # Tail: 1 or 2 remaining iterations after the
                # unrolled loop, gated by whether Phase 2 is
                # non-empty / has at least 2 elements.
                is_p2 = arith.cmpi(
                    CmpIPredicate.slt,
                    split_i32.ir_value(),
                    k_end_i32.ir_value(),
                )
                is_p2_ge2 = arith.cmpi(
                    CmpIPredicate.slt,
                    (split_i32 + 1).ir_value(),
                    k_end_i32.ir_value(),
                )
                # When p2_count is even the loop consumed all but
                # the last pair's second element; when odd it left
                # two. Use is_odd to pick the right gate.
                is_odd = arith.cmpi(
                    CmpIPredicate.ne,
                    (p2_count & 1).ir_value(),
                    arith.constant(0, type=i32),
                )

                m_t = list(loop_final[0:VEC])
                kv_t = list(loop_final[VEC : 2 * VEC])
                w_t = list(loop_final[2 * VEC : 3 * VEC])
                t0_kv = list(loop_final[3 * VEC : 4 * VEC])
                t0_sc = list(loop_final[4 * VEC : 5 * VEC])
                t0_ape = list(loop_final[5 * VEC : 6 * VEC])
                t1_kv = list(loop_final[6 * VEC : 7 * VEC])
                t1_sc = list(loop_final[7 * VEC : 8 * VEC])
                t1_ape = list(loop_final[8 * VEC : 9 * VEC])

                # Tail iteration 0 (penultimate).
                tail0_score = [
                    arith.select(
                        is_p2,
                        arith.AddFOp(
                            t0_sc[i],
                            t0_ape[i],
                            fastmath=fm_fast,
                        ).result,
                        c_neg_inf,
                    )
                    for i in range(VEC)
                ]
                r0_m, r0_kv, r0_w = _softmax_step_padded(
                    m_t, kv_t, w_t, tail0_score, t0_kv
                )

                # Tail iteration 1 (last). Only when p2_count is
                # odd (the unrolled loop left an even number of
                # remaining iters, and tail0 consumed one).
                tail1_gate = arith.andi(is_p2_ge2, is_odd)
                tail1_score = [
                    arith.select(
                        tail1_gate,
                        arith.AddFOp(
                            t1_sc[i],
                            t1_ape[i],
                            fastmath=fm_fast,
                        ).result,
                        c_neg_inf,
                    )
                    for i in range(VEC)
                ]
                r1_m, r1_kv, r1_w = _softmax_step_padded(
                    r0_m, r0_kv, r0_w, tail1_score, t1_kv
                )
                final = list(r1_m) + list(r1_kv) + list(r1_w)

            m_local = list(final[0:VEC])
            kv_local = list(final[VEC : 2 * VEC])
            w_local = list(final[2 * VEC : 3 * VEC])

            # -- LDS write + barrier (same as Kernel A) --
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            lds_m_ptr = lds.lds_m.ptr
            lds_kv_ptr = lds.lds_kv.ptr
            lds_w_ptr = lds.lds_w.ptr
            lds_thread_base = wid * SLICE_SZ + lid * VEC
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + i
                fx.ptr_store(m_local[i], lds_m_ptr + idx_i)
                fx.ptr_store(kv_local[i], lds_kv_ptr + idx_i)
                fx.ptr_store(w_local[i], lds_w_ptr + idx_i)

            gpu.barrier()

            # -- Wave 0: cross-wave reduce + scratch write + atomic --
            def _wave0():
                comp_list = []
                for i in range_constexpr(VEC):
                    lane_off = lid * VEC + i
                    m_g = fx.Float32(c_neg_inf)
                    m_arr = []
                    for w in range_constexpr(NW):
                        m_w = fx.ptr_load(lds_m_ptr + (lane_off + w * SLICE_SZ))
                        m_arr.append(m_w)
                        m_g = m_g.maximumf(m_w)

                    kv_sum = fx.Float32(0.0)
                    w_sum = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        idx_w = lane_off + w * SLICE_SZ
                        kv_w = fx.ptr_load(lds_kv_ptr + idx_w)
                        w_w = fx.ptr_load(lds_w_ptr + idx_w)
                        m_w = m_arr[w]
                        scale_w = fx.Float32(fexp_f32((m_w - m_g).ir_value()))
                        kv_sum = kv_sum + kv_w * scale_w
                        w_sum = w_sum + w_w * scale_w
                    rcp_w = fx.Float32(
                        llvm.call_intrinsic(
                            f32,
                            "llvm.amdgcn.rcp.f32",
                            [w_sum.ir_value()],
                            [],
                            [],
                        )
                    )
                    comp_list.append(kv_sum * rcp_w)

                # Write this slice to scratch
                out_rsrc = buffer_ops.create_buffer_resource(
                    kv_compressed, max_size=True
                )
                out_off = (
                    fx.Int32(pid) * fx.Int32(kv_compressed_row_stride) + col_off_base
                )
                # sc0|sc1 coherent stores: write-through to L2 without
                # flushing the whole cache (cf. splitk_epilogue pattern).
                CPOL_COHERENT = 0x1 | 0x10
                if const_expr(VEC == 1):
                    buffer_ops.buffer_store(
                        comp_list[0].ir_value(),
                        out_rsrc,
                        out_off,
                        cache_modifier=CPOL_COHERENT,
                    )
                elif const_expr(VEC <= 4):
                    out_vec = fx.Vector.from_elements(comp_list, dtype=fx.Float32)
                    buffer_ops.buffer_store(
                        out_vec.ir_value(),
                        out_rsrc,
                        out_off,
                        cache_modifier=CPOL_COHERENT,
                    )
                else:
                    quarter = 4
                    n_chunks = VEC // quarter
                    for q in range_constexpr(n_chunks):
                        base = q * quarter
                        sv = fx.Vector.from_elements(
                            comp_list[base : base + quarter],
                            dtype=fx.Float32,
                        )
                        buffer_ops.buffer_store(
                            sv.ir_value(),
                            out_rsrc,
                            out_off + base,
                            cache_modifier=CPOL_COHERENT,
                        )

            # -- Tail: norm + rope + scatter (last block only, wave 0) --
            def _tail():
                """Runs on wave 0 of the last-arriving block per boundary."""
                CPOL_COHERENT = 0x1 | 0x10

                c_eps = arith.constant(rms_eps, type=f32)
                c_inv_D = arith.constant(1.0 / D, type=f32)

                # Load full D=512 compressed row from scratch (sc0|sc1
                # coherent reads — matches the coherent stores above).
                kvc_rsrc = buffer_ops.create_buffer_resource(
                    kv_compressed, max_size=True
                )
                tid_x_vec_full = lid * VEC_FULL
                base_off = (
                    fx.Int32(pid) * fx.Int32(kv_compressed_row_stride) + tid_x_vec_full
                )
                quarter = 4
                n_chunks_full = VEC_FULL // quarter
                comp_lane = []
                for q in range_constexpr(n_chunks_full):
                    r = fx.Vector(
                        buffer_ops.buffer_load(
                            kvc_rsrc,
                            base_off + q * quarter,
                            vec_width=quarter,
                            dtype=f32,
                            cache_modifier=CPOL_COHERENT,
                        )
                    )
                    comp_lane += [r[i].ir_value() for i in range_constexpr(quarter)]

                # RMSNorm
                sq_local = arith.constant(0.0, type=f32)
                for i in range_constexpr(VEC_FULL):
                    sq_local = arith.AddFOp(
                        sq_local,
                        arith.MulFOp(
                            comp_lane[i],
                            comp_lane[i],
                            fastmath=fm_fast,
                        ).result,
                        fastmath=fm_fast,
                    ).result
                sq_full = wave_reduce_add(sq_local)
                var = arith.MulFOp(sq_full, c_inv_D, fastmath=fm_fast).result
                rrms = fmath.rsqrt(
                    arith.AddFOp(var, c_eps, fastmath=fm_fast).result,
                    fastmath=fm_fast,
                )

                # rms_weight load (VEC_FULL=16 elements at lid*VEC_FULL)
                rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
                if const_expr(rms_weight_is_bf16):
                    dwords = (VEC_FULL + 1) // 2
                    off_dw = fx.Int32(
                        (fx.Uint32(tid_x_vec_full.ir_value()) >> 1).ir_value()
                    )
                    if const_expr(dwords <= 4):
                        raw = fx.Vector(
                            buffer_ops.buffer_load(
                                rmsw_rsrc,
                                off_dw,
                                vec_width=dwords,
                                dtype=i32,
                            )
                        )
                        vec_bf16 = raw.bitcast(fx.BFloat16)
                        rmsw_lane = [
                            vec_bf16[i].to(fx.Float32).ir_value()
                            for i in range_constexpr(VEC_FULL)
                        ]
                    else:
                        half_dw = 4
                        half_bf16 = half_dw * 2
                        rmsw_lane = []
                        for chunk in range_constexpr(dwords // half_dw):
                            r = buffer_ops.buffer_load(
                                rmsw_rsrc,
                                off_dw + chunk * half_dw,
                                vec_width=half_dw,
                                dtype=i32,
                            )
                            vbf16 = fx.Vector(r).bitcast(fx.BFloat16)
                            rmsw_lane += [
                                vbf16[i].to(fx.Float32).ir_value()
                                for i in range_constexpr(half_bf16)
                            ]
                else:
                    rmsw_lane = []
                    for q in range_constexpr(n_chunks_full):
                        r = fx.Vector(
                            buffer_ops.buffer_load(
                                rmsw_rsrc,
                                tid_x_vec_full + q * quarter,
                                vec_width=quarter,
                                dtype=f32,
                            )
                        )
                        rmsw_lane += [r[i].ir_value() for i in range_constexpr(quarter)]

                normed_lane = [
                    arith.MulFOp(
                        arith.MulFOp(comp_lane[i], rrms, fastmath=fm_fast).result,
                        rmsw_lane[i],
                        fastmath=fm_fast,
                    ).result
                    for i in range(VEC_FULL)
                ]

                # GPT-J RoPE on last RD dims
                comp_pos_i32 = (
                    fx.Int32((fx.Uint32(position) // ratio).ir_value()) * ratio
                )
                cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
                sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
                cos_row_base = comp_pos_i32 * (RD // 2)

                is_rope_t = arith.cmpi(
                    CmpIPredicate.sge,
                    lid.ir_value(),
                    arith.constant(ROPE_THREAD_LO_FULL, type=i32),
                )
                rope_rel_raw = lid - ROPE_THREAD_LO_FULL
                rope_rel = fx.max(rope_rel_raw, fx.Int32(0))
                cs_lo = rope_rel * PAIRS_PER_THREAD_FULL

                if const_expr(PAIRS_PER_THREAD_FULL <= 4):
                    cos_vec = fx.Vector(
                        buffer_ops.buffer_load(
                            cos_rsrc,
                            cos_row_base + cs_lo,
                            vec_width=PAIRS_PER_THREAD_FULL,
                            dtype=T.bf16,
                        )
                    )
                    sin_vec = fx.Vector(
                        buffer_ops.buffer_load(
                            sin_rsrc,
                            cos_row_base + cs_lo,
                            vec_width=PAIRS_PER_THREAD_FULL,
                            dtype=T.bf16,
                        )
                    )
                    cos_vals = [
                        cos_vec[i].to(fx.Float32).ir_value()
                        for i in range(PAIRS_PER_THREAD_FULL)
                    ]
                    sin_vals = [
                        sin_vec[i].to(fx.Float32).ir_value()
                        for i in range(PAIRS_PER_THREAD_FULL)
                    ]
                else:
                    # PAIRS_PER_THREAD_FULL > 4: split into chunks
                    cos_vals = []
                    sin_vals = []
                    ppt_chunks = PAIRS_PER_THREAD_FULL // 4
                    for pq in range_constexpr(ppt_chunks):
                        cv = fx.Vector(
                            buffer_ops.buffer_load(
                                cos_rsrc,
                                cos_row_base + cs_lo + pq * 4,
                                vec_width=4,
                                dtype=T.bf16,
                            )
                        )
                        sv = fx.Vector(
                            buffer_ops.buffer_load(
                                sin_rsrc,
                                cos_row_base + cs_lo + pq * 4,
                                vec_width=4,
                                dtype=T.bf16,
                            )
                        )
                        cos_vals += [cv[i].to(fx.Float32).ir_value() for i in range(4)]
                        sin_vals += [sv[i].to(fx.Float32).ir_value() for i in range(4)]

                rotated_lane = list(normed_lane)
                for k in range_constexpr(PAIRS_PER_THREAD_FULL):
                    e = normed_lane[2 * k]
                    o = normed_lane[2 * k + 1]
                    c = cos_vals[k]
                    s = sin_vals[k]
                    new_e = arith.subf(
                        arith.MulFOp(e, c, fastmath=fm_fast).result,
                        arith.MulFOp(o, s, fastmath=fm_fast).result,
                    )
                    new_o = arith.AddFOp(
                        arith.MulFOp(e, s, fastmath=fm_fast).result,
                        arith.MulFOp(o, c, fastmath=fm_fast).result,
                        fastmath=fm_fast,
                    ).result
                    rotated_lane[2 * k] = new_e
                    rotated_lane[2 * k + 1] = new_o

                # BF16 paged scatter
                ci = fx.Int32((fx.Uint32(position) // ratio).ir_value())
                block_in_seq = fx.Int32(
                    (fx.Uint32(ci.ir_value()) // k_per_block).ir_value()
                )
                slot_in_block = fx.Int32(
                    (fx.Uint32(ci.ir_value()) % k_per_block).ir_value()
                )
                bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
                bt_off = (
                    fx.Int32(batch_id) * fx.Int32(block_table_seq_stride) + block_in_seq
                )
                physical_block = buffer_ops.buffer_load(
                    bt_rsrc, bt_off, vec_width=1, dtype=i32
                )
                cache_base = slot_in_block * fx.Int32(kv_cache_token_stride)
                out_cache_rsrc = buffer_ops.create_buffer_resource(
                    kv_cache,
                    max_size=True,
                    base_byte_offset=block_base_bytes_i64(
                        physical_block, kv_cache_block_stride, 2
                    ),
                )

                out_lane = [
                    arith.select(is_rope_t, rotated_lane[i], normed_lane[i])
                    for i in range_constexpr(VEC_FULL)
                ]
                cache_off = cache_base + tid_x_vec_full
                out_vec_t = T.vec(VEC_FULL, T.bf16)
                raw_vec = fx.Vector.from_elements(out_lane, dtype=fx.Float32)
                bf16_vec = raw_vec.truncf(out_vec_t)
                cache_off_dw = fx.Int32(
                    (fx.Uint32(cache_off.ir_value()) >> 1).ir_value()
                )
                bf16_as_i32 = bf16_vec.bitcast(fx.Int32)
                dwords_out = (VEC_FULL + 1) // 2
                if const_expr(dwords_out <= 4):
                    buffer_ops.buffer_store(
                        bf16_as_i32.ir_value(),
                        out_cache_rsrc,
                        cache_off_dw,
                    )
                else:
                    lo = fx.Vector.from_elements(
                        [bf16_as_i32[i] for i in range(4)],
                        dtype=fx.Int32,
                    )
                    hi = fx.Vector.from_elements(
                        [bf16_as_i32[i] for i in range(4, 8)],
                        dtype=fx.Int32,
                    )
                    buffer_ops.buffer_store(lo.ir_value(), out_cache_rsrc, cache_off_dw)
                    buffer_ops.buffer_store(
                        hi.ir_value(), out_cache_rsrc, cache_off_dw + 4
                    )

            if wid == 0:
                _wave0()
                # Atomic: lane 0 increments the per-boundary counter.
                old_val_lane = arith.constant(0, type=i32)
                if lid == 0:
                    counter_addr = (
                        fx.Int64(fx.ptrtoint(fx.get_iter(tile_done)))
                        + fx.Int64(pid) * 4
                    )
                    old_val_lane = atomic_add_agent(counter_addr, fx.Int32(1))
                # Broadcast lane 0's result to all lanes via shuffle
                # (avoids LDS store/load ordering pitfall).
                flag_i32 = fx.Int32(gpu.shuffle_idx(old_val_lane, 0, BLOCK_THREADS))
                is_last = arith.cmpi(
                    CmpIPredicate.eq,
                    flag_i32.ir_value(),
                    arith.constant(NUM_SPLIT - 1, type=i32),
                )
                if is_last:
                    _tail()

        if fx.Int32(position) >= 0:
            _body()

    @flyc.jit
    def launch_hca_atomic_fused(
        kv_in: fx.Tensor,
        kv_in_row_stride: fx.Int32,
        score_in: fx.Tensor,
        score_in_row_stride: fx.Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,
        kv_state_slot_stride: fx.Int32,
        kv_state_pos_stride: fx.Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: fx.Int32,
        score_state_pos_stride: fx.Int32,
        state_slot_mapping: fx.Tensor,
        ape: fx.Tensor,
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: fx.Int32,
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: fx.Int32,
        kv_cache_token_stride: fx.Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: fx.Int32,
        tile_done: fx.Tensor,
        plan_capacity: fx.Int32,
        stream: fx.Stream,
    ):
        idx_p = fx.Int64(plan_capacity)
        idx_s = fx.Int64(NUM_SPLIT)
        k = kernel(
            kv_in,
            kv_in_row_stride,
            score_in,
            score_in_row_stride,
            plan,
            kv_state,
            kv_state_slot_stride,
            kv_state_pos_stride,
            score_state,
            score_state_slot_stride,
            score_state_pos_stride,
            state_slot_mapping,
            ape,
            kv_compressed,
            kv_compressed_row_stride,
            rms_weight,
            cos_cache,
            sin_cache,
            kv_cache,
            kv_cache_block_stride,
            kv_cache_token_stride,
            block_table,
            block_table_seq_stride,
            tile_done,
        )
        k.launch(
            grid=(idx_p, idx_s, 1),
            block=(BLOCK_TH, 1, 1),
            stream=stream,
        )

    return launch_hca_atomic_fused


@lru_cache(maxsize=16)
def compile_hca_atomic_fused_gfx1250(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    slice_size: int = 128,
    k_per_block: int = 64,
    rms_weight_is_bf16: bool = False,
    rms_eps: float = 1e-6,
    enable_prefetch_input: bool = True,
):
    """Compile the atomic-fused HCA kernel (pool+softmax+norm+rope+scatter)."""
    launcher = _build_atomic_fused_compress_kernel(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        state_size=state_size,
        k_split_num_waves=k_split_num_waves,
        slice_size=slice_size,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        enable_prefetch_input=enable_prefetch_input,
    )
    launcher.compile_hints = dict(_FUSED_COMPILE_HINTS)
    return launcher


# ============================================================================
# Cached compile + public API
# ============================================================================


_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


@lru_cache(maxsize=32)
def compile_hca_compress_forward_gfx1250(
    *,
    head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    slice_size: int = 64,
    enable_prefetch_input: bool = True,
    enable_tdm: bool = False,
):
    """Build the HCA compress_forward launcher (multi-wave LDS K-split).

    Each wave handles K / ``k_split_num_waves`` K-positions; cross-wave LDS
    reduction merges per-wave softmax accumulators. Each iter selects
    between Phase 1 (state cache, ``k < window_len``) and Phase 2 (input)
    by splitting the wave's K range at ``clamp(window_len, k_start, k_end)``.

    ``slice_size`` controls per-thread vector width (VEC = slice_size / 64).
    Larger slice_size means each thread handles more head_dim elements per
    K-iter (wider buffer_load -> better HBM coalescing), but fewer blocks
    per boundary (NUM_SPLIT = head_dim / slice_size). slice_size=64 -> VEC=1
    (8 blocks/boundary, small-N champion); slice_size=512 -> VEC=8
    (1 block/boundary, v1-like HBM access, large-N champion).

    ``enable_prefetch_input``: when True, Phase 2 uses single-iter prefetch
    to overlap memory latency with softmax compute.

    ``enable_tdm``: when True, Phase 2 kv_in loads use TDM async DMA
    (Global -> LDS) instead of wavefront buffer_load.

    ``state_size`` is the ring-buffer modulo of ``kv_state.shape[1]`` (>= ratio).
    """
    launcher = _build_compress_forward_kernel(
        head_dim=head_dim,
        ratio=ratio,
        state_size=state_size,
        k_split_num_waves=k_split_num_waves,
        slice_size=slice_size,
        enable_prefetch_input=enable_prefetch_input,
        enable_tdm=enable_tdm,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


@lru_cache(maxsize=16)
def compile_hca_norm_rope_scatter_gfx1250(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    k_per_block: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    quant: bool = False,
    quant_group_size: int = 64,
):
    launcher = _build_norm_rope_scatter_kernel(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant=quant,
        quant_group_size=quant_group_size,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_hca_compress_attn_gfx1250(
    *,
    kv_in: torch.Tensor,  # [num_q_tokens, head_dim] bf16
    score_in: torch.Tensor,  # [num_q_tokens, head_dim] bf16
    kv_state: torch.Tensor,  # [num_slots, STATE_SIZE, head_dim] f32
    score_state: torch.Tensor,  # same shape as kv_state
    state_slot_mapping: torch.Tensor,  # [bs] i32
    plan_gpu: torch.Tensor,  # [num_compress, 4] i32
    ape: torch.Tensor,  # [ratio, head_dim] f32
    rms_weight: torch.Tensor,  # [head_dim] f32 or bf16
    rms_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    k_per_block: int,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    kv_compressed_scratch: torch.Tensor | None = None,
    quant: bool = False,
    k_rope_cache: torch.Tensor | None = None,
    quant_group_size: int = 64,
    k_split_num_waves: int | None = None,
    slice_size: int | None = None,
    stream: torch.cuda.Stream | None = None,
    fused: bool = True,
) -> None:
    """HCA-only compress + norm+rope+scatter (V4-Pro Main path).

    Restrictions: ratio=128, overlap=False (implicit), head_dim=512 supported.

    Cache scatter dtype:
      * ``quant=False`` (default): BF16 single-buffer scatter -- nope + rope written
        contiguously into ``kv_cache`` [NB, k_per_block, head_dim] bf16.
      * ``quant=True``: FP8 1xG e8m0 group-quant. ``kv_cache`` is fp8
        [NB, k_per_block, entry] holding nope fp8 + inline duplicated e8m0 scale
        (V4 nm asm layout); rotated PE bf16 goes to ``k_rope_cache``
        [NB, k_per_block, rope_head_dim] bf16. Byte-identical to the C++
        ``fused_kv_compress_scatter`` k_wave output.

    Phase 1 (state cache) is enabled by passing real ``kv_state`` /
    ``score_state`` / ``state_slot_mapping``. When ``window_len > 0`` in
    the plan, the corresponding K iters are sourced from the state cache
    ring buffer instead of kv_in / score_in.

    When ``k_split_num_waves`` / ``slice_size`` are ``None`` (the default),
    the launcher auto-picks via :func:`hca_per_n_config` keyed on
    ``plan_gpu.shape[0]`` (CUDAGraph-stable dispatch -- see that function's
    docstring). Override only when bench-sweeping; the default matches the
    production tuning used by ATOM's compressor.
    """
    if k_split_num_waves is None or slice_size is None:
        from .fused_compress_attn_gfx1250 import hca_per_n_config_gfx1250

        auto_slice, auto_kw = hca_per_n_config_gfx1250(plan_gpu.shape[0])
        if slice_size is None:
            slice_size = auto_slice
        if k_split_num_waves is None:
            k_split_num_waves = auto_kw
    # User-facing input validation -- must be ``raise`` not ``assert`` (asserts
    # are stripped under ``python -O``, which would let invalid inputs reach
    # the kernel and silently corrupt outputs / fault the GPU).
    if head_dim != 512:
        raise ValueError(f"HCA 2-kernel only supports head_dim=512, got {head_dim}")
    if ratio != 128:
        raise ValueError(f"HCA 2-kernel only supports ratio=128, got {ratio}")
    if kv_in.dim() != 2 or kv_in.shape[1] != head_dim:
        raise ValueError(f"kv_in shape {tuple(kv_in.shape)} != [*, {head_dim}]")
    if score_in.shape != kv_in.shape:
        raise ValueError(f"score_in shape {tuple(score_in.shape)} != kv_in")
    if kv_in.dtype != torch.bfloat16 or score_in.dtype != torch.bfloat16:
        raise TypeError(
            f"kv_in/score_in must be bf16; got {kv_in.dtype}/{score_in.dtype}"
        )
    if kv_in.stride(-1) != 1 or score_in.stride(-1) != 1:
        raise ValueError("kv_in/score_in inner stride must be 1")
    if kv_in.stride(0) % 2 != 0 or score_in.stride(0) % 2 != 0:
        raise ValueError(
            "kv_in/score_in row strides (bf16 elem) must be even for dword bitcast"
        )

    plan_capacity = plan_gpu.shape[0]
    if plan_capacity == 0:
        return

    if ape.shape != (ratio, head_dim) or ape.dtype != torch.float32:
        raise ValueError(
            f"ape shape {tuple(ape.shape)} dtype {ape.dtype} != ({ratio}, {head_dim}) f32"
        )
    if not ape.is_contiguous():
        raise ValueError("ape must be contiguous")

    # State cache validation.
    if kv_state.dim() != 3 or kv_state.shape[2] != head_dim:
        raise ValueError(
            f"kv_state shape {tuple(kv_state.shape)} != [*, *, {head_dim}]"
        )
    state_size = kv_state.shape[1]
    if state_size < ratio:
        raise ValueError(f"state_size={state_size} must be >= K={ratio}")
    if score_state.shape != kv_state.shape:
        raise ValueError("score_state shape != kv_state")
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        raise TypeError("kv_state/score_state must be fp32")
    # Slot and ring strides are passed to the kernel and the descriptor is
    # rebased per slot, so the states may be strided views — a per-request
    # arena hands out a view whose slot stride is a whole entry. Only the
    # innermost dim must be unit stride: the kernel addresses it as
    # `col_off + lane`.
    if kv_state.stride(-1) != 1 or score_state.stride(-1) != 1:
        raise ValueError("kv_state/score_state inner stride must be 1")
    if state_slot_mapping.dim() != 1 or state_slot_mapping.dtype != torch.int32:
        raise ValueError("state_slot_mapping must be 1D int32")

    if quant:
        if kv_cache.dtype not in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            raise TypeError(
                f"HCA fp8 kv_cache must be fp8 (e4m3fnuz/e4m3fn); got {kv_cache.dtype}"
            )
        if k_rope_cache is None:
            raise ValueError(
                "HCA fp8 path requires k_rope_cache (paged bf16 rope buffer)"
            )
        if k_rope_cache.dtype != torch.bfloat16:
            raise TypeError(f"k_rope_cache must be bf16; got {k_rope_cache.dtype}")
        if k_rope_cache.dim() != 3 or k_rope_cache.shape[2] != rope_head_dim:
            raise ValueError(
                f"k_rope_cache shape {tuple(k_rope_cache.shape)} != [NB, k_per_block, {rope_head_dim}]"
            )
        if k_rope_cache.stride(2) != 1:
            raise ValueError("k_rope_cache must be dense in the last dim")
    else:
        if kv_cache.dtype != torch.bfloat16:
            raise TypeError(f"HCA 2-kernel kv_cache must be bf16; got {kv_cache.dtype}")
    if block_tables.dtype != torch.int32:
        raise TypeError(f"block_tables must be int32; got {block_tables.dtype}")
    if not block_tables.is_contiguous():
        raise ValueError("block_tables must be contiguous")

    # Allocate kv_compressed scratch on demand.
    if kv_compressed_scratch is None:
        kv_compressed = torch.empty(
            (plan_capacity, head_dim),
            dtype=torch.float32,
            device=kv_in.device,
        )
    else:
        if kv_compressed_scratch.shape != (plan_capacity, head_dim):
            raise ValueError(
                f"kv_compressed_scratch shape {tuple(kv_compressed_scratch.shape)}"
                f" != ({plan_capacity}, {head_dim})"
            )
        if kv_compressed_scratch.dtype != torch.float32:
            raise TypeError("kv_compressed_scratch must be fp32")
        kv_compressed = kv_compressed_scratch

    # CRITICAL: must pass current_stream when stream is None. Stream(None) =
    # NULL/default stream, which during CUDA graph capture produces an empty
    # graph entry (kernel launches don't get recorded into the active graph),
    # so replay is a no-op -> HCA boundaries silently never fire in decode CG.
    # Match v1 single-kernel pattern (fused_compress_attn.py:1381).
    if stream is None:
        stream = torch.cuda.current_stream()
    stream_obj = Stream(stream)

    # ---- Atomic-fused single-launch path (BF16 non-quant only) ----
    # Keeps SL=128 (optimal CU occupancy) and uses an atomic completion
    # counter per boundary.  The last block to finish its slice runs the
    # norm+rope+scatter tail, eliminating one kernel launch.
    if fused and not quant:
        rms_weight_is_bf16 = rms_weight.dtype == torch.bfloat16
        fused_fn = compile_hca_atomic_fused_gfx1250(
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            ratio=ratio,
            state_size=int(kv_state.shape[1]),
            k_split_num_waves=k_split_num_waves,
            slice_size=slice_size,
            k_per_block=k_per_block,
            rms_weight_is_bf16=rms_weight_is_bf16,
            rms_eps=rms_eps,
        )
        tile_done = torch.zeros(plan_capacity, dtype=torch.int32, device=kv_in.device)
        fused_args = (
            kv_in,
            int(kv_in.stride(0)),
            score_in,
            int(score_in.stride(0)),
            plan_gpu,
            kv_state,
            int(kv_state.stride(0)),
            int(kv_state.stride(1)),
            score_state,
            int(score_state.stride(0)),
            int(score_state.stride(1)),
            state_slot_mapping,
            ape,
            kv_compressed,
            int(kv_compressed.stride(0)),
            rms_weight,
            cos_cache,
            sin_cache,
            kv_cache,
            int(kv_cache.stride(0)),
            int(kv_cache.stride(1)),
            block_tables,
            int(block_tables.stride(0)),
            tile_done,
            int(plan_capacity),
            stream_obj,
        )
        _run_compiled(fused_fn, *fused_args)
        return

    # ---- Legacy 2-kernel path (quant / forced) ----
    compress_fn = compile_hca_compress_forward_gfx1250(
        head_dim=head_dim,
        ratio=ratio,
        state_size=int(state_size),
        k_split_num_waves=k_split_num_waves,
        slice_size=slice_size,
    )
    compress_args = (
        kv_in,
        int(kv_in.stride(0)),
        score_in,
        int(score_in.stride(0)),
        plan_gpu,
        kv_state,
        int(kv_state.stride(0)),
        int(kv_state.stride(1)),
        score_state,
        int(score_state.stride(0)),
        int(score_state.stride(1)),
        state_slot_mapping,
        ape,
        kv_compressed,
        int(kv_compressed.stride(0)),
        int(plan_capacity),
        stream_obj,
    )
    _run_compiled(compress_fn, *compress_args)

    rms_weight_is_bf16 = rms_weight.dtype == torch.bfloat16
    norm_fn = compile_hca_norm_rope_scatter_gfx1250(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant=quant,
        quant_group_size=quant_group_size,
    )
    # k_rope_buff is referenced only on the quant path; pass kv_cache as a dummy
    # (valid tensor, never read) when bf16 so the launcher arity stays fixed.
    if quant:
        krope_buf = k_rope_cache
        krope_bs = int(k_rope_cache.stride(0))
        krope_ts = int(k_rope_cache.stride(1))
    else:
        krope_buf = kv_cache
        krope_bs = 0
        krope_ts = 0
    norm_args = (
        kv_compressed,
        int(kv_compressed.stride(0)),
        plan_gpu,
        rms_weight,
        cos_cache,
        sin_cache,
        kv_cache,
        int(kv_cache.stride(0)),
        int(kv_cache.stride(1)),
        block_tables,
        int(block_tables.stride(0)),
        krope_buf,
        krope_bs,
        krope_ts,
        int(plan_capacity),
        stream_obj,
    )
    _run_compiled(norm_fn, *norm_args)
