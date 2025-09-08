    .amdgcn_target "amdgcn-amd-amdhsa--gfx906"
    
    .text
    .globl gfx906_memcpy
    .p2align 8
    .type gfx906_memcpy,@function
gfx906_memcpy:
    ; Optimized memory copy for AMD Instinct MI50 (GFX906)
    ; Achieves ~809 GB/s (79% of 1024 GB/s theoretical peak)
    ; 
    ; SGPR allocation (with private_segment_buffer enabled):
    ; s[0:3] = private_segment_buffer
    ; s[4:5] = kernarg_segment_ptr
    ; s6 = workgroup_id_x
    
    ; Load kernel arguments
    s_load_dwordx2 s[8:9], s[4:5], 0x0    ; dst pointer
    s_load_dwordx2 s[10:11], s[4:5], 0x8  ; src pointer
    s_load_dword s12, s[4:5], 0x10        ; n_bytes
    s_waitcnt lgkmcnt(0)
    
    ; Calculate global thread ID
    v_mov_b32_e32 v1, s6                  ; v1 = workgroup_id
    v_lshlrev_b32_e32 v1, 8, v1           ; v1 = workgroup_id * 256
    v_add_u32_e32 v0, v0, v1              ; v0 = global thread id
    
    ; Each thread processes 16 bytes (1 x dwordx4)
    v_lshlrev_b32_e32 v1, 4, v0           ; v1 = tid * 16 (byte offset)
    
    ; Check bounds
    v_add_u32_e32 v2, 16, v1              ; v2 = offset + 16
    v_mov_b32_e32 v3, s12                 ; v3 = n_bytes
    v_cmp_le_u32_e32 vcc, v2, v3          ; vcc = (offset + 16 <= n_bytes)
    s_and_saveexec_b64 s[14:15], vcc      ; Skip if out of bounds
    s_cbranch_execz .Lexit
    
    ; Calculate source address = src + offset
    v_mov_b32_e32 v2, s10                 ; v2 = src_lo
    v_mov_b32_e32 v3, s11                 ; v3 = src_hi
    v_add_co_u32_e32 v2, vcc, v2, v1      ; v2 = src_lo + offset
    v_mov_b32_e32 v4, 0
    v_addc_co_u32_e32 v3, vcc, v3, v4, vcc ; v3 = src_hi + carry
    
    ; Load 16 bytes (without cache bypass for better performance)
    global_load_dwordx4 v[4:7], v[2:3], off
    
    ; Calculate destination address = dst + offset
    v_mov_b32_e32 v8, s8                  ; v8 = dst_lo
    v_mov_b32_e32 v9, s9                  ; v9 = dst_hi
    v_add_co_u32_e32 v8, vcc, v8, v1      ; v8 = dst_lo + offset
    v_mov_b32_e32 v10, 0
    v_addc_co_u32_e32 v9, vcc, v9, v10, vcc ; v9 = dst_hi + carry
    
    ; Wait for load to complete
    s_waitcnt vmcnt(0)
    
    ; Store 16 bytes
    global_store_dwordx4 v[8:9], v[4:7], off
    
.Lexit:
    ; Ensure all stores complete
    s_waitcnt vmcnt(0)
    s_endpgm
.Lfunc_end0:
    .size gfx906_memcpy, .Lfunc_end0-gfx906_memcpy
    
    .rodata
    .p2align 6
    .amdhsa_kernel gfx906_memcpy
        .amdhsa_group_segment_fixed_size 0
        .amdhsa_private_segment_fixed_size 0
        .amdhsa_kernarg_size 24
        .amdhsa_user_sgpr_private_segment_buffer 1
        .amdhsa_user_sgpr_dispatch_ptr 0
        .amdhsa_user_sgpr_queue_ptr 0
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_user_sgpr_dispatch_id 0
        .amdhsa_user_sgpr_flat_scratch_init 0
        .amdhsa_user_sgpr_private_segment_size 0
        .amdhsa_system_sgpr_private_segment_wavefront_offset 0
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 0
        .amdhsa_system_sgpr_workgroup_id_z 0
        .amdhsa_system_sgpr_workgroup_info 0
        .amdhsa_system_vgpr_workitem_id 0
        .amdhsa_next_free_vgpr 11
        .amdhsa_next_free_sgpr 16
        .amdhsa_reserve_vcc 1
        .amdhsa_reserve_flat_scratch 0
        .amdhsa_reserve_xnack_mask 1
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_dx10_clamp 1
        .amdhsa_ieee_mode 1
        .amdhsa_exception_fp_ieee_invalid_op 0
        .amdhsa_exception_fp_denorm_src 0
        .amdhsa_exception_fp_ieee_div_zero 0
        .amdhsa_exception_fp_ieee_overflow 0
        .amdhsa_exception_fp_ieee_underflow 0
        .amdhsa_exception_fp_ieee_inexact 0
        .amdhsa_exception_int_div_zero 0
    .end_amdhsa_kernel
    
    .amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: gfx906_memcpy
    .symbol: gfx906_memcpy.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 11
    .max_flat_workgroup_size: 256
    .args:
      - .name: dst
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .value_type: i8
        .address_space: global
        .is_const: false
      - .name: src
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .value_type: i8
        .address_space: global
        .is_const: true
      - .name: n_bytes
        .size: 4
        .offset: 16
        .value_kind: by_value
        .value_type: u32
...
    .end_amdgpu_metadata