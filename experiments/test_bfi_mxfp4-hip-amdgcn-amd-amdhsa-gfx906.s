	.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi ; -- Begin function _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi
	.globl	_Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi
	.p2align	8
	.type	_Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi,@function
_Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi: ; @_Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dword s1, s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s1, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	s_mov_b32 s4, 0xff00
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s1
	v_add_co_u32_e32 v2, vcc, s0, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v2, v[2:3], off
	s_mov_b32 s0, 0xc080604
	v_mov_b32_e32 v3, 0x3020100
	v_mov_b32_e32 v5, s3
	v_add_co_u32_e32 v0, vcc, s2, v0
	s_mov_b32 s1, 0xf4f8fafc
	v_mov_b32_e32 v4, 0xfdfeff00
	v_addc_co_u32_e32 v1, vcc, v5, v1, vcc
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v6, 4, v2
	v_and_b32_e32 v7, 0x7070707, v2
	v_lshrrev_b32_e32 v8, 3, v2
	v_lshrrev_b32_e32 v2, 7, v2
	v_and_b32_e32 v6, 0x7070707, v6
	v_perm_b32 v9, s0, v3, v7
	v_and_b32_e32 v8, 0x1010101, v8
	v_and_b32_e32 v2, 0x1010101, v2
	v_perm_b32 v3, s0, v3, v6
	v_perm_b32 v7, s1, v4, v7
	v_perm_b32 v4, s1, v4, v6
	v_perm_b32 v6, s4, s4, v8
	v_perm_b32 v8, s4, s4, v2
	;;#ASMSTART
	v_bfi_b32 v2, v6, v7, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v3, v8, v4, v3
	;;#ASMEND
	global_store_dwordx2 v[0:1], v[2:3], off
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 7
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi, .Lfunc_end0-_Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi
                                        ; -- End function
	.set _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.num_vgpr, 10
	.set _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.num_agpr, 0
	.set _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.numbered_sgpr, 7
	.set _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.private_seg_size, 0
	.set _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.uses_vcc, 1
	.set _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.uses_flat_scratch, 0
	.set _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.has_dyn_sized_stack, 0
	.set _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.has_recursion, 0
	.set _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 276
; TotalNumSgprs: 11
; NumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 10
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi ; -- Begin function _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi
	.globl	_Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi
	.p2align	8
	.type	_Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi,@function
_Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi: ; @_Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dword s1, s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s1, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s1
	v_add_co_u32_e32 v2, vcc, s0, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v2, v[2:3], off
	s_mov_b32 s0, 0xc080604
	v_mov_b32_e32 v3, 0x3020100
	s_mov_b32 s1, 0xf4f8fafc
	v_mov_b32_e32 v4, 0xfdfeff00
	v_mov_b32_e32 v5, s3
	v_add_co_u32_e32 v0, vcc, s2, v0
	v_addc_co_u32_e32 v1, vcc, v5, v1, vcc
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v6, 4, v2
	v_and_b32_e32 v7, 0x7070707, v2
	v_lshrrev_b32_e32 v8, 1, v2
	v_lshrrev_b32_e32 v2, 5, v2
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v2, 0x4040404, v2
	v_perm_b32 v9, s0, v3, v7
	v_perm_b32 v7, s1, v4, v7
	v_perm_b32 v3, s0, v3, v6
	v_perm_b32 v4, s1, v4, v6
	v_or_b32_e32 v6, 0x3020100, v8
	v_or_b32_e32 v8, 0x3020100, v2
	v_perm_b32 v2, v7, v9, v6
	v_perm_b32 v3, v4, v3, v8
	global_store_dwordx2 v[0:1], v[2:3], off
.LBB1_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 7
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	_Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi, .Lfunc_end1-_Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi
                                        ; -- End function
	.set _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.num_vgpr, 10
	.set _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.num_agpr, 0
	.set _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.numbered_sgpr, 7
	.set _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.private_seg_size, 0
	.set _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.uses_vcc, 1
	.set _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.uses_flat_scratch, 0
	.set _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.has_dyn_sized_stack, 0
	.set _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.has_recursion, 0
	.set _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 268
; TotalNumSgprs: 11
; NumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 10
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_349aa7714f3c5371,@object ; @__hip_cuid_349aa7714f3c5371
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_349aa7714f3c5371
__hip_cuid_349aa7714f3c5371:
	.byte	0                               ; 0x0
	.size	__hip_cuid_349aa7714f3c5371, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_349aa7714f3c5371
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z8test_bfiPKjP15HIP_vector_typeIiLj2EEi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z13test_baselinePKjP15HIP_vector_typeIiLj2EEi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
