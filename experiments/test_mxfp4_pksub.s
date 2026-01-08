
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx906
	.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i ; -- Begin function _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.globl	_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.p2align	8
	.type	_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i,@function
_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i: ; @_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
; %bb.0:
	s_load_dword s0, s[4:5], 0x3c
	s_load_dword s1, s[4:5], 0x28
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s1, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, kvalues_mxfp4@rel32@lo+4
	s_addc_u32 s1, s1, kvalues_mxfp4@rel32@hi+12
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s9
	v_add_co_u32_e32 v2, vcc, s8, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v6, v[2:3], off
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	s_mov_b32 s6, 0xc080604
	v_mov_b32_e32 v7, 0x3020100
	v_mov_b32_e32 v8, s0
	v_mov_b32_e32 v9, s2
	v_mov_b32_e32 v3, s11
	v_add_co_u32_e32 v2, vcc, s10, v0
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
	s_mov_b32 s7, 0x7020500
	s_mov_b32 s8, 0xff00
	v_mov_b32_e32 v5, s13
	v_add_co_u32_e32 v4, vcc, s12, v0
	v_addc_co_u32_e32 v5, vcc, v5, v1, vcc
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v10, 4, v6
	v_lshrrev_b32_e32 v12, 1, v6
	v_lshrrev_b32_e32 v13, 5, v6
	v_and_b32_e32 v11, 0x7070707, v6
	v_lshrrev_b32_e32 v14, 3, v6
	v_lshrrev_b32_e32 v15, 7, v6
	v_lshrrev_b32_e32 v16, 11, v6
	v_lshrrev_b32_e32 v6, 15, v6
	v_and_b32_e32 v10, 0x7070707, v10
	v_and_b32_e32 v12, 0x4040404, v12
	v_and_b32_e32 v13, 0x4040404, v13
	v_perm_b32 v17, s1, v8, v11
	v_perm_b32 v18, s3, v9, v11
	v_perm_b32 v11, s6, v7, v11
	v_perm_b32 v8, s1, v8, v10
	v_perm_b32 v9, s3, v9, v10
	v_perm_b32 v10, s6, v7, v10
	v_and_b32_e32 v21, 0x10001, v6
	v_or_b32_e32 v6, 0x3020100, v12
	v_or_b32_e32 v7, 0x3020100, v13
	v_perm_b32 v6, v18, v17, v6
	v_perm_b32 v7, v9, v8, v7
	v_lshrrev_b32_e32 v8, 8, v11
	global_store_dwordx2 v[2:3], v[6:7], off
	v_lshrrev_b32_e32 v2, 8, v10
	v_and_b32_e32 v12, 0xff00ff, v11
	v_and_b32_e32 v8, 0xff00ff, v8
	v_and_b32_e32 v6, 0xff00ff, v2
	;;#ASMSTART
	v_pk_sub_u16 v2, 0, v12
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v3, 0, v8
	;;#ASMEND
	v_lshlrev_b32_e32 v3, 8, v3
	v_and_b32_e32 v18, 0xff00ff, v10
	;;#ASMSTART
	v_pk_sub_u16 v7, 0, v18
	;;#ASMEND
	v_perm_b32 v2, v3, v2, s7
	;;#ASMSTART
	v_pk_sub_u16 v3, 0, v6
	;;#ASMEND
	v_lshlrev_b32_e32 v3, 8, v3
	v_and_b32_e32 v19, 0x1010101, v14
	v_and_b32_e32 v20, 0x1010101, v15
	v_perm_b32 v3, v3, v7, s7
	v_perm_b32 v9, s8, s8, v19
	v_perm_b32 v13, s8, s8, v20
	;;#ASMSTART
	v_bfi_b32 v2, v9, v2, v11
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v3, v13, v3, v10
	;;#ASMEND
	global_store_dwordx2 v[4:5], v[2:3], off
	;;#ASMSTART
	v_pk_sub_u16 v2, 0, v12
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v3, 0, v8
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v4, 0, v18
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v5, 0, v6
	;;#ASMEND
	v_and_b32_e32 v14, 0x10001, v14
	v_and_b32_e32 v16, 0x10001, v16
	v_lshlrev_b32_e32 v3, 8, v3
	v_lshlrev_b32_e32 v5, 8, v5
	v_lshlrev_b32_e32 v17, 8, v14
	v_lshlrev_b32_e32 v7, 8, v16
	v_perm_b32 v2, v3, v2, s7
	;;#ASMSTART
	v_pk_sub_u16 v3, v17, v14
	;;#ASMEND
	v_perm_b32 v4, v5, v4, s7
	;;#ASMSTART
	v_pk_sub_u16 v5, v7, v16
	;;#ASMEND
	v_and_b32_e32 v15, 0x10001, v15
	v_lshlrev_b32_e32 v5, 8, v5
	v_lshlrev_b32_e32 v9, 8, v15
	v_lshlrev_b32_e32 v13, 8, v21
	v_perm_b32 v3, v5, v3, s7
	;;#ASMSTART
	v_pk_sub_u16 v5, v9, v15
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v19, v13, v21
	;;#ASMEND
	v_lshlrev_b32_e32 v19, 8, v19
	v_perm_b32 v5, v19, v5, s7
	;;#ASMSTART
	v_bfi_b32 v2, v3, v2, v11
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v3, v5, v4, v10
	;;#ASMEND
	v_mov_b32_e32 v5, s15
	v_add_co_u32_e32 v4, vcc, s14, v0
	v_addc_co_u32_e32 v5, vcc, v5, v1, vcc
	global_store_dwordx2 v[4:5], v[2:3], off
	;;#ASMSTART
	v_pk_sub_u16 v2, 0, v12
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v3, 0, v8
	;;#ASMEND
	s_load_dwordx2 s[0:1], s[4:5], 0x20
	v_lshlrev_b32_e32 v3, 8, v3
	;;#ASMSTART
	v_pk_sub_u16 v4, 0, v18
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v5, 0, v6
	;;#ASMEND
	v_perm_b32 v2, v3, v2, s7
	v_lshlrev_b32_e32 v3, 8, v5
	v_perm_b32 v3, v3, v4, s7
	;;#ASMSTART
	v_pk_sub_u16 v4, v17, v14
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v5, v7, v16
	;;#ASMEND
	v_lshlrev_b32_e32 v5, 8, v5
	v_perm_b32 v4, v5, v4, s7
	;;#ASMSTART
	v_pk_sub_u16 v6, v9, v15
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v7, v13, v21
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v2, v4, v2, v11
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s1
	v_add_co_u32_e32 v0, vcc, s0, v0
	v_lshlrev_b32_e32 v5, 8, v7
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
	v_perm_b32 v5, v5, v6, s7
	;;#ASMSTART
	v_bfi_b32 v3, v5, v3, v10
	;;#ASMEND
	global_store_dwordx2 v[0:1], v[2:3], off
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 304
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
		.amdhsa_next_free_vgpr 22
		.amdhsa_next_free_sgpr 16
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
	.size	_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i, .Lfunc_end0-_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
                                        ; -- End function
	.set _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.num_vgpr, 22
	.set _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.num_agpr, 0
	.set _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.numbered_sgpr, 16
	.set _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.private_seg_size, 0
	.set _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.uses_vcc, 1
	.set _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.uses_flat_scratch, 0
	.set _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.has_dyn_sized_stack, 0
	.set _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.has_recursion, 0
	.set _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 856
; TotalNumSgprs: 20
; NumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 22
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
	.protected	_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii ; -- Begin function _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii
	.globl	_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii
	.p2align	8
	.type	_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii,@function
_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii: ; @_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_5
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	s_mov_b32 s8, 0
	s_cmp_lt_i32 s9, 1
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	s_cbranch_scc1 .LBB1_4
; %bb.2:
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s1
	v_add_co_u32_e32 v2, vcc, s0, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v4, v[2:3], off
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, kvalues_mxfp4@rel32@lo+4
	s_addc_u32 s1, s1, kvalues_mxfp4@rel32@hi+12
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, s4
	v_mov_b32_e32 v6, s6
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt vmcnt(0)
	v_xor_b32_e32 v7, s8, v4
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v9, 0x7070707, v7
	v_lshrrev_b32_e32 v10, 1, v7
	v_lshrrev_b32_e32 v7, 5, v7
	v_and_b32_e32 v8, 0x7070707, v8
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v7, 0x4040404, v7
	v_perm_b32 v11, s5, v5, v9
	v_perm_b32 v9, s7, v6, v9
	v_perm_b32 v12, s5, v5, v8
	v_perm_b32 v8, s7, v6, v8
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v7, 0x3020100, v7
	s_add_i32 s8, s8, 1
	v_perm_b32 v9, v9, v11, v10
	v_perm_b32 v7, v8, v12, v7
	s_cmp_eq_u32 s9, s8
	v_add_u32_e32 v2, v9, v2
	v_add_u32_e32 v3, v7, v3
	s_cbranch_scc0 .LBB1_3
.LBB1_4:
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s3
	v_add_co_u32_e32 v0, vcc, s2, v0
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
	global_store_dwordx2 v[0:1], v[2:3], off
.LBB1_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii
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
		.amdhsa_next_free_vgpr 13
		.amdhsa_next_free_sgpr 10
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
	.size	_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii, .Lfunc_end1-_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii
                                        ; -- End function
	.set _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.num_vgpr, 13
	.set _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.num_agpr, 0
	.set _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.numbered_sgpr, 10
	.set _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.private_seg_size, 0
	.set _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.uses_vcc, 1
	.set _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.uses_flat_scratch, 0
	.set _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.has_dyn_sized_stack, 0
	.set _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.has_recursion, 0
	.set _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 332
; TotalNumSgprs: 14
; NumVgprs: 13
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 13
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
	.protected	_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii ; -- Begin function _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.globl	_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.p2align	8
	.type	_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii,@function
_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii: ; @_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_5
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	s_mov_b32 s4, 0
	s_cmp_lt_i32 s9, 1
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	s_cbranch_scc1 .LBB2_4
; %bb.2:
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s1
	v_add_co_u32_e32 v2, vcc, s0, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v4, v[2:3], off
	v_mov_b32_e32 v3, 0
	s_mov_b32 s0, 0xc080604
	v_mov_b32_e32 v5, 0x3020100
	s_mov_b32 s1, 0x7020500
	s_mov_b32 s5, 0xff00
	v_mov_b32_e32 v2, 0
.LBB2_3:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt vmcnt(0)
	v_xor_b32_e32 v6, s4, v4
	v_and_b32_e32 v8, 0x7070707, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_perm_b32 v8, s0, v5, v8
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 8, v8
	v_perm_b32 v7, s0, v5, v7
	v_and_b32_e32 v10, 0xff00ff, v8
	v_and_b32_e32 v11, 0xff00ff, v11
	v_lshrrev_b32_e32 v9, 3, v6
	v_lshrrev_b32_e32 v6, 7, v6
	v_lshrrev_b32_e32 v13, 8, v7
	;;#ASMSTART
	v_pk_sub_u16 v10, 0, v10
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v11, 0, v11
	;;#ASMEND
	v_and_b32_e32 v6, 0x1010101, v6
	v_and_b32_e32 v12, 0xff00ff, v7
	v_and_b32_e32 v13, 0xff00ff, v13
	v_lshlrev_b32_e32 v11, 8, v11
	s_add_i32 s4, s4, 1
	v_and_b32_e32 v9, 0x1010101, v9
	v_perm_b32 v6, s5, s5, v6
	;;#ASMSTART
	v_pk_sub_u16 v12, 0, v12
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v13, 0, v13
	;;#ASMEND
	v_perm_b32 v10, v11, v10, s1
	v_lshlrev_b32_e32 v11, 8, v13
	s_cmp_eq_u32 s9, s4
	v_perm_b32 v9, s5, s5, v9
	v_perm_b32 v11, v11, v12, s1
	;;#ASMSTART
	v_bfi_b32 v8, v9, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v6, v6, v11, v7
	;;#ASMEND
	v_add_u32_e32 v2, v8, v2
	v_add_u32_e32 v3, v6, v3
	s_cbranch_scc0 .LBB2_3
.LBB2_4:
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s3
	v_add_co_u32_e32 v0, vcc, s2, v0
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
	global_store_dwordx2 v[0:1], v[2:3], off
.LBB2_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii
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
		.amdhsa_next_free_vgpr 14
		.amdhsa_next_free_sgpr 10
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
.Lfunc_end2:
	.size	_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii, .Lfunc_end2-_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii
                                        ; -- End function
	.set _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.num_vgpr, 14
	.set _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.num_agpr, 0
	.set _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.numbered_sgpr, 10
	.set _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.private_seg_size, 0
	.set _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.uses_vcc, 1
	.set _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.uses_flat_scratch, 0
	.set _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.has_dyn_sized_stack, 0
	.set _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.has_recursion, 0
	.set _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 404
; TotalNumSgprs: 14
; NumVgprs: 14
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 14
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
	.protected	_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii ; -- Begin function _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii
	.globl	_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii
	.p2align	8
	.type	_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii,@function
_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii: ; @_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB3_5
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	s_mov_b32 s4, 0
	s_cmp_lt_i32 s9, 1
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	s_cbranch_scc1 .LBB3_4
; %bb.2:
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s1
	v_add_co_u32_e32 v2, vcc, s0, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v4, v[2:3], off
	v_mov_b32_e32 v3, 0
	s_mov_b32 s0, 0xc080604
	v_mov_b32_e32 v5, 0x3020100
	s_mov_b32 s1, 0x7020500
	v_mov_b32_e32 v2, 0
.LBB3_3:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt vmcnt(0)
	v_xor_b32_e32 v6, s4, v4
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_perm_b32 v8, s0, v5, v8
	v_perm_b32 v7, s0, v5, v7
	v_lshrrev_b32_e32 v9, 3, v6
	v_lshrrev_b32_e32 v10, 7, v6
	v_lshrrev_b32_e32 v11, 11, v6
	v_lshrrev_b32_e32 v6, 15, v6
	v_lshrrev_b32_e32 v13, 8, v8
	v_lshrrev_b32_e32 v19, 8, v7
	v_and_b32_e32 v9, 0x10001, v9
	v_and_b32_e32 v11, 0x10001, v11
	v_and_b32_e32 v10, 0x10001, v10
	v_and_b32_e32 v6, 0x10001, v6
	v_and_b32_e32 v12, 0xff00ff, v8
	v_and_b32_e32 v13, 0xff00ff, v13
	v_and_b32_e32 v18, 0xff00ff, v7
	v_and_b32_e32 v19, 0xff00ff, v19
	v_lshlrev_b32_e32 v14, 8, v9
	v_lshlrev_b32_e32 v15, 8, v11
	v_lshlrev_b32_e32 v16, 8, v10
	v_lshlrev_b32_e32 v17, 8, v6
	;;#ASMSTART
	v_pk_sub_u16 v12, 0, v12
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v13, 0, v13
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v18, 0, v18
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v19, 0, v19
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v9, v14, v9
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v11, v15, v11
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v10, v16, v10
	;;#ASMEND
	;;#ASMSTART
	v_pk_sub_u16 v6, v17, v6
	;;#ASMEND
	v_lshlrev_b32_e32 v13, 8, v13
	v_lshlrev_b32_e32 v6, 8, v6
	s_add_i32 s4, s4, 1
	v_perm_b32 v12, v13, v12, s1
	v_lshlrev_b32_e32 v13, 8, v19
	v_lshlrev_b32_e32 v11, 8, v11
	v_perm_b32 v6, v6, v10, s1
	s_cmp_eq_u32 s9, s4
	v_perm_b32 v13, v13, v18, s1
	v_perm_b32 v9, v11, v9, s1
	;;#ASMSTART
	v_bfi_b32 v8, v9, v12, v8
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v6, v6, v13, v7
	;;#ASMEND
	v_add_u32_e32 v2, v8, v2
	v_add_u32_e32 v3, v6, v3
	s_cbranch_scc0 .LBB3_3
.LBB3_4:
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s3
	v_add_co_u32_e32 v0, vcc, s2, v0
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
	global_store_dwordx2 v[0:1], v[2:3], off
.LBB3_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii
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
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 10
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
.Lfunc_end3:
	.size	_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii, .Lfunc_end3-_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii
                                        ; -- End function
	.set _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.num_vgpr, 20
	.set _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.num_agpr, 0
	.set _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.numbered_sgpr, 10
	.set _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.private_seg_size, 0
	.set _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.uses_vcc, 1
	.set _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.uses_flat_scratch, 0
	.set _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.has_dyn_sized_stack, 0
	.set _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.has_recursion, 0
	.set _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 476
; TotalNumSgprs: 14
; NumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 20
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
	.protected	kvalues_mxfp4           ; @kvalues_mxfp4
	.type	kvalues_mxfp4,@object
	.section	.rodata,"a",@progbits
	.globl	kvalues_mxfp4
	.p2align	4, 0x0
kvalues_mxfp4:
	.ascii	"\000\001\002\003\004\006\b\f\000\377\376\375\374\372\370\364"
	.size	kvalues_mxfp4, 16

	.type	__hip_cuid_e3f2f33f74ed7331,@object ; @__hip_cuid_e3f2f33f74ed7331
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e3f2f33f74ed7331
__hip_cuid_e3f2f33f74ed7331:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e3f2f33f74ed7331, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym kvalues_mxfp4
	.addrsig_sym __hip_cuid_e3f2f33f74ed7331
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
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         52
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         60
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         62
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         64
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         66
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         68
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         70
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         112
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 304
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     22
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
      - .offset:         20
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
    .name:           _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     13
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
      - .offset:         20
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
    .name:           _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     14
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
      - .offset:         20
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
    .name:           _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx906

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.file	"test_mxfp4_pksub.hip"
	.text
	.globl	_Z31__device_stub__test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i # -- Begin function _Z31__device_stub__test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.p2align	4
	.type	_Z31__device_stub__test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i,@function
_Z31__device_stub__test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i: # @_Z31__device_stub__test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	96(%rsp), %r9
	movl	$_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end0:
	.size	_Z31__device_stub__test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i, .Lfunc_end0-_Z31__device_stub__test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.cfi_endproc
                                        # -- End function
	.globl	_Z30__device_stub__bench_referencePjP15HIP_vector_typeIiLj2EEii # -- Begin function _Z30__device_stub__bench_referencePjP15HIP_vector_typeIiLj2EEii
	.p2align	4
	.type	_Z30__device_stub__bench_referencePjP15HIP_vector_typeIiLj2EEii,@function
_Z30__device_stub__bench_referencePjP15HIP_vector_typeIiLj2EEii: # @_Z30__device_stub__bench_referencePjP15HIP_vector_typeIiLj2EEii
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	movl	%ecx, 8(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end1:
	.size	_Z30__device_stub__bench_referencePjP15HIP_vector_typeIiLj2EEii, .Lfunc_end1-_Z30__device_stub__bench_referencePjP15HIP_vector_typeIiLj2EEii
	.cfi_endproc
                                        # -- End function
	.globl	_Z26__device_stub__bench_pksubPjP15HIP_vector_typeIiLj2EEii # -- Begin function _Z26__device_stub__bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.p2align	4
	.type	_Z26__device_stub__bench_pksubPjP15HIP_vector_typeIiLj2EEii,@function
_Z26__device_stub__bench_pksubPjP15HIP_vector_typeIiLj2EEii: # @_Z26__device_stub__bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	movl	%ecx, 8(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end2:
	.size	_Z26__device_stub__bench_pksubPjP15HIP_vector_typeIiLj2EEii, .Lfunc_end2-_Z26__device_stub__bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.cfi_endproc
                                        # -- End function
	.globl	_Z26__device_stub__bench_2permPjP15HIP_vector_typeIiLj2EEii # -- Begin function _Z26__device_stub__bench_2permPjP15HIP_vector_typeIiLj2EEii
	.p2align	4
	.type	_Z26__device_stub__bench_2permPjP15HIP_vector_typeIiLj2EEii,@function
_Z26__device_stub__bench_2permPjP15HIP_vector_typeIiLj2EEii: # @_Z26__device_stub__bench_2permPjP15HIP_vector_typeIiLj2EEii
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	movl	%ecx, 8(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end3:
	.size	_Z26__device_stub__bench_2permPjP15HIP_vector_typeIiLj2EEii, .Lfunc_end3-_Z26__device_stub__bench_2permPjP15HIP_vector_typeIiLj2EEii
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI4_0:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
.LCPI4_1:
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
.LCPI4_2:
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
.LCPI4_3:
	.long	12                              # 0xc
	.long	12                              # 0xc
	.long	12                              # 0xc
	.long	12                              # 0xc
.LCPI4_4:
	.long	16                              # 0x10
	.long	16                              # 0x10
	.long	16                              # 0x10
	.long	16                              # 0x10
.LCPI4_5:
	.long	20                              # 0x14
	.long	20                              # 0x14
	.long	20                              # 0x14
	.long	20                              # 0x14
.LCPI4_6:
	.long	24                              # 0x18
	.long	24                              # 0x18
	.long	24                              # 0x18
	.long	24                              # 0x18
.LCPI4_7:
	.long	28                              # 0x1c
	.long	28                              # 0x1c
	.long	28                              # 0x1c
	.long	28                              # 0x1c
.LCPI4_8:
	.long	32                              # 0x20
	.long	32                              # 0x20
	.long	32                              # 0x20
	.long	32                              # 0x20
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$216, %rsp
	.cfi_def_cfa_offset 272
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$.Lstr, %edi
	callq	puts@PLT
	movl	$262144, %edi                   # imm = 0x40000
	callq	_Znam
	movq	%rax, %rbx
	movl	$524288, %edi                   # imm = 0x80000
	callq	_Znam
	movq	%rax, %r14
	movl	$524288, %edi                   # imm = 0x80000
	callq	_Znam
	movq	%rax, %r12
	movl	$524288, %edi                   # imm = 0x80000
	callq	_Znam
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	movl	$524288, %edi                   # imm = 0x80000
	callq	_Znam
	movq	%rax, %r13
	movdqa	.LCPI4_0(%rip), %xmm0           # xmm0 = [0,1,2,3]
	movl	$28, %eax
	movdqa	.LCPI4_1(%rip), %xmm1           # xmm1 = [4,4,4,4]
	movdqa	.LCPI4_2(%rip), %xmm2           # xmm2 = [8,8,8,8]
	movdqa	.LCPI4_3(%rip), %xmm3           # xmm3 = [12,12,12,12]
	movdqa	.LCPI4_4(%rip), %xmm4           # xmm4 = [16,16,16,16]
	movdqa	.LCPI4_5(%rip), %xmm5           # xmm5 = [20,20,20,20]
	movdqa	.LCPI4_6(%rip), %xmm6           # xmm6 = [24,24,24,24]
	movdqa	.LCPI4_7(%rip), %xmm7           # xmm7 = [28,28,28,28]
	movdqa	.LCPI4_8(%rip), %xmm8           # xmm8 = [32,32,32,32]
	.p2align	4
.LBB4_1:                                # =>This Inner Loop Header: Depth=1
	movdqa	%xmm0, %xmm9
	paddd	%xmm1, %xmm9
	movdqu	%xmm0, -112(%rbx,%rax,4)
	movdqu	%xmm9, -96(%rbx,%rax,4)
	movdqa	%xmm0, %xmm9
	paddd	%xmm2, %xmm9
	movdqa	%xmm0, %xmm10
	paddd	%xmm3, %xmm10
	movdqu	%xmm9, -80(%rbx,%rax,4)
	movdqu	%xmm10, -64(%rbx,%rax,4)
	movdqa	%xmm0, %xmm9
	paddd	%xmm4, %xmm9
	movdqa	%xmm0, %xmm10
	paddd	%xmm5, %xmm10
	movdqu	%xmm9, -48(%rbx,%rax,4)
	movdqu	%xmm10, -32(%rbx,%rax,4)
	movdqa	%xmm0, %xmm9
	paddd	%xmm6, %xmm9
	movdqa	%xmm0, %xmm10
	paddd	%xmm7, %xmm10
	movdqu	%xmm9, -16(%rbx,%rax,4)
	movdqu	%xmm10, (%rbx,%rax,4)
	paddd	%xmm8, %xmm0
	addq	$32, %rax
	cmpq	$65564, %rax                    # imm = 0x1001C
	jne	.LBB4_1
# %bb.2:
	movabsq	$4294967552, %r15               # imm = 0x100000100
	leaq	96(%rsp), %rdi
	movl	$262144, %esi                   # imm = 0x40000
	callq	hipMalloc
	leaq	120(%rsp), %rdi
	movl	$524288, %esi                   # imm = 0x80000
	callq	hipMalloc
	leaq	192(%rsp), %rdi
	movl	$524288, %esi                   # imm = 0x80000
	callq	hipMalloc
	leaq	200(%rsp), %rdi
	movl	$524288, %esi                   # imm = 0x80000
	callq	hipMalloc
	leaq	184(%rsp), %rdi
	movl	$524288, %esi                   # imm = 0x80000
	callq	hipMalloc
	movq	96(%rsp), %rdi
	movl	$262144, %edx                   # imm = 0x40000
	movq	%rbx, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	movq	%r15, %rdi
	movl	$1, %esi
	movq	%r15, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB4_4
# %bb.3:
	movq	96(%rsp), %rax
	movq	120(%rsp), %rcx
	movq	192(%rsp), %rdx
	movq	200(%rsp), %rsi
	movq	184(%rsp), %rdi
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movq	%rdx, 40(%rsp)
	movq	%rsi, 32(%rsp)
	movq	%rdi, 24(%rsp)
	movl	$65536, 4(%rsp)                 # imm = 0x10000
	leaq	88(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	40(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	32(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	64(%rsp), %rdi
	leaq	48(%rsp), %rsi
	leaq	8(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	64(%rsp), %rsi
	movl	72(%rsp), %edx
	movq	48(%rsp), %rcx
	movl	56(%rsp), %r8d
	leaq	128(%rsp), %r9
	movl	$_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB4_4:
	callq	hipDeviceSynchronize
	movq	120(%rsp), %rsi
	movl	$524288, %edx                   # imm = 0x80000
	movq	%r14, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
	movq	192(%rsp), %rsi
	movl	$524288, %edx                   # imm = 0x80000
	movq	%r12, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
	movq	200(%rsp), %rsi
	movl	$524288, %edx                   # imm = 0x80000
	movq	112(%rsp), %rdi                 # 8-byte Reload
	movl	$2, %ecx
	callq	hipMemcpy
	movq	184(%rsp), %rsi
	movl	$524288, %edx                   # imm = 0x80000
	movq	%r13, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
	movl	$.Lstr.14, %edi
	callq	puts@PLT
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
	.p2align	4
.LBB4_5:                                # =>This Inner Loop Header: Depth=1
	movl	(%r14,%r15,8), %ecx
	movl	(%r12,%r15,8), %r9d
	cmpl	%r9d, %ecx
	jne	.LBB4_7
# %bb.6:                                #   in Loop: Header=BB4_5 Depth=1
	movl	4(%r14,%r15,8), %eax
	cmpl	4(%r12,%r15,8), %eax
	je	.LBB4_10
.LBB4_7:                                #   in Loop: Header=BB4_5 Depth=1
	testl	%ebp, %ebp
	jne	.LBB4_9
# %bb.8:                                #   in Loop: Header=BB4_5 Depth=1
	movl	$.L.str.10, %edi
	movl	$.L.str.2, %esi
	xorl	%eax, %eax
	callq	printf
	movl	(%r14,%r15,8), %ecx
	movl	(%r12,%r15,8), %r9d
.LBB4_9:                                #   in Loop: Header=BB4_5 Depth=1
	movl	(%rbx,%r15,4), %edx
	movl	4(%r14,%r15,8), %r8d
	movl	4(%r12,%r15,8), %r10d
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movl	$.L.str.11, %edi
	movl	%r15d, %esi
	xorl	%eax, %eax
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	callq	printf
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	incl	%ebp
.LBB4_10:                               #   in Loop: Header=BB4_5 Depth=1
	cmpq	$65534, %r15                    # imm = 0xFFFE
	ja	.LBB4_12
# %bb.11:                               #   in Loop: Header=BB4_5 Depth=1
	incq	%r15
	cmpl	$3, %ebp
	jl	.LBB4_5
.LBB4_12:
	testl	%ebp, %ebp
	movq	%r12, 208(%rsp)                 # 8-byte Spill
	je	.LBB4_13
# %bb.14:
	movl	$.L.str.13, %edi
	movl	$.L.str.2, %esi
	movl	%ebp, %edx
	xorl	%eax, %eax
	callq	printf
	jmp	.LBB4_15
.LBB4_13:
	movl	$.L.str.12, %edi
	movl	$.L.str.2, %esi
	xorl	%eax, %eax
	callq	printf
.LBB4_15:
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
	movq	112(%rsp), %rsi                 # 8-byte Reload
	.p2align	4
.LBB4_16:                               # =>This Inner Loop Header: Depth=1
	movl	(%r14,%r12,8), %ecx
	movl	(%rsi,%r12,8), %r9d
	cmpl	%r9d, %ecx
	jne	.LBB4_18
# %bb.17:                               #   in Loop: Header=BB4_16 Depth=1
	movl	4(%r14,%r12,8), %eax
	cmpl	4(%rsi,%r12,8), %eax
	je	.LBB4_21
.LBB4_18:                               #   in Loop: Header=BB4_16 Depth=1
	testl	%r15d, %r15d
	jne	.LBB4_20
# %bb.19:                               #   in Loop: Header=BB4_16 Depth=1
	movl	$.L.str.10, %edi
	movl	$.L.str.3, %esi
	xorl	%eax, %eax
	callq	printf
	movq	112(%rsp), %rsi                 # 8-byte Reload
	movl	(%r14,%r12,8), %ecx
	movl	(%rsi,%r12,8), %r9d
.LBB4_20:                               #   in Loop: Header=BB4_16 Depth=1
	movl	(%rbx,%r12,4), %edx
	movl	4(%r14,%r12,8), %r8d
	movl	4(%rsi,%r12,8), %r10d
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movl	$.L.str.11, %edi
	movl	%r12d, %esi
	xorl	%eax, %eax
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	callq	printf
	movq	128(%rsp), %rsi                 # 8-byte Reload
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	incl	%r15d
.LBB4_21:                               #   in Loop: Header=BB4_16 Depth=1
	cmpq	$65534, %r12                    # imm = 0xFFFE
	ja	.LBB4_23
# %bb.22:                               #   in Loop: Header=BB4_16 Depth=1
	incq	%r12
	cmpl	$3, %r15d
	jl	.LBB4_16
.LBB4_23:
	testl	%r15d, %r15d
	je	.LBB4_24
# %bb.25:
	movl	$.L.str.13, %edi
	movl	$.L.str.3, %esi
	movl	%r15d, %edx
	xorl	%eax, %eax
	callq	printf
	jmp	.LBB4_26
.LBB4_24:
	movl	$.L.str.12, %edi
	movl	$.L.str.3, %esi
	xorl	%eax, %eax
	callq	printf
.LBB4_26:
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
	.p2align	4
.LBB4_27:                               # =>This Inner Loop Header: Depth=1
	movl	(%r14,%r12,8), %ecx
	movl	(%r13,%r12,8), %r9d
	cmpl	%r9d, %ecx
	jne	.LBB4_29
# %bb.28:                               #   in Loop: Header=BB4_27 Depth=1
	movl	4(%r14,%r12,8), %eax
	cmpl	4(%r13,%r12,8), %eax
	je	.LBB4_32
.LBB4_29:                               #   in Loop: Header=BB4_27 Depth=1
	testl	%r15d, %r15d
	jne	.LBB4_31
# %bb.30:                               #   in Loop: Header=BB4_27 Depth=1
	movl	$.L.str.10, %edi
	movl	$.L.str.4, %esi
	xorl	%eax, %eax
	callq	printf
	movl	(%r14,%r12,8), %ecx
	movl	(%r13,%r12,8), %r9d
.LBB4_31:                               #   in Loop: Header=BB4_27 Depth=1
	movl	(%rbx,%r12,4), %edx
	movl	4(%r14,%r12,8), %r8d
	movl	4(%r13,%r12,8), %r10d
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movl	$.L.str.11, %edi
	movl	%r12d, %esi
	xorl	%eax, %eax
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	callq	printf
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	incl	%r15d
.LBB4_32:                               #   in Loop: Header=BB4_27 Depth=1
	cmpq	$65534, %r12                    # imm = 0xFFFE
	ja	.LBB4_34
# %bb.33:                               #   in Loop: Header=BB4_27 Depth=1
	incq	%r12
	cmpl	$3, %r15d
	jl	.LBB4_27
.LBB4_34:
	testl	%r15d, %r15d
	je	.LBB4_35
# %bb.36:
	movl	$.L.str.13, %edi
	movl	$.L.str.4, %esi
	movl	%r15d, %edx
	xorl	%eax, %eax
	callq	printf
	jmp	.LBB4_37
.LBB4_35:
	movl	$.L.str.12, %edi
	movl	$.L.str.4, %esi
	xorl	%eax, %eax
	callq	printf
.LBB4_37:
	movl	$.L.str.5, %edi
	movl	$65536, %esi                    # imm = 0x10000
	movl	$10000, %edx                    # imm = 0x2710
	xorl	%eax, %eax
	callq	printf
	leaq	24(%rsp), %rdi
	callq	hipEventCreate
	leaq	8(%rsp), %rdi
	callq	hipEventCreate
	movabsq	$4294967552, %r12               # imm = 0x100000100
	movq	%r12, %rdi
	movl	$1, %esi
	movq	%r12, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB4_39
# %bb.38:
	movq	96(%rsp), %rax
	movq	120(%rsp), %rcx
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movl	$65536, 16(%rsp)                # imm = 0x10000
	movl	$100, 4(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	64(%rsp), %rdi
	leaq	48(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	32(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	64(%rsp), %rsi
	movl	72(%rsp), %edx
	movq	48(%rsp), %rcx
	movl	56(%rsp), %r8d
	leaq	128(%rsp), %r9
	movl	$_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii, %edi
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB4_39:
	callq	hipDeviceSynchronize
	movq	24(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	movq	%r12, %rdi
	movl	$1, %esi
	movq	%r12, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB4_41
# %bb.40:
	movq	96(%rsp), %rax
	movq	120(%rsp), %rcx
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movl	$65536, 16(%rsp)                # imm = 0x10000
	movl	$10000, 4(%rsp)                 # imm = 0x2710
	leaq	88(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	64(%rsp), %rdi
	leaq	48(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	32(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	64(%rsp), %rsi
	movl	72(%rsp), %edx
	movq	48(%rsp), %rcx
	movl	56(%rsp), %r8d
	leaq	128(%rsp), %r9
	movl	$_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii, %edi
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB4_41:
	movq	8(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	movq	8(%rsp), %rdi
	callq	hipEventSynchronize
	movq	24(%rsp), %rsi
	movq	8(%rsp), %rdx
	leaq	108(%rsp), %rdi
	callq	hipEventElapsedTime
	movss	108(%rsp), %xmm0                # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str.6, %edi
	movb	$1, %al
	callq	printf
	testl	%ebp, %ebp
	movabsq	$4294967552, %rbp               # imm = 0x100000100
	je	.LBB4_42
# %bb.45:
	testl	%r15d, %r15d
	jne	.LBB4_49
	jmp	.LBB4_46
.LBB4_42:
	movq	24(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	movq	%rbp, %rdi
	movl	$1, %esi
	movq	%rbp, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB4_44
# %bb.43:
	movq	96(%rsp), %rax
	movq	192(%rsp), %rcx
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movl	$65536, 16(%rsp)                # imm = 0x10000
	movl	$10000, 4(%rsp)                 # imm = 0x2710
	leaq	88(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	64(%rsp), %rdi
	leaq	48(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	32(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	64(%rsp), %rsi
	movl	72(%rsp), %edx
	movq	48(%rsp), %rcx
	movl	56(%rsp), %r8d
	leaq	128(%rsp), %r9
	movl	$_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii, %edi
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB4_44:
	movq	8(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	movq	8(%rsp), %rdi
	callq	hipEventSynchronize
	movq	24(%rsp), %rsi
	movq	8(%rsp), %rdx
	leaq	128(%rsp), %rdi
	callq	hipEventElapsedTime
	movss	128(%rsp), %xmm1                # xmm1 = mem[0],zero,zero,zero
	xorps	%xmm0, %xmm0
	cvtss2sd	%xmm1, %xmm0
	movss	108(%rsp), %xmm2                # xmm2 = mem[0],zero,zero,zero
	divss	%xmm1, %xmm2
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm2, %xmm1
	movl	$.L.str.7, %edi
	movb	$2, %al
	callq	printf
	testl	%r15d, %r15d
	jne	.LBB4_49
.LBB4_46:
	movq	24(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	movq	%rbp, %rdi
	movl	$1, %esi
	movq	%rbp, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB4_48
# %bb.47:
	movq	96(%rsp), %rax
	movq	184(%rsp), %rcx
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movl	$65536, 16(%rsp)                # imm = 0x10000
	movl	$10000, 4(%rsp)                 # imm = 0x2710
	leaq	88(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	64(%rsp), %rdi
	leaq	48(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	32(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	64(%rsp), %rsi
	movl	72(%rsp), %edx
	movq	48(%rsp), %rcx
	movl	56(%rsp), %r8d
	leaq	128(%rsp), %r9
	movl	$_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii, %edi
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB4_48:
	movq	8(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	movq	8(%rsp), %rdi
	callq	hipEventSynchronize
	movq	24(%rsp), %rsi
	movq	8(%rsp), %rdx
	leaq	128(%rsp), %rdi
	callq	hipEventElapsedTime
	movss	128(%rsp), %xmm1                # xmm1 = mem[0],zero,zero,zero
	xorps	%xmm0, %xmm0
	cvtss2sd	%xmm1, %xmm0
	movss	108(%rsp), %xmm2                # xmm2 = mem[0],zero,zero,zero
	divss	%xmm1, %xmm2
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm2, %xmm1
	movl	$.L.str.8, %edi
	movb	$2, %al
	callq	printf
.LBB4_49:
	movq	96(%rsp), %rdi
	callq	hipFree
	movq	120(%rsp), %rdi
	callq	hipFree
	movq	192(%rsp), %rdi
	callq	hipFree
	movq	200(%rsp), %rdi
	callq	hipFree
	movq	184(%rsp), %rdi
	callq	hipFree
	movq	%rbx, %rdi
	callq	_ZdaPv
	movq	%r14, %rdi
	callq	_ZdaPv
	movq	208(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdaPv
	movq	112(%rsp), %rdi                 # 8-byte Reload
	callq	_ZdaPv
	movq	%r13, %rdi
	callq	_ZdaPv
	movq	24(%rsp), %rdi
	callq	hipEventDestroy
	movq	8(%rsp), %rdi
	callq	hipEventDestroy
	movl	$.Lstr.15, %edi
	callq	puts@PLT
	xorl	%eax, %eax
	addq	$216, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	main, .Lfunc_end4-main
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_e3f2f33f74ed7331(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB5_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_e3f2f33f74ed7331(%rip)
.LBB5_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii, %esi
	movl	$.L__unnamed_2, %edx
	movl	$.L__unnamed_2, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii, %esi
	movl	$.L__unnamed_3, %edx
	movl	$.L__unnamed_3, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii, %esi
	movl	$.L__unnamed_4, %edx
	movl	$.L__unnamed_4, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$0, 8(%rsp)
	movl	$1, (%rsp)
	movl	$kvalues_mxfp4, %esi
	movl	$.L__unnamed_5, %edx
	movl	$.L__unnamed_5, %ecx
	movl	$16, %r9d
	movq	%rbx, %rdi
	xorl	%r8d, %r8d
	callq	__hipRegisterVar
	movl	$__hip_module_dtor, %edi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end5:
	.size	__hip_module_ctor, .Lfunc_end5-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_e3f2f33f74ed7331(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB6_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_e3f2f33f74ed7331(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB6_2:
	retq
.Lfunc_end6:
	.size	__hip_module_dtor, .Lfunc_end6-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	kvalues_mxfp4,@object           # @kvalues_mxfp4
	.local	kvalues_mxfp4
	.comm	kvalues_mxfp4,16,16
	.type	_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i,@object # @_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.section	.rodata,"a",@progbits
	.globl	_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.p2align	3, 0x0
_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i:
	.quad	_Z31__device_stub__test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.size	_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i, 8

	.type	_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii,@object # @_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii
	.globl	_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii
	.p2align	3, 0x0
_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii:
	.quad	_Z30__device_stub__bench_referencePjP15HIP_vector_typeIiLj2EEii
	.size	_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii, 8

	.type	_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii,@object # @_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.globl	_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.p2align	3, 0x0
_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii:
	.quad	_Z26__device_stub__bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.size	_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii, 8

	.type	_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii,@object # @_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii
	.globl	_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii
	.p2align	3, 0x0
_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii:
	.quad	_Z26__device_stub__bench_2permPjP15HIP_vector_typeIiLj2EEii
	.size	_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii, 8

	.type	.L.str.2,@object                # @.str.2
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.2:
	.asciz	"pksub (2 perm + pk_sub)"
	.size	.L.str.2, 24

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"arith_sign (2 perm + pk_sub sign)"
	.size	.L.str.3, 34

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"2perm (minimal perm)"
	.size	.L.str.4, 21

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"\n=== Benchmark (%d x %d iters) ===\n"
	.size	.L.str.5, 36

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"Reference (6 v_perm):     %.2f ms\n"
	.size	.L.str.6, 35

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"pksub (2 perm + pk):      %.2f ms (%.2fx)\n"
	.size	.L.str.7, 43

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"2perm (minimal perm):     %.2f ms (%.2fx)\n"
	.size	.L.str.8, 43

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"[%s] MISMATCH:\n"
	.size	.L.str.10, 16

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"  i=%d (0x%04x): ref=(0x%08x,0x%08x) got=(0x%08x,0x%08x)\n"
	.size	.L.str.11, 58

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"[%s] PASSED\n"
	.size	.L.str.12, 13

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"[%s] FAILED (%d+ errors)\n"
	.size	.L.str.13, 26

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i"
	.size	.L__unnamed_1, 58

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z15bench_referencePjP15HIP_vector_typeIiLj2EEii"
	.size	.L__unnamed_2, 49

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii"
	.size	.L__unnamed_3, 45

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_Z11bench_2permPjP15HIP_vector_typeIiLj2EEii"
	.size	.L__unnamed_4, 45

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"kvalues_mxfp4"
	.size	.L__unnamed_5, 14

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_e3f2f33f74ed7331
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_e3f2f33f74ed7331,@object # @__hip_gpubin_handle_e3f2f33f74ed7331
	.local	__hip_gpubin_handle_e3f2f33f74ed7331
	.comm	__hip_gpubin_handle_e3f2f33f74ed7331,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_e3f2f33f74ed7331,@object # @__hip_cuid_e3f2f33f74ed7331
	.bss
	.globl	__hip_cuid_e3f2f33f74ed7331
__hip_cuid_e3f2f33f74ed7331:
	.byte	0                               # 0x0
	.size	__hip_cuid_e3f2f33f74ed7331, 1

	.type	.Lstr,@object                   # @str
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lstr:
	.asciz	"=== MXFP4 v_pk_sub Approach Test ===\n"
	.size	.Lstr, 38

	.type	.Lstr.14,@object                # @str.14
.Lstr.14:
	.asciz	"=== Correctness ==="
	.size	.Lstr.14, 20

	.type	.Lstr.15,@object                # @str.15
.Lstr.15:
	.asciz	"\n=== Done ==="
	.size	.Lstr.15, 14

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z31__device_stub__test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.addrsig_sym _Z30__device_stub__bench_referencePjP15HIP_vector_typeIiLj2EEii
	.addrsig_sym _Z26__device_stub__bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.addrsig_sym _Z26__device_stub__bench_2permPjP15HIP_vector_typeIiLj2EEii
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym kvalues_mxfp4
	.addrsig_sym _Z16test_correctnessPjP15HIP_vector_typeIiLj2EES2_S2_S2_i
	.addrsig_sym _Z15bench_referencePjP15HIP_vector_typeIiLj2EEii
	.addrsig_sym _Z11bench_pksubPjP15HIP_vector_typeIiLj2EEii
	.addrsig_sym _Z11bench_2permPjP15HIP_vector_typeIiLj2EEii
	.addrsig_sym __hip_fatbin_e3f2f33f74ed7331
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_e3f2f33f74ed7331

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
