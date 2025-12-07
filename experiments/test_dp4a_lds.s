
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx906
	.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf ; -- Begin function _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.globl	_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.p2align	8
	.type	_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf,@function
_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf: ; @_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf
; %bb.0:
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	v_lshlrev_b32_e32 v5, 4, v0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx4 v[1:4], v5, s[2:3]
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[1:4]
.LBB0_2:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, 64, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	s_load_dwordx2 s[2:3], s[4:5], 0x8
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	global_load_ushort v2, v1, s[2:3]
	s_waitcnt vmcnt(0)
	ds_write_b16 v1, v2 offset:1536
.LBB0_4:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v1, 63, v0
	v_cmp_gt_u32_e32 vcc, 32, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_load_dwordx4 s[0:3], s[4:5], 0x10
	s_load_dwordx2 s[6:7], s[4:5], 0x20
	v_lshrrev_b32_e32 v0, 5, v0
	v_and_b32_e32 v10, 30, v0
	v_lshlrev_b32_e32 v11, 6, v10
	s_waitcnt lgkmcnt(0)
	global_load_dwordx4 v[2:5], v11, s[0:1]
	v_mul_u32_u24_e32 v12, 48, v1
	ds_read_b128 v[6:9], v12
	v_lshlrev_b32_e32 v13, 1, v1
	ds_read_u16 v14, v13 offset:1600
	v_or_b32_e32 v0, 1, v0
	v_lshlrev_b32_e32 v15, 6, v0
	v_lshlrev_b32_e32 v1, 3, v1
	s_waitcnt vmcnt(0) lgkmcnt(1)
	v_dot4_i32_i8 v2, v6, v2, 0
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v16, v9, v5, v2
	global_load_dwordx4 v[2:5], v15, s[0:1]
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, 0
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v17, v9, v5, v2
	global_load_dwordx4 v[2:5], v11, s[0:1] offset:32
	ds_read_b128 v[6:9], v12 offset:32
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, 0
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v18, v9, v5, v2
	global_load_dwordx4 v[2:5], v15, s[0:1] offset:32
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, 0
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v19, v9, v5, v2
	global_load_dwordx4 v[2:5], v11, s[0:1] offset:16
	ds_read_b128 v[6:9], v12 offset:16
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, v16
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v16, v9, v5, v2
	global_load_dwordx4 v[2:5], v15, s[0:1] offset:16
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, v17
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v17, v9, v5, v2
	global_load_dwordx4 v[2:5], v11, s[0:1] offset:48
	ds_read_b128 v[6:9], v12 offset:48
	v_lshlrev_b32_e32 v11, 1, v10
	v_add_lshl_u32 v10, v10, v1, 2
	v_lshlrev_b32_e32 v12, 1, v0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, v18
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v4, v9, v5, v2
	global_load_ushort v5, v11, s[2:3] offset:16
	global_load_ushort v18, v11, s[2:3]
	global_load_ushort v20, v12, s[2:3] offset:16
	global_load_ushort v21, v12, s[2:3]
	v_add_lshl_u32 v11, v0, v1, 2
	global_load_dwordx4 v[0:3], v15, s[0:1] offset:48
	ds_read_u16 v12, v13 offset:1536
	s_waitcnt vmcnt(4)
	v_mul_f16_e32 v5, v5, v14
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_mul_f16_e32 v13, v18, v12
	s_waitcnt vmcnt(2)
	v_mul_f16_e32 v14, v20, v14
	s_waitcnt vmcnt(1)
	v_mul_f16_e32 v12, v21, v12
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v0, v6, v0, v19
	v_dot4_i32_i8 v0, v7, v1, v0
	v_dot4_i32_i8 v0, v8, v2, v0
	v_cvt_f32_i32_e32 v1, v16
	v_dot4_i32_i8 v0, v9, v3, v0
	v_cvt_f32_i32_e32 v2, v17
	v_cvt_f32_i32_e32 v3, v4
	v_cvt_f32_i32_e32 v0, v0
	v_fma_mix_f32 v1, v1, v13, 0 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v2, v2, v12, 0 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v1, v3, v5, v1 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v0, v0, v14, v2 op_sel_hi:[0,1,0]
	global_store_dword v10, v1, s[6:7]
	global_store_dword v11, v0, s[6:7]
.LBB0_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf
		.amdhsa_group_segment_fixed_size 1664
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_sgpr 8
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
	.size	_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf, .Lfunc_end0-_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf
                                        ; -- End function
	.set _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.num_vgpr, 22
	.set _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.num_agpr, 0
	.set _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.numbered_sgpr, 8
	.set _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.private_seg_size, 0
	.set _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.uses_vcc, 1
	.set _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.uses_flat_scratch, 0
	.set _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.has_dyn_sized_stack, 0
	.set _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.has_recursion, 0
	.set _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 740
; TotalNumSgprs: 12
; NumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1664 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 12
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
	.protected	_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf ; -- Begin function _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.globl	_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.p2align	8
	.type	_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf,@function
_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf: ; @_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
; %bb.0:
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_2
; %bb.1:
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	v_lshlrev_b32_e32 v5, 4, v0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx4 v[1:4], v5, s[2:3]
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[1:4]
.LBB1_2:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, 64, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_4
; %bb.3:
	s_load_dwordx2 s[2:3], s[4:5], 0x8
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	global_load_ushort v2, v1, s[2:3]
	s_waitcnt vmcnt(0)
	ds_write_b16 v1, v2 offset:1536
.LBB1_4:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v1, 63, v0
	v_cmp_gt_u32_e32 vcc, 32, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_6
; %bb.5:
	s_load_dwordx4 s[0:3], s[4:5], 0x10
	v_lshrrev_b32_e32 v10, 5, v0
	v_and_b32_e32 v0, 30, v10
	v_lshlrev_b32_e32 v11, 6, v0
	v_mul_u32_u24_e32 v12, 48, v1
	s_waitcnt lgkmcnt(0)
	global_load_dwordx4 v[2:5], v11, s[0:1]
	ds_read_b128 v[6:9], v12
	v_or_b32_e32 v10, 1, v10
	v_lshlrev_b32_e32 v14, 6, v10
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, 0
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v13, v9, v5, v2
	global_load_dwordx4 v[2:5], v14, s[0:1]
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, 0
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v15, v9, v5, v2
	global_load_dwordx4 v[2:5], v11, s[0:1] offset:32
	ds_read_b128 v[6:9], v12 offset:32
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, 0
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v16, v9, v5, v2
	global_load_dwordx4 v[2:5], v14, s[0:1] offset:32
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, 0
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v17, v9, v5, v2
	global_load_dwordx4 v[2:5], v11, s[0:1] offset:16
	ds_read_b128 v[6:9], v12 offset:16
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, v13
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v13, v9, v5, v2
	global_load_dwordx4 v[2:5], v14, s[0:1] offset:16
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, v15
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v15, v9, v5, v2
	global_load_dwordx4 v[2:5], v11, s[0:1] offset:48
	ds_read_b128 v[6:9], v12 offset:48
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, v16
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_dot4_i32_i8 v11, v9, v5, v2
	global_load_dwordx4 v[2:5], v14, s[0:1] offset:48
	s_load_dwordx2 s[0:1], s[4:5], 0x20
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v2, v6, v2, v17
	v_dot4_i32_i8 v2, v7, v3, v2
	v_dot4_i32_i8 v2, v8, v4, v2
	v_lshlrev_b32_e32 v3, 1, v0
	v_dot4_i32_i8 v2, v9, v5, v2
	v_lshlrev_b32_e32 v4, 1, v10
	global_load_ushort v5, v3, s[2:3]
	global_load_ushort v6, v3, s[2:3] offset:16
	global_load_ushort v7, v4, s[2:3]
	global_load_ushort v8, v4, s[2:3] offset:16
	v_lshlrev_b32_e32 v3, 1, v1
	ds_read_u16 v4, v3 offset:1536
	ds_read_u16 v3, v3 offset:1600
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v9, v15
	v_lshlrev_b32_e32 v1, 3, v1
	v_add_lshl_u32 v0, v0, v1, 2
	v_add_lshl_u32 v1, v10, v1, 2
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_mul_f16_e32 v5, v5, v4
	s_waitcnt vmcnt(2)
	v_mul_f16_e32 v6, v6, v3
	s_waitcnt vmcnt(1)
	v_mul_f16_e32 v4, v7, v4
	s_waitcnt vmcnt(0)
	v_mul_f16_e32 v3, v8, v3
	v_cvt_f32_f16_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v11
	v_cvt_f32_f16_e32 v3, v3
	v_cvt_f32_i32_e32 v8, v13
	v_mul_f32_e32 v6, v6, v7
	v_mul_f32_e32 v2, v3, v2
	v_fma_mix_f32 v3, v5, v8, v6 op_sel_hi:[1,0,0]
	v_fma_mix_f32 v2, v4, v9, v2 op_sel_hi:[1,0,0]
	global_store_dword v0, v3, s[0:1]
	global_store_dword v1, v2, s[0:1]
.LBB1_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
		.amdhsa_group_segment_fixed_size 1664
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_vgpr 18
		.amdhsa_next_free_sgpr 6
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
	.size	_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf, .Lfunc_end1-_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
                                        ; -- End function
	.set _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.num_vgpr, 18
	.set _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.num_agpr, 0
	.set _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.numbered_sgpr, 6
	.set _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.private_seg_size, 0
	.set _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.uses_vcc, 1
	.set _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.uses_flat_scratch, 0
	.set _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.has_dyn_sized_stack, 0
	.set _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.has_recursion, 0
	.set _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 740
; TotalNumSgprs: 10
; NumVgprs: 18
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1664 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 18
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
	.protected	_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf ; -- Begin function _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.globl	_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.p2align	8
	.type	_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf,@function
_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf: ; @_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
; %bb.0:
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_2
; %bb.1:
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	v_lshlrev_b32_e32 v5, 4, v0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx4 v[1:4], v5, s[2:3]
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[1:4]
.LBB2_2:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, 64, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_4
; %bb.3:
	s_load_dwordx2 s[2:3], s[4:5], 0x8
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	global_load_ushort v2, v1, s[2:3]
	s_waitcnt vmcnt(0)
	ds_write_b16 v1, v2 offset:1536
.LBB2_4:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v1, 63, v0
	v_cmp_gt_u32_e32 vcc, 32, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_6
; %bb.5:
	s_load_dwordx4 s[0:3], s[4:5], 0x10
	v_lshrrev_b32_e32 v0, 5, v0
	v_and_b32_e32 v6, 30, v0
	v_lshlrev_b32_e32 v19, 6, v6
	v_mul_u32_u24_e32 v20, 48, v1
	s_waitcnt lgkmcnt(0)
	global_load_dwordx4 v[7:10], v19, s[0:1]
	ds_read_b128 v[11:14], v20
	v_or_b32_e32 v5, 1, v0
	v_lshlrev_b32_e32 v21, 6, v5
	global_load_dwordx4 v[15:18], v21, s[0:1] offset:48
	s_waitcnt vmcnt(1) lgkmcnt(0)
	v_dot4_i32_i8 v0, v11, v7, 0
	v_dot4_i32_i8 v0, v12, v8, v0
	v_dot4_i32_i8 v0, v13, v9, v0
	v_dot4_i32_i8 v0, v14, v10, v0
	global_load_dwordx4 v[7:10], v21, s[0:1]
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v2, v11, v7, 0
	v_dot4_i32_i8 v2, v12, v8, v2
	v_dot4_i32_i8 v2, v13, v9, v2
	v_dot4_i32_i8 v2, v14, v10, v2
	global_load_dwordx4 v[7:10], v19, s[0:1] offset:32
	ds_read_b128 v[11:14], v20 offset:32
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v3, v11, v7, 0
	v_dot4_i32_i8 v3, v12, v8, v3
	v_dot4_i32_i8 v3, v13, v9, v3
	v_dot4_i32_i8 v3, v14, v10, v3
	global_load_dwordx4 v[7:10], v21, s[0:1] offset:32
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v4, v11, v7, 0
	v_dot4_i32_i8 v4, v12, v8, v4
	v_dot4_i32_i8 v4, v13, v9, v4
	v_dot4_i32_i8 v4, v14, v10, v4
	global_load_dwordx4 v[7:10], v19, s[0:1] offset:48
	ds_read_b128 v[11:14], v20 offset:48
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v7, v11, v7, 0
	v_dot4_i32_i8 v7, v12, v8, v7
	v_dot4_i32_i8 v8, v11, v15, 0
	v_dot4_i32_i8 v8, v12, v16, v8
	v_dot4_i32_i8 v7, v13, v9, v7
	v_dot4_i32_i8 v8, v13, v17, v8
	v_dot4_i32_i8 v7, v14, v10, v7
	v_dot4_i32_i8 v8, v14, v18, v8
	global_load_dwordx3 v[14:16], v19, s[0:1] offset:16
	ds_read_b128 v[10:13], v20 offset:16
	v_add_u32_e32 v3, v7, v3
	v_add_u32_e32 v4, v8, v4
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_dot4_i32_i8 v9, v10, v14, 0
	v_dot4_i32_i8 v9, v11, v15, v9
	v_dot4_i32_i8 v9, v12, v16, v9
	v_dot4_i32_i8 v9, v13, v15, v9
	global_load_dwordx3 v[14:16], v21, s[0:1] offset:16
	v_add_u32_e32 v0, v9, v0
	s_load_dwordx2 s[0:1], s[4:5], 0x20
	v_cvt_f32_i32_e32 v0, v0
	s_waitcnt vmcnt(0)
	v_dot4_i32_i8 v10, v10, v14, 0
	v_dot4_i32_i8 v10, v11, v15, v10
	v_dot4_i32_i8 v10, v12, v16, v10
	v_dot4_i32_i8 v10, v13, v15, v10
	v_lshlrev_b32_e32 v15, 1, v6
	v_lshlrev_b32_e32 v16, 1, v5
	global_load_ushort v14, v15, s[2:3]
	global_load_ushort v13, v15, s[2:3] offset:16
	global_load_ushort v12, v16, s[2:3]
	global_load_ushort v11, v16, s[2:3] offset:16
	v_lshlrev_b32_e32 v15, 1, v1
	ds_read_u16 v16, v15 offset:1536
	ds_read_u16 v15, v15 offset:1600
	v_add_u32_e32 v2, v10, v2
	v_cvt_f32_i32_e32 v2, v2
	v_lshlrev_b32_e32 v1, 3, v1
	v_add_lshl_u32 v6, v6, v1, 2
	v_add_lshl_u32 v1, v5, v1, 2
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_mul_f16_e32 v5, v14, v16
	s_waitcnt vmcnt(2)
	v_mul_f16_e32 v13, v13, v15
	v_cvt_f32_f16_e32 v13, v13
	s_waitcnt vmcnt(0)
	v_mul_f16_e32 v11, v11, v15
	v_cvt_f32_f16_e32 v11, v11
	v_mul_f16_e32 v12, v12, v16
	v_mul_f32_e32 v3, v13, v3
	v_fma_mix_f32 v0, v5, v0, v3 op_sel_hi:[1,0,0]
	v_mul_f32_e32 v4, v11, v4
	v_fma_mix_f32 v2, v12, v2, v4 op_sel_hi:[1,0,0]
	global_store_dword v6, v0, s[0:1]
	global_store_dword v1, v2, s[0:1]
.LBB2_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
		.amdhsa_group_segment_fixed_size 1664
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_sgpr 6
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
	.size	_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf, .Lfunc_end2-_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
                                        ; -- End function
	.set _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.num_vgpr, 22
	.set _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.num_agpr, 0
	.set _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.numbered_sgpr, 6
	.set _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.private_seg_size, 0
	.set _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.uses_vcc, 1
	.set _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.uses_flat_scratch, 0
	.set _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.has_dyn_sized_stack, 0
	.set _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.has_recursion, 0
	.set _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 748
; TotalNumSgprs: 10
; NumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1664 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 10
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
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_e5024e03cb76923c,@object ; @__hip_cuid_e5024e03cb76923c
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e5024e03cb76923c
__hip_cuid_e5024e03cb76923c:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e5024e03cb76923c, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e5024e03cb76923c
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1664
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     22
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1664
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     18
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1664
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     22
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
	.file	"test_dp4a_lds.hip"
	.text
	.globl	_Z32__device_stub__fattn_kq_baselinePKaPK6__halfS0_S3_Pf # -- Begin function _Z32__device_stub__fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.p2align	4
	.type	_Z32__device_stub__fattn_kq_baselinePKaPK6__halfS0_S3_Pf,@function
_Z32__device_stub__fattn_kq_baselinePKaPK6__halfS0_S3_Pf: # @_Z32__device_stub__fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.cfi_startproc
# %bb.0:
	subq	$136, %rsp
	.cfi_def_cfa_offset 144
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
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
	movl	$_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$152, %rsp
	.cfi_adjust_cfa_offset -152
	retq
.Lfunc_end0:
	.size	_Z32__device_stub__fattn_kq_baselinePKaPK6__halfS0_S3_Pf, .Lfunc_end0-_Z32__device_stub__fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.cfi_endproc
                                        # -- End function
	.globl	_Z33__device_stub__fattn_kq_optimizedPKaPK6__halfS0_S3_Pf # -- Begin function _Z33__device_stub__fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.p2align	4
	.type	_Z33__device_stub__fattn_kq_optimizedPKaPK6__halfS0_S3_Pf,@function
_Z33__device_stub__fattn_kq_optimizedPKaPK6__halfS0_S3_Pf: # @_Z33__device_stub__fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.cfi_startproc
# %bb.0:
	subq	$136, %rsp
	.cfi_def_cfa_offset 144
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
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
	movl	$_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$152, %rsp
	.cfi_adjust_cfa_offset -152
	retq
.Lfunc_end1:
	.size	_Z33__device_stub__fattn_kq_optimizedPKaPK6__halfS0_S3_Pf, .Lfunc_end1-_Z33__device_stub__fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.cfi_endproc
                                        # -- End function
	.globl	_Z36__device_stub__fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf # -- Begin function _Z36__device_stub__fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.p2align	4
	.type	_Z36__device_stub__fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf,@function
_Z36__device_stub__fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf: # @_Z36__device_stub__fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.cfi_startproc
# %bb.0:
	subq	$136, %rsp
	.cfi_def_cfa_offset 144
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
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
	movl	$_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$152, %rsp
	.cfi_adjust_cfa_offset -152
	retq
.Lfunc_end2:
	.size	_Z36__device_stub__fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf, .Lfunc_end2-_Z36__device_stub__fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function main
.LCPI3_0:
	.long	0x3c23d70a                      # float 0.00999999977
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI3_1:
	.quad	0x408f400000000000              # double 1000
.LCPI3_2:
	.quad	0x407f400000000000              # double 500
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
	subq	$184, %rsp
	.cfi_def_cfa_offset 240
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$.Lstr, %edi
	callq	puts@PLT
	movl	$.L.str.1, %edi
	movl	$32, %esi
	movl	$8, %edx
	movl	$64, %ecx
	movl	$48, %r8d
	xorl	%eax, %eax
	callq	printf
	leaq	112(%rsp), %rdi
	movl	$2048, %esi                     # imm = 0x800
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB3_1
# %bb.3:
	leaq	104(%rsp), %rdi
	movl	$512, %esi                      # imm = 0x200
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB3_4
# %bb.5:
	leaq	96(%rsp), %rdi
	movl	$128, %esi
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB3_6
# %bb.7:
	leaq	88(%rsp), %rdi
	movl	$32, %esi
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB3_8
# %bb.9:
	leaq	120(%rsp), %rdi
	movl	$1024, %esi                     # imm = 0x400
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB3_10
# %bb.11:
	movl	$2048, %edi                     # imm = 0x800
	callq	_Znam
	movq	%rax, %r14
	movl	$512, %edi                      # imm = 0x200
	callq	_Znam
	movq	%rax, %r15
	movl	$128, %edi
	callq	_Znam
	movq	%rax, %r12
	movl	$32, %edi
	callq	_Znam
	movq	%rax, %r13
	xorl	%ebx, %ebx
	.p2align	4
.LBB3_12:                               # =>This Inner Loop Header: Depth=1
	callq	rand
	addb	$-128, %al
	movb	%al, (%r14,%rbx)
	incq	%rbx
	cmpq	$2048, %rbx                     # imm = 0x800
	jne	.LBB3_12
# %bb.13:
	xorl	%ebx, %ebx
	.p2align	4
.LBB3_14:                               # =>This Inner Loop Header: Depth=1
	callq	rand
	addb	$-128, %al
	movb	%al, (%r15,%rbx)
	incq	%rbx
	cmpq	$512, %rbx                      # imm = 0x200
	jne	.LBB3_14
# %bb.15:
	xorl	%ebx, %ebx
	.p2align	4
.LBB3_16:                               # =>This Inner Loop Header: Depth=1
	callq	rand
	cltq
	imulq	$1374389535, %rax, %rcx         # imm = 0x51EB851F
	movq	%rcx, %rdx
	shrq	$63, %rdx
	sarq	$37, %rcx
	addl	%edx, %ecx
	imull	$100, %ecx, %ecx
	subl	%ecx, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI3_0(%rip), %xmm0
	callq	__truncsfhf2@PLT
	pextrw	$0, %xmm0, %eax
	movw	%ax, (%r12,%rbx,2)
	incq	%rbx
	cmpq	$64, %rbx
	jne	.LBB3_16
# %bb.17:
	xorl	%ebx, %ebx
	.p2align	4
.LBB3_18:                               # =>This Inner Loop Header: Depth=1
	callq	rand
	cltq
	imulq	$1374389535, %rax, %rcx         # imm = 0x51EB851F
	movq	%rcx, %rdx
	shrq	$63, %rdx
	sarq	$37, %rcx
	addl	%edx, %ecx
	imull	$100, %ecx, %ecx
	subl	%ecx, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	.LCPI3_0(%rip), %xmm0
	callq	__truncsfhf2@PLT
	pextrw	$0, %xmm0, %eax
	movw	%ax, (%r13,%rbx,2)
	incq	%rbx
	cmpq	$16, %rbx
	jne	.LBB3_18
# %bb.19:
	movq	112(%rsp), %rdi
	movl	$2048, %edx                     # imm = 0x800
	movq	%r14, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	testl	%eax, %eax
	jne	.LBB3_20
# %bb.21:
	movq	104(%rsp), %rdi
	movl	$512, %edx                      # imm = 0x200
	movq	%r15, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	testl	%eax, %eax
	jne	.LBB3_22
# %bb.23:
	movq	96(%rsp), %rdi
	movl	$128, %edx
	movq	%r12, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	testl	%eax, %eax
	jne	.LBB3_24
# %bb.25:
	movq	88(%rsp), %rdi
	movl	$32, %edx
	movq	%r13, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
	testl	%eax, %eax
	jne	.LBB3_26
# %bb.27:
	movabsq	$4294967297, %rbx               # imm = 0x100000001
	movq	%r14, %rdi
	callq	_ZdaPv
	movq	%r15, %rdi
	callq	_ZdaPv
	movq	%r12, %rdi
	callq	_ZdaPv
	movq	%r13, %rdi
	callq	_ZdaPv
	xorl	%r12d, %r12d
	leaq	255(%rbx), %r14
	movq	%rsp, %rbp
	leaq	128(%rsp), %r15
	jmp	.LBB3_28
	.p2align	4
.LBB3_32:                               #   in Loop: Header=BB3_28 Depth=1
	incl	%r12d
	cmpl	$50, %r12d
	je	.LBB3_33
.LBB3_28:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_29 Depth 2
	movl	$100, %r13d
	jmp	.LBB3_29
	.p2align	4
.LBB3_31:                               #   in Loop: Header=BB3_29 Depth=2
	decl	%r13d
	je	.LBB3_32
.LBB3_29:                               #   Parent Loop BB3_28 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB3_31
# %bb.30:                               #   in Loop: Header=BB3_29 Depth=2
	movq	112(%rsp), %rax
	movq	96(%rsp), %rcx
	movq	104(%rsp), %rdx
	movq	88(%rsp), %rsi
	movq	120(%rsp), %rdi
	movq	%rax, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rsi, 56(%rsp)
	movq	%rdi, 48(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rbp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	movl	$_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf, %edi
	movq	%r15, %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	jmp	.LBB3_31
.LBB3_33:
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB3_34
# %bb.35:
	xorl	%r13d, %r13d
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, 168(%rsp)                 # 8-byte Spill
	movq	%rsp, %r15
	leaq	128(%rsp), %r12
	jmp	.LBB3_36
	.p2align	4
.LBB3_40:                               #   in Loop: Header=BB3_36 Depth=1
	incl	%r13d
	cmpl	$500, %r13d                     # imm = 0x1F4
	je	.LBB3_41
.LBB3_36:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_37 Depth 2
	movl	$100, %ebp
	jmp	.LBB3_37
	.p2align	4
.LBB3_39:                               #   in Loop: Header=BB3_37 Depth=2
	decl	%ebp
	je	.LBB3_40
.LBB3_37:                               #   Parent Loop BB3_36 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB3_39
# %bb.38:                               #   in Loop: Header=BB3_37 Depth=2
	movq	112(%rsp), %rax
	movq	96(%rsp), %rcx
	movq	104(%rsp), %rdx
	movq	88(%rsp), %rsi
	movq	120(%rsp), %rdi
	movq	%rax, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rsi, 56(%rsp)
	movq	%rdi, 48(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%r15, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	movl	$_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf, %edi
	movq	%r12, %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	jmp	.LBB3_39
.LBB3_41:
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB3_42
# %bb.43:
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	subq	168(%rsp), %rax                 # 8-byte Folded Reload
	cvtsi2sd	%rax, %xmm0
	divsd	.LCPI3_1(%rip), %xmm0
	divsd	.LCPI3_2(%rip), %xmm0
	movapd	%xmm0, %xmm1
	divsd	%xmm0, %xmm1
	movl	$.L.str.7, %edi
	movl	$.L.str.4, %esi
	movsd	%xmm0, 168(%rsp)                # 8-byte Spill
	movb	$2, %al
	callq	printf
	xorl	%r12d, %r12d
	movq	%rsp, %rbp
	leaq	128(%rsp), %r15
	jmp	.LBB3_44
	.p2align	4
.LBB3_48:                               #   in Loop: Header=BB3_44 Depth=1
	incl	%r12d
	cmpl	$50, %r12d
	je	.LBB3_49
.LBB3_44:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_45 Depth 2
	movl	$100, %r13d
	jmp	.LBB3_45
	.p2align	4
.LBB3_47:                               #   in Loop: Header=BB3_45 Depth=2
	decl	%r13d
	je	.LBB3_48
.LBB3_45:                               #   Parent Loop BB3_44 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB3_47
# %bb.46:                               #   in Loop: Header=BB3_45 Depth=2
	movq	112(%rsp), %rax
	movq	96(%rsp), %rcx
	movq	104(%rsp), %rdx
	movq	88(%rsp), %rsi
	movq	120(%rsp), %rdi
	movq	%rax, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rsi, 56(%rsp)
	movq	%rdi, 48(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rbp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	movl	$_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf, %edi
	movq	%r15, %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	jmp	.LBB3_47
.LBB3_49:
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB3_34
# %bb.50:
	xorl	%r13d, %r13d
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, 176(%rsp)                 # 8-byte Spill
	movq	%rsp, %r15
	leaq	128(%rsp), %r12
	jmp	.LBB3_51
	.p2align	4
.LBB3_55:                               #   in Loop: Header=BB3_51 Depth=1
	incl	%r13d
	cmpl	$500, %r13d                     # imm = 0x1F4
	je	.LBB3_56
.LBB3_51:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_52 Depth 2
	movl	$100, %ebp
	jmp	.LBB3_52
	.p2align	4
.LBB3_54:                               #   in Loop: Header=BB3_52 Depth=2
	decl	%ebp
	je	.LBB3_55
.LBB3_52:                               #   Parent Loop BB3_51 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB3_54
# %bb.53:                               #   in Loop: Header=BB3_52 Depth=2
	movq	112(%rsp), %rax
	movq	96(%rsp), %rcx
	movq	104(%rsp), %rdx
	movq	88(%rsp), %rsi
	movq	120(%rsp), %rdi
	movq	%rax, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rsi, 56(%rsp)
	movq	%rdi, 48(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%r15, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	movl	$_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf, %edi
	movq	%r12, %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	jmp	.LBB3_54
.LBB3_56:
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB3_42
# %bb.57:
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	subq	176(%rsp), %rax                 # 8-byte Folded Reload
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rax, %xmm0
	divsd	.LCPI3_1(%rip), %xmm0
	divsd	.LCPI3_2(%rip), %xmm0
	movsd	168(%rsp), %xmm1                # 8-byte Reload
                                        # xmm1 = mem[0],zero
	divsd	%xmm0, %xmm1
	movl	$.L.str.7, %edi
	movl	$.L.str.5, %esi
	movb	$2, %al
	callq	printf
	xorl	%r12d, %r12d
	movq	%rsp, %rbp
	leaq	128(%rsp), %r15
	jmp	.LBB3_58
	.p2align	4
.LBB3_62:                               #   in Loop: Header=BB3_58 Depth=1
	incl	%r12d
	cmpl	$50, %r12d
	je	.LBB3_63
.LBB3_58:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_59 Depth 2
	movl	$100, %r13d
	jmp	.LBB3_59
	.p2align	4
.LBB3_61:                               #   in Loop: Header=BB3_59 Depth=2
	decl	%r13d
	je	.LBB3_62
.LBB3_59:                               #   Parent Loop BB3_58 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB3_61
# %bb.60:                               #   in Loop: Header=BB3_59 Depth=2
	movq	112(%rsp), %rax
	movq	96(%rsp), %rcx
	movq	104(%rsp), %rdx
	movq	88(%rsp), %rsi
	movq	120(%rsp), %rdi
	movq	%rax, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rsi, 56(%rsp)
	movq	%rdi, 48(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rbp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	movl	$_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf, %edi
	movq	%r15, %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	jmp	.LBB3_61
.LBB3_63:
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB3_34
# %bb.64:
	xorl	%r13d, %r13d
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, 176(%rsp)                 # 8-byte Spill
	movq	%rsp, %r15
	leaq	128(%rsp), %r12
	jmp	.LBB3_65
	.p2align	4
.LBB3_69:                               #   in Loop: Header=BB3_65 Depth=1
	incl	%r13d
	cmpl	$500, %r13d                     # imm = 0x1F4
	je	.LBB3_70
.LBB3_65:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_66 Depth 2
	movl	$100, %ebp
	jmp	.LBB3_66
	.p2align	4
.LBB3_68:                               #   in Loop: Header=BB3_66 Depth=2
	decl	%ebp
	je	.LBB3_69
.LBB3_66:                               #   Parent Loop BB3_65 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB3_68
# %bb.67:                               #   in Loop: Header=BB3_66 Depth=2
	movq	112(%rsp), %rax
	movq	96(%rsp), %rcx
	movq	104(%rsp), %rdx
	movq	88(%rsp), %rsi
	movq	120(%rsp), %rdi
	movq	%rax, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rsi, 56(%rsp)
	movq	%rdi, 48(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%r15, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	movl	$_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf, %edi
	movq	%r12, %r9
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	jmp	.LBB3_68
.LBB3_70:
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB3_42
# %bb.71:
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	subq	176(%rsp), %rax                 # 8-byte Folded Reload
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rax, %xmm0
	divsd	.LCPI3_1(%rip), %xmm0
	divsd	.LCPI3_2(%rip), %xmm0
	movsd	168(%rsp), %xmm1                # 8-byte Reload
                                        # xmm1 = mem[0],zero
	divsd	%xmm0, %xmm1
	movl	$.L.str.7, %edi
	movl	$.L.str.6, %esi
	movb	$2, %al
	callq	printf
	movq	112(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB3_72
# %bb.73:
	movq	104(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB3_74
# %bb.75:
	movq	96(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB3_76
# %bb.77:
	movq	88(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB3_78
# %bb.79:
	movq	120(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB3_80
# %bb.81:
	xorl	%eax, %eax
	addq	$184, %rsp
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
.LBB3_34:
	.cfi_def_cfa_offset 240
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$355, %ecx                      # imm = 0x163
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_42:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$364, %ecx                      # imm = 0x16C
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_1:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$303, %ecx                      # imm = 0x12F
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_4:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$304, %ecx                      # imm = 0x130
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_6:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$305, %ecx                      # imm = 0x131
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_8:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$306, %ecx                      # imm = 0x132
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_10:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$307, %ecx                      # imm = 0x133
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_20:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$321, %ecx                      # imm = 0x141
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_22:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$322, %ecx                      # imm = 0x142
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_24:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$323, %ecx                      # imm = 0x143
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_26:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$324, %ecx                      # imm = 0x144
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_72:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$375, %ecx                      # imm = 0x177
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_74:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$376, %ecx                      # imm = 0x178
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_76:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$377, %ecx                      # imm = 0x179
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_78:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$378, %ecx                      # imm = 0x17A
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.LBB3_80:
	movl	%eax, %edi
	callq	hipGetErrorString
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %edx
	movq	%rax, %rsi
	movl	$379, %ecx                      # imm = 0x17B
	xorl	%eax, %eax
	callq	printf
	movl	$1, %edi
	callq	exit
.Lfunc_end3:
	.size	main, .Lfunc_end3-main
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
	movq	__hip_gpubin_handle_e5024e03cb76923c(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB4_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_e5024e03cb76923c(%rip)
.LBB4_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf, %esi
	movl	$.L__unnamed_2, %edx
	movl	$.L__unnamed_2, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf, %esi
	movl	$.L__unnamed_3, %edx
	movl	$.L__unnamed_3, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end4:
	.size	__hip_module_ctor, .Lfunc_end4-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_e5024e03cb76923c(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB5_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_e5024e03cb76923c(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB5_2:
	retq
.Lfunc_end5:
	.size	__hip_module_dtor, .Lfunc_end5-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf,@object # @_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.section	.rodata,"a",@progbits
	.globl	_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.p2align	3, 0x0
_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf:
	.quad	_Z32__device_stub__fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.size	_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf, 8

	.type	_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf,@object # @_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.globl	_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.p2align	3, 0x0
_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf:
	.quad	_Z33__device_stub__fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.size	_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf, 8

	.type	_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf,@object # @_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.globl	_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.p2align	3, 0x0
_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf:
	.quad	_Z36__device_stub__fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.size	_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf, 8

	.type	.L.str.1,@object                # @.str.1
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.1:
	.asciz	"nbatch_fa=%d, ncols=%d, DKQ=%d, K_row_stride=%d\n\n"
	.size	.L.str.1, 50

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"HIP error %s at %s:%d\n"
	.size	.L.str.2, 23

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"test_dp4a_lds.hip"
	.size	.L.str.3, 18

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"baseline"
	.size	.L.str.4, 9

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"optimized (2-interleave)"
	.size	.L.str.5, 25

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"optimized_v2 (4-interleave)"
	.size	.L.str.6, 28

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"%-30s: %8.2f us  (%.2fx baseline)\n"
	.size	.L.str.7, 35

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf"
	.size	.L__unnamed_1, 42

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf"
	.size	.L__unnamed_2, 43

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf"
	.size	.L__unnamed_3, 46

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_e5024e03cb76923c
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_e5024e03cb76923c,@object # @__hip_gpubin_handle_e5024e03cb76923c
	.local	__hip_gpubin_handle_e5024e03cb76923c
	.comm	__hip_gpubin_handle_e5024e03cb76923c,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_e5024e03cb76923c,@object # @__hip_cuid_e5024e03cb76923c
	.bss
	.globl	__hip_cuid_e5024e03cb76923c
__hip_cuid_e5024e03cb76923c:
	.byte	0                               # 0x0
	.size	__hip_cuid_e5024e03cb76923c, 1

	.type	.Lstr,@object                   # @str
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lstr:
	.asciz	"Testing K\302\267Q dp4a with LDS (flash attention pattern)"
	.size	.Lstr, 53

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z32__device_stub__fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.addrsig_sym _Z33__device_stub__fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.addrsig_sym _Z36__device_stub__fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Z17fattn_kq_baselinePKaPK6__halfS0_S3_Pf
	.addrsig_sym _Z18fattn_kq_optimizedPKaPK6__halfS0_S3_Pf
	.addrsig_sym _Z21fattn_kq_optimized_v2PKaPK6__halfS0_S3_Pf
	.addrsig_sym __hip_fatbin_e5024e03cb76923c
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_e5024e03cb76923c

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
