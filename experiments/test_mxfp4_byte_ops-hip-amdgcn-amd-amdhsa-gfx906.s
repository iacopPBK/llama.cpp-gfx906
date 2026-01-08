	.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z18benchmark_baselinePjPKji ; -- Begin function _Z18benchmark_baselinePjPKji
	.globl	_Z18benchmark_baselinePjPKji
	.p2align	8
	.type	_Z18benchmark_baselinePjPKji,@function
_Z18benchmark_baselinePjPKji:           ; @_Z18benchmark_baselinePjPKji
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
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	v_mov_b32_e32 v4, 0xfdfeff00
	v_lshlrev_b32_e32 v0, 1, v0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s3
	v_add_co_u32_e32 v1, vcc, s2, v1
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc
	global_load_dword v5, v[1:2], off
	v_mov_b32_e32 v1, 0xc080604
	v_mov_b32_e32 v2, 0x3020100
	v_mov_b32_e32 v3, 0xf4f8fafc
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 1, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v9, 0x4040404, v9
	v_lshrrev_b32_e32 v6, 4, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v9, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v9, 0x4040404, v9
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v6, v6, v8, v9
	;;#ASMEND
	v_xor_b32_e32 v6, v7, v5
	v_lshrrev_b32_e32 v5, 4, v6
	v_lshrrev_b32_e32 v9, 1, v6
	v_and_b32_e32 v7, 0x7070707, v6
	v_and_b32_e32 v8, 0x7070707, v5
	;;#ASMSTART
	v_perm_b32 v5, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v5, v7, v5, v9
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v1, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v3, v4, v8
	;;#ASMEND
	v_lshrrev_b32_e32 v3, 5, v6
	v_and_b32_e32 v3, 0x4040404, v3
	v_or_b32_e32 v3, 0x3020100, v3
	;;#ASMSTART
	v_perm_b32 v6, v2, v1, v3
	;;#ASMEND
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_mov_b32_e32 v2, s1
	v_add_co_u32_e32 v0, vcc, s0, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	global_store_dwordx2 v[0:1], v[5:6], off
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z18benchmark_baselinePjPKji
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
		.amdhsa_next_free_vgpr 11
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
	.size	_Z18benchmark_baselinePjPKji, .Lfunc_end0-_Z18benchmark_baselinePjPKji
                                        ; -- End function
	.set _Z18benchmark_baselinePjPKji.num_vgpr, 11
	.set _Z18benchmark_baselinePjPKji.num_agpr, 0
	.set _Z18benchmark_baselinePjPKji.numbered_sgpr, 7
	.set _Z18benchmark_baselinePjPKji.private_seg_size, 0
	.set _Z18benchmark_baselinePjPKji.uses_vcc, 1
	.set _Z18benchmark_baselinePjPKji.uses_flat_scratch, 0
	.set _Z18benchmark_baselinePjPKji.has_dyn_sized_stack, 0
	.set _Z18benchmark_baselinePjPKji.has_recursion, 0
	.set _Z18benchmark_baselinePjPKji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11364
; TotalNumSgprs: 11
; NumVgprs: 11
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 11
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
	.protected	_Z15benchmark_opt1aPjPKji ; -- Begin function _Z15benchmark_opt1aPjPKji
	.globl	_Z15benchmark_opt1aPjPKji
	.p2align	8
	.type	_Z15benchmark_opt1aPjPKji,@function
_Z15benchmark_opt1aPjPKji:              ; @_Z15benchmark_opt1aPjPKji
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
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	v_mov_b32_e32 v4, 0xfdfeff00
	v_mov_b32_e32 v5, 0xff00
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s3
	v_add_co_u32_e32 v1, vcc, s2, v1
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc
	global_load_dword v6, v[1:2], off
	v_mov_b32_e32 v1, 0xc080604
	v_mov_b32_e32 v2, 0x3020100
	v_mov_b32_e32 v3, 0xf4f8fafc
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_lshrrev_b32_e32 v9, 3, v6
	v_lshrrev_b32_e32 v10, 7, v6
	v_and_b32_e32 v7, 0x7070707, v7
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v8
	;;#ASMEND
	v_and_b32_e32 v9, 0x1010101, v9
	v_and_b32_e32 v10, 0x1010101, v10
	;;#ASMSTART
	v_perm_b32 v12, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v5, v5, v9
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v5, v5, v10
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v9, v8, v11
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v10, v7, v12
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_lshrrev_b32_e32 v9, 3, v6
	v_lshrrev_b32_e32 v10, 7, v6
	v_and_b32_e32 v7, 0x7070707, v7
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v8
	;;#ASMEND
	v_and_b32_e32 v9, 0x1010101, v9
	v_and_b32_e32 v10, 0x1010101, v10
	;;#ASMSTART
	v_perm_b32 v12, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v5, v5, v9
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v5, v5, v10
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v9, v8, v11
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v10, v7, v12
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_lshrrev_b32_e32 v9, 3, v6
	v_lshrrev_b32_e32 v10, 7, v6
	v_and_b32_e32 v7, 0x7070707, v7
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v8
	;;#ASMEND
	v_and_b32_e32 v9, 0x1010101, v9
	v_and_b32_e32 v10, 0x1010101, v10
	;;#ASMSTART
	v_perm_b32 v12, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v5, v5, v9
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v5, v5, v10
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v9, v8, v11
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v10, v7, v12
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_lshrrev_b32_e32 v9, 3, v6
	v_lshrrev_b32_e32 v10, 7, v6
	v_and_b32_e32 v7, 0x7070707, v7
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v8
	;;#ASMEND
	v_and_b32_e32 v9, 0x1010101, v9
	v_and_b32_e32 v10, 0x1010101, v10
	;;#ASMSTART
	v_perm_b32 v12, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v5, v5, v9
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v5, v5, v10
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v9, v8, v11
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v10, v7, v12
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_lshrrev_b32_e32 v9, 3, v6
	v_lshrrev_b32_e32 v10, 7, v6
	v_and_b32_e32 v7, 0x7070707, v7
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v8
	;;#ASMEND
	v_and_b32_e32 v9, 0x1010101, v9
	v_and_b32_e32 v10, 0x1010101, v10
	;;#ASMSTART
	v_perm_b32 v12, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v5, v5, v9
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v5, v5, v10
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v9, v8, v11
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v10, v7, v12
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_lshrrev_b32_e32 v9, 3, v6
	v_lshrrev_b32_e32 v10, 7, v6
	v_and_b32_e32 v7, 0x7070707, v7
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v8
	;;#ASMEND
	v_and_b32_e32 v9, 0x1010101, v9
	v_and_b32_e32 v10, 0x1010101, v10
	;;#ASMSTART
	v_perm_b32 v12, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v5, v5, v9
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v5, v5, v10
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v9, v8, v11
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v10, v7, v12
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_lshrrev_b32_e32 v9, 3, v6
	v_lshrrev_b32_e32 v10, 7, v6
	v_and_b32_e32 v7, 0x7070707, v7
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v8
	;;#ASMEND
	v_and_b32_e32 v9, 0x1010101, v9
	v_and_b32_e32 v10, 0x1010101, v10
	;;#ASMSTART
	v_perm_b32 v12, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v5, v5, v9
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v5, v5, v10
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v9, v8, v11
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v10, v7, v12
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	v_lshrrev_b32_e32 v11, 3, v6
	v_lshrrev_b32_e32 v12, 7, v6
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v11, 0x1010101, v11
	v_and_b32_e32 v12, 0x1010101, v12
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v5, v5, v11
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v12, v5, v5, v12
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v8, v11, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v7, v12, v7, v10
	;;#ASMEND
	v_xor_b32_e32 v6, v8, v6
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v7, 0x7070707, v7
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v1, v2, v7
	;;#ASMEND
	v_lshrrev_b32_e32 v1, 3, v6
	v_and_b32_e32 v1, 0x1010101, v1
	v_lshrrev_b32_e32 v6, 7, v6
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v1, v5, v5, v1
	;;#ASMEND
	v_and_b32_e32 v6, 0x1010101, v6
	;;#ASMSTART
	v_perm_b32 v4, v5, v5, v6
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v1, v1, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_bfi_b32 v2, v4, v3, v2
	;;#ASMEND
	v_lshlrev_b32_e32 v3, 1, v0
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_mov_b32_e32 v0, s1
	v_add_co_u32_e32 v3, vcc, s0, v3
	v_addc_co_u32_e32 v4, vcc, v0, v4, vcc
	global_store_dwordx2 v[3:4], v[1:2], off
.LBB1_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z15benchmark_opt1aPjPKji
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
	.size	_Z15benchmark_opt1aPjPKji, .Lfunc_end1-_Z15benchmark_opt1aPjPKji
                                        ; -- End function
	.set _Z15benchmark_opt1aPjPKji.num_vgpr, 13
	.set _Z15benchmark_opt1aPjPKji.num_agpr, 0
	.set _Z15benchmark_opt1aPjPKji.numbered_sgpr, 7
	.set _Z15benchmark_opt1aPjPKji.private_seg_size, 0
	.set _Z15benchmark_opt1aPjPKji.uses_vcc, 1
	.set _Z15benchmark_opt1aPjPKji.uses_flat_scratch, 0
	.set _Z15benchmark_opt1aPjPKji.has_dyn_sized_stack, 0
	.set _Z15benchmark_opt1aPjPKji.has_recursion, 0
	.set _Z15benchmark_opt1aPjPKji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11372
; TotalNumSgprs: 11
; NumVgprs: 13
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 11
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
	.protected	_Z15benchmark_opt1bPjPKji ; -- Begin function _Z15benchmark_opt1bPjPKji
	.globl	_Z15benchmark_opt1bPjPKji
	.p2align	8
	.type	_Z15benchmark_opt1bPjPKji,@function
_Z15benchmark_opt1bPjPKji:              ; @_Z15benchmark_opt1bPjPKji
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dword s1, s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s1, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	v_mov_b32_e32 v4, 0xfdfeff00
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s3
	v_add_co_u32_e32 v1, vcc, s2, v1
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc
	global_load_dword v5, v[1:2], off
	v_mov_b32_e32 v1, 0xc080604
	v_mov_b32_e32 v2, 0x3020100
	v_mov_b32_e32 v3, 0xf4f8fafc
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v9, 0x4040404, v9
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v10, 0x4040404, v10
	v_and_b32_e32 v11, 0x4040404, v11
	v_or_b32_e32 v10, 0x3020100, v10
	v_or_b32_e32 v11, 0x3020100, v11
	;;#ASMSTART
	v_perm_b32 v7, v7, v8, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v9, v11
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v1, v2, v6
	;;#ASMEND
	v_lshrrev_b32_e32 v1, 1, v5
	;;#ASMSTART
	v_perm_b32 v3, v3, v4, v6
	;;#ASMEND
	v_and_b32_e32 v1, 0x4040404, v1
	v_lshrrev_b32_e32 v4, 5, v5
	v_or_b32_e32 v1, 0x3020100, v1
	v_and_b32_e32 v4, 0x4040404, v4
	v_or_b32_e32 v4, 0x3020100, v4
	;;#ASMSTART
	v_perm_b32 v1, v7, v8, v1
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v3, v2, v4
	;;#ASMEND
	v_lshlrev_b32_e32 v3, 1, v0
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_mov_b32_e32 v0, s1
	v_add_co_u32_e32 v3, vcc, s0, v3
	v_addc_co_u32_e32 v4, vcc, v0, v4, vcc
	global_store_dwordx2 v[3:4], v[1:2], off
.LBB2_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z15benchmark_opt1bPjPKji
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
		.amdhsa_next_free_vgpr 12
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
.Lfunc_end2:
	.size	_Z15benchmark_opt1bPjPKji, .Lfunc_end2-_Z15benchmark_opt1bPjPKji
                                        ; -- End function
	.set _Z15benchmark_opt1bPjPKji.num_vgpr, 12
	.set _Z15benchmark_opt1bPjPKji.num_agpr, 0
	.set _Z15benchmark_opt1bPjPKji.numbered_sgpr, 7
	.set _Z15benchmark_opt1bPjPKji.private_seg_size, 0
	.set _Z15benchmark_opt1bPjPKji.uses_vcc, 1
	.set _Z15benchmark_opt1bPjPKji.uses_flat_scratch, 0
	.set _Z15benchmark_opt1bPjPKji.has_dyn_sized_stack, 0
	.set _Z15benchmark_opt1bPjPKji.has_recursion, 0
	.set _Z15benchmark_opt1bPjPKji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11364
; TotalNumSgprs: 11
; NumVgprs: 12
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 12
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
	.protected	_Z14benchmark_opt2PjPKji ; -- Begin function _Z14benchmark_opt2PjPKji
	.globl	_Z14benchmark_opt2PjPKji
	.p2align	8
	.type	_Z14benchmark_opt2PjPKji,@function
_Z14benchmark_opt2PjPKji:               ; @_Z14benchmark_opt2PjPKji
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dword s1, s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s1, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB3_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	v_mov_b32_e32 v4, 0xfdfeff00
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s3
	v_add_co_u32_e32 v1, vcc, s2, v1
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc
	global_load_dword v5, v[1:2], off
	s_mov_b32 s3, 0x80808080
	s_mov_b32 s2, 0x7070707
	v_mov_b32_e32 v1, 0xc080604
	v_mov_b32_e32 v2, 0x3020100
	v_mov_b32_e32 v3, 0xf4f8fafc
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v9, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_or_b32 v6, v6, s2, v9
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v9, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_or_b32 v6, v6, s2, v9
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v9, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_or_b32 v6, v6, s2, v9
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v9, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_or_b32 v6, v6, s2, v9
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v9, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_or_b32 v6, v6, s2, v9
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v9, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_or_b32 v6, v6, s2, v9
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v9, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_or_b32 v6, v6, s2, v9
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v9, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_or_b32 v6, v6, s2, v9
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v9, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_or_b32 v6, v6, s2, v9
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v8, 0x80808080, v5
	v_and_or_b32 v6, v6, s2, v8
	v_xor_b32_e32 v8, 0x80808080, v7
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v6, v8, v7
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_or_b32 v7, v8, s3, v7
	v_and_b32_e32 v5, 0x80808080, v5
	v_and_or_b32 v5, v6, s2, v5
	v_xor_b32_e32 v6, 0x80808080, v7
	v_xor_b32_e32 v8, 0x80808080, v5
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v1, v2, v5
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v3, v4, v8
	;;#ASMEND
	v_or_b32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v3, 1, v0
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_mov_b32_e32 v0, s1
	v_add_co_u32_e32 v3, vcc, s0, v3
	v_or_b32_e32 v1, v6, v7
	v_addc_co_u32_e32 v4, vcc, v0, v4, vcc
	global_store_dwordx2 v[3:4], v[1:2], off
.LBB3_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z14benchmark_opt2PjPKji
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
.Lfunc_end3:
	.size	_Z14benchmark_opt2PjPKji, .Lfunc_end3-_Z14benchmark_opt2PjPKji
                                        ; -- End function
	.set _Z14benchmark_opt2PjPKji.num_vgpr, 10
	.set _Z14benchmark_opt2PjPKji.num_agpr, 0
	.set _Z14benchmark_opt2PjPKji.numbered_sgpr, 7
	.set _Z14benchmark_opt2PjPKji.private_seg_size, 0
	.set _Z14benchmark_opt2PjPKji.uses_vcc, 1
	.set _Z14benchmark_opt2PjPKji.uses_flat_scratch, 0
	.set _Z14benchmark_opt2PjPKji.has_dyn_sized_stack, 0
	.set _Z14benchmark_opt2PjPKji.has_recursion, 0
	.set _Z14benchmark_opt2PjPKji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9784
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
	.protected	_Z14benchmark_opt6PjPKji ; -- Begin function _Z14benchmark_opt6PjPKji
	.globl	_Z14benchmark_opt6PjPKji
	.p2align	8
	.type	_Z14benchmark_opt6PjPKji,@function
_Z14benchmark_opt6PjPKji:               ; @_Z14benchmark_opt6PjPKji
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dword s1, s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s1, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB4_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	v_mov_b32_e32 v4, 0xfdfeff00
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s3
	v_add_co_u32_e32 v1, vcc, s2, v1
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc
	global_load_dword v5, v[1:2], off
	v_mov_b32_e32 v1, 0xc080604
	v_mov_b32_e32 v2, 0x3020100
	v_mov_b32_e32 v3, 0xf4f8fafc
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v9, 0x3020100, v9
	;;#ASMSTART
	v_perm_b32 v7, v7, v10, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v6, v11, v9
	;;#ASMEND
	v_xor_b32_e32 v5, v7, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v5, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v5, 0x4040404, v5
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v3, v4, v6
	;;#ASMEND
	v_or_b32_e32 v8, 0x3020100, v8
	v_or_b32_e32 v5, 0x3020100, v5
	;;#ASMSTART
	v_perm_b32 v1, v7, v9, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v3, v2, v5
	;;#ASMEND
	v_lshlrev_b32_e32 v3, 1, v0
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_mov_b32_e32 v0, s1
	v_add_co_u32_e32 v3, vcc, s0, v3
	v_addc_co_u32_e32 v4, vcc, v0, v4, vcc
	global_store_dwordx2 v[3:4], v[1:2], off
.LBB4_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z14benchmark_opt6PjPKji
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
		.amdhsa_next_free_vgpr 12
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
.Lfunc_end4:
	.size	_Z14benchmark_opt6PjPKji, .Lfunc_end4-_Z14benchmark_opt6PjPKji
                                        ; -- End function
	.set _Z14benchmark_opt6PjPKji.num_vgpr, 12
	.set _Z14benchmark_opt6PjPKji.num_agpr, 0
	.set _Z14benchmark_opt6PjPKji.numbered_sgpr, 7
	.set _Z14benchmark_opt6PjPKji.private_seg_size, 0
	.set _Z14benchmark_opt6PjPKji.uses_vcc, 1
	.set _Z14benchmark_opt6PjPKji.uses_flat_scratch, 0
	.set _Z14benchmark_opt6PjPKji.has_dyn_sized_stack, 0
	.set _Z14benchmark_opt6PjPKji.has_recursion, 0
	.set _Z14benchmark_opt6PjPKji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11364
; TotalNumSgprs: 11
; NumVgprs: 12
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 12
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
	.protected	_Z18verify_correctnessPjPi ; -- Begin function _Z18verify_correctnessPjPi
	.globl	_Z18verify_correctnessPjPi
	.p2align	8
	.type	_Z18verify_correctnessPjPi,@function
_Z18verify_correctnessPjPi:             ; @_Z18verify_correctnessPjPi
; %bb.0:
	s_load_dword s0, s[4:5], 0x1c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v7, s6, v0
	s_mov_b32 s0, 0x10000
	v_cmp_gt_i32_e32 vcc, s0, v7
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB5_15
; %bb.1:
	v_lshl_or_b32 v0, v7, 16, v7
	v_lshrrev_b32_e32 v1, 4, v0
	v_lshrrev_b32_e32 v3, 1, v0
	s_load_dwordx4 s[4:7], s[4:5], 0x0
	v_and_b32_e32 v4, 0x7070707, v0
	v_and_b32_e32 v6, 0x7070707, v1
	v_mov_b32_e32 v8, 0xc080604
	v_mov_b32_e32 v9, 0x3020100
	;;#ASMSTART
	v_perm_b32 v1, v8, v9, v4
	;;#ASMEND
	v_mov_b32_e32 v10, 0xf4f8fafc
	v_mov_b32_e32 v11, 0xfdfeff00
	;;#ASMSTART
	v_perm_b32 v2, v10, v11, v4
	;;#ASMEND
	v_and_b32_e32 v3, 0x4040404, v3
	v_lshrrev_b32_e32 v5, 5, v0
	v_or_b32_e32 v12, 0x3020100, v3
	;;#ASMSTART
	v_perm_b32 v1, v2, v1, v12
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v8, v9, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v10, v11, v6
	;;#ASMEND
	v_and_b32_e32 v5, 0x4040404, v5
	v_or_b32_e32 v13, 0x3020100, v5
	;;#ASMSTART
	v_perm_b32 v3, v3, v2, v13
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v8, v9, v4
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v10, v11, v4
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v14, v8, v9, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v10, v11, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v5, v2, v12
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v15, v14, v13
	;;#ASMEND
	v_cmp_ne_u32_e32 vcc, v1, v2
	v_cmp_ne_u32_e64 s[0:1], v3, v5
	;;#ASMSTART
	v_perm_b32 v14, v8, v9, v4
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v10, v11, v4
	;;#ASMEND
	s_or_b64 s[2:3], vcc, s[0:1]
	;;#ASMSTART
	v_perm_b32 v9, v8, v9, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v10, v11, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v4, v14, v12
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v6, v9, v13
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB5_8
; %bb.2:
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v6, s8, 0
	v_mbcnt_hi_u32_b32 v6, s9, v6
	v_cmp_eq_u32_e32 vcc, 0, v6
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB5_4
; %bb.3:
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v9, s8
	s_waitcnt lgkmcnt(0)
	global_atomic_add v6, v9, s[6:7]
.LBB5_4:
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[8:9], exec
	v_mbcnt_lo_u32_b32 v6, s8, 0
	v_mbcnt_hi_u32_b32 v6, s9, v6
	v_cmp_eq_u32_e32 vcc, 0, v6
                                        ; implicit-def: $vgpr9
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB5_6
; %bb.5:
	s_bcnt1_i32_b64 s8, s[8:9]
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, s8
	s_waitcnt lgkmcnt(0)
	global_atomic_add v9, v9, v10, s[6:7] offset:4 glc
.LBB5_6:
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v9
	v_add_u32_e32 v6, s2, v6
	v_cmp_gt_i32_e32 vcc, 10, v6
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB5_8
; %bb.7:
	v_mul_lo_u32 v9, v7, 6
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, s5
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 2, v[9:10]
	v_add_co_u32_e32 v9, vcc, s4, v9
	v_addc_co_u32_e32 v10, vcc, v6, v10, vcc
	v_mov_b32_e32 v6, 0xdead0001
	global_store_dwordx4 v[9:10], v[0:3], off
	global_store_dwordx2 v[9:10], v[5:6], off offset:16
.LBB5_8:                                ; %Flow66
	s_or_b64 exec, exec, s[0:1]
	v_cmp_ne_u32_e32 vcc, v1, v8
	v_cmp_ne_u32_e64 s[0:1], v3, v4
	s_or_b64 s[0:1], vcc, s[0:1]
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB5_15
; %bb.9:
	s_mov_b64 s[2:3], exec
	v_mbcnt_lo_u32_b32 v2, s2, 0
	v_mbcnt_hi_u32_b32 v2, s3, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB5_11
; %bb.10:
	s_bcnt1_i32_b64 s2, s[2:3]
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v5, s2
	s_waitcnt lgkmcnt(0)
	global_atomic_add v2, v5, s[6:7] offset:8
.LBB5_11:
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], exec
	v_mbcnt_lo_u32_b32 v2, s2, 0
	v_mbcnt_hi_u32_b32 v2, s3, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
                                        ; implicit-def: $vgpr5
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB5_13
; %bb.12:
	s_bcnt1_i32_b64 s2, s[2:3]
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, s2
	s_waitcnt lgkmcnt(0)
	global_atomic_add v5, v5, v6, s[6:7] offset:12 glc
.LBB5_13:
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s0, v5
	v_add_u32_e32 v2, s0, v2
	v_cmp_gt_i32_e32 vcc, 10, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB5_15
; %bb.14:
	v_mul_lo_u32 v5, v7, 6
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, s5
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v9, vcc, s4, v5
	v_addc_co_u32_e32 v10, vcc, v2, v6, vcc
	v_mov_b32_e32 v2, v8
	v_mov_b32_e32 v5, 0xdead0006
	global_store_dwordx4 v[9:10], v[0:3], off
	global_store_dwordx2 v[9:10], v[4:5], off offset:16
.LBB5_15:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z18verify_correctnessPjPi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
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
		.amdhsa_next_free_vgpr 16
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
.Lfunc_end5:
	.size	_Z18verify_correctnessPjPi, .Lfunc_end5-_Z18verify_correctnessPjPi
                                        ; -- End function
	.set _Z18verify_correctnessPjPi.num_vgpr, 16
	.set _Z18verify_correctnessPjPi.num_agpr, 0
	.set _Z18verify_correctnessPjPi.numbered_sgpr, 10
	.set _Z18verify_correctnessPjPi.private_seg_size, 0
	.set _Z18verify_correctnessPjPi.uses_vcc, 1
	.set _Z18verify_correctnessPjPi.uses_flat_scratch, 0
	.set _Z18verify_correctnessPjPi.has_dyn_sized_stack, 0
	.set _Z18verify_correctnessPjPi.has_recursion, 0
	.set _Z18verify_correctnessPjPi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 768
; TotalNumSgprs: 14
; NumVgprs: 16
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 16
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
	.type	__hip_cuid_a90dab6e147871d1,@object ; @__hip_cuid_a90dab6e147871d1
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_a90dab6e147871d1
__hip_cuid_a90dab6e147871d1:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a90dab6e147871d1, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a90dab6e147871d1
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  write_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
    .name:           _Z18benchmark_baselinePjPKji
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z18benchmark_baselinePjPKji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     11
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  write_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
    .name:           _Z15benchmark_opt1aPjPKji
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z15benchmark_opt1aPjPKji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     13
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  write_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
    .name:           _Z15benchmark_opt1bPjPKji
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z15benchmark_opt1bPjPKji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     12
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  write_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
    .name:           _Z14benchmark_opt2PjPKji
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z14benchmark_opt2PjPKji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .actual_access:  write_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
    .name:           _Z14benchmark_opt6PjPKji
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z14benchmark_opt6PjPKji.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     12
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
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z18verify_correctnessPjPi
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z18verify_correctnessPjPi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     16
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
