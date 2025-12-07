	.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z9bench_refPjPKji      ; -- Begin function _Z9bench_refPjPKji
	.globl	_Z9bench_refPjPKji
	.p2align	8
	.type	_Z9bench_refPjPKji,@function
_Z9bench_refPjPKji:                     ; @_Z9bench_refPjPKji
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z9bench_refPjPKji
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
	.size	_Z9bench_refPjPKji, .Lfunc_end0-_Z9bench_refPjPKji
                                        ; -- End function
	.set _Z9bench_refPjPKji.num_vgpr, 10
	.set _Z9bench_refPjPKji.num_agpr, 0
	.set _Z9bench_refPjPKji.numbered_sgpr, 7
	.set _Z9bench_refPjPKji.private_seg_size, 0
	.set _Z9bench_refPjPKji.uses_vcc, 1
	.set _Z9bench_refPjPKji.uses_flat_scratch, 0
	.set _Z9bench_refPjPKji.has_dyn_sized_stack, 0
	.set _Z9bench_refPjPKji.has_recursion, 0
	.set _Z9bench_refPjPKji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10576
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
	.protected	_Z7bench_hPjPKji        ; -- Begin function _Z7bench_hPjPKji
	.globl	_Z7bench_hPjPKji
	.p2align	8
	.type	_Z7bench_hPjPKji,@function
_Z7bench_hPjPKji:                       ; @_Z7bench_hPjPKji
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
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	v_and_b32_e32 v9, 0x4040404, v9
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshrrev_b32_e32 v8, 1, v5
	v_lshrrev_b32_e32 v9, 5, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v10, v1, v2, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x4040404, v8
	;;#ASMSTART
	v_perm_b32 v11, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	v_lshrrev_b32_e32 v10, 1, v5
	v_lshrrev_b32_e32 v11, 5, v5
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
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
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v6, 0x7070707, v6
	;;#ASMSTART
	v_perm_b32 v8, v1, v2, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v1, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v3, v4, v6
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 1, v5
	v_lshrrev_b32_e32 v5, 5, v5
	v_and_b32_e32 v4, 0x4040404, v4
	v_and_b32_e32 v5, 0x4040404, v5
	v_or_b32_e32 v4, 0x3020100, v4
	v_or_b32_e32 v5, 0x3020100, v5
	;;#ASMSTART
	v_perm_b32 v1, v1, v8, v4
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
.LBB1_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z7bench_hPjPKji
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
.Lfunc_end1:
	.size	_Z7bench_hPjPKji, .Lfunc_end1-_Z7bench_hPjPKji
                                        ; -- End function
	.set _Z7bench_hPjPKji.num_vgpr, 12
	.set _Z7bench_hPjPKji.num_agpr, 0
	.set _Z7bench_hPjPKji.numbered_sgpr, 7
	.set _Z7bench_hPjPKji.private_seg_size, 0
	.set _Z7bench_hPjPKji.uses_vcc, 1
	.set _Z7bench_hPjPKji.uses_flat_scratch, 0
	.set _Z7bench_hPjPKji.has_dyn_sized_stack, 0
	.set _Z7bench_hPjPKji.has_recursion, 0
	.set _Z7bench_hPjPKji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11760
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
	.protected	_Z7bench_iPjPKji        ; -- Begin function _Z7bench_iPjPKji
	.globl	_Z7bench_iPjPKji
	.p2align	8
	.type	_Z7bench_iPjPKji,@function
_Z7bench_iPjPKji:                       ; @_Z7bench_iPjPKji
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_lshlrev_b32_e32 v8, 4, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v5, 0x80808080, v5
	v_and_or_b32 v7, v8, s3, v7
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
.LBB2_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z7bench_iPjPKji
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
.Lfunc_end2:
	.size	_Z7bench_iPjPKji, .Lfunc_end2-_Z7bench_iPjPKji
                                        ; -- End function
	.set _Z7bench_iPjPKji.num_vgpr, 10
	.set _Z7bench_iPjPKji.num_agpr, 0
	.set _Z7bench_iPjPKji.numbered_sgpr, 7
	.set _Z7bench_iPjPKji.private_seg_size, 0
	.set _Z7bench_iPjPKji.uses_vcc, 1
	.set _Z7bench_iPjPKji.uses_flat_scratch, 0
	.set _Z7bench_iPjPKji.has_dyn_sized_stack, 0
	.set _Z7bench_iPjPKji.has_recursion, 0
	.set _Z7bench_iPjPKji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10576
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
	.protected	_Z7bench_lPjPKji        ; -- Begin function _Z7bench_lPjPKji
	.globl	_Z7bench_lPjPKji
	.p2align	8
	.type	_Z7bench_lPjPKji,@function
_Z7bench_lPjPKji:                       ; @_Z7bench_lPjPKji
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
	v_mov_b32_e32 v1, 0xc080604
	v_mov_b32_e32 v2, 0x3020100
	v_mov_b32_e32 v3, 0xf4f8fafc
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x8080808, v5
	v_and_b32_e32 v9, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v7, v8, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v9
	;;#ASMEND
	v_xor_b32_e32 v8, 0x80808080, v7
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x8080808, v5
	v_and_b32_e32 v9, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v7, v8, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v9
	;;#ASMEND
	v_xor_b32_e32 v8, 0x80808080, v7
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x8080808, v5
	v_and_b32_e32 v9, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v7, v8, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v9
	;;#ASMEND
	v_xor_b32_e32 v8, 0x80808080, v7
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x8080808, v5
	v_and_b32_e32 v9, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v7, v8, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v9
	;;#ASMEND
	v_xor_b32_e32 v8, 0x80808080, v7
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x8080808, v5
	v_and_b32_e32 v9, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v7, v8, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v9
	;;#ASMEND
	v_xor_b32_e32 v8, 0x80808080, v7
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x8080808, v5
	v_and_b32_e32 v9, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v7, v8, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v9
	;;#ASMEND
	v_xor_b32_e32 v8, 0x80808080, v7
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x8080808, v5
	v_and_b32_e32 v9, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v7, v8, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v9
	;;#ASMEND
	v_xor_b32_e32 v8, 0x80808080, v7
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x8080808, v5
	v_and_b32_e32 v9, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v7, v8, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v9
	;;#ASMEND
	v_xor_b32_e32 v8, 0x80808080, v7
	;;#ASMSTART
	v_perm_b32 v7, v1, v2, v7
	;;#ASMEND
	v_xor_b32_e32 v9, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v8, v3, v4, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v1, v2, v6
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x8080808, v5
	v_and_b32_e32 v9, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v7, v8, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v9
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	v_and_b32_e32 v9, 0x8080808, v5
	;;#ASMSTART
	v_lshl_add_u32 v7, v9, 4, v7
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
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
	v_perm_b32 v9, v3, v4, v9
	;;#ASMEND
	v_or_b32_e32 v7, v8, v7
	v_or_b32_e32 v6, v9, v6
	v_xor_b32_e32 v6, v7, v6
	v_xor_b32_e32 v5, v6, v5
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v7, 0x7070707, v5
	v_and_b32_e32 v5, 0x8080808, v5
	v_and_b32_e32 v8, 0x7070707, v6
	v_and_b32_e32 v6, 0x8080808, v6
	;;#ASMSTART
	v_lshl_add_u32 v5, v5, 4, v7
	;;#ASMEND
	v_xor_b32_e32 v7, 0x80808080, v5
	;;#ASMSTART
	v_lshl_add_u32 v6, v6, 4, v8
	;;#ASMEND
	v_xor_b32_e32 v8, 0x80808080, v6
	;;#ASMSTART
	v_perm_b32 v5, v1, v2, v5
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v3, v4, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v2, v1, v2, v6
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
	v_or_b32_e32 v1, v7, v5
	v_addc_co_u32_e32 v4, vcc, v0, v4, vcc
	global_store_dwordx2 v[3:4], v[1:2], off
.LBB3_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z7bench_lPjPKji
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
	.size	_Z7bench_lPjPKji, .Lfunc_end3-_Z7bench_lPjPKji
                                        ; -- End function
	.set _Z7bench_lPjPKji.num_vgpr, 10
	.set _Z7bench_lPjPKji.num_agpr, 0
	.set _Z7bench_lPjPKji.numbered_sgpr, 7
	.set _Z7bench_lPjPKji.private_seg_size, 0
	.set _Z7bench_lPjPKji.uses_vcc, 1
	.set _Z7bench_lPjPKji.uses_flat_scratch, 0
	.set _Z7bench_lPjPKji.has_dyn_sized_stack, 0
	.set _Z7bench_lPjPKji.has_recursion, 0
	.set _Z7bench_lPjPKji.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 11760
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
	.protected	_Z6verifyPi             ; -- Begin function _Z6verifyPi
	.globl	_Z6verifyPi
	.p2align	8
	.type	_Z6verifyPi,@function
_Z6verifyPi:                            ; @_Z6verifyPi
; %bb.0:
	s_load_dword s0, s[4:5], 0x14
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	v_add_u32_e32 v0, s6, v0
	s_mov_b32 s0, 0x10000
	v_cmp_gt_i32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB4_10
; %bb.1:
	v_lshl_or_b32 v2, v0, 16, v0
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v4, 0x7070707, v2
	v_lshlrev_b32_e32 v0, 4, v2
	s_mov_b32 s0, 0x80808080
	v_and_b32_e32 v10, 0x7070707, v3
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	v_and_or_b32 v5, v0, s0, v4
	v_and_or_b32 v8, v2, s0, v10
	v_xor_b32_e32 v7, 0x80808080, v5
	v_xor_b32_e32 v9, 0x80808080, v8
	v_mov_b32_e32 v11, 0xc080604
	v_mov_b32_e32 v12, 0x3020100
	;;#ASMSTART
	v_perm_b32 v0, v11, v12, v5
	;;#ASMEND
	v_mov_b32_e32 v13, 0xf4f8fafc
	v_mov_b32_e32 v14, 0xfdfeff00
	;;#ASMSTART
	v_perm_b32 v1, v13, v14, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v6, v11, v12, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v13, v14, v9
	;;#ASMEND
	v_lshrrev_b32_e32 v18, 1, v2
	v_lshrrev_b32_e32 v19, 5, v2
	v_or_b32_e32 v0, v1, v0
	v_or_b32_e32 v1, v15, v6
	;;#ASMSTART
	v_perm_b32 v6, v11, v12, v4
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v11, v12, v10
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v16, v13, v14, v4
	;;#ASMEND
	v_and_b32_e32 v18, 0x4040404, v18
	v_and_b32_e32 v19, 0x4040404, v19
	;;#ASMSTART
	v_perm_b32 v17, v13, v14, v10
	;;#ASMEND
	v_or_b32_e32 v18, 0x3020100, v18
	v_or_b32_e32 v19, 0x3020100, v19
	;;#ASMSTART
	v_perm_b32 v16, v16, v6, v18
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v15, v17, v15, v19
	;;#ASMEND
	v_and_b32_e32 v2, 0x8080808, v2
	v_and_b32_e32 v3, 0x8080808, v3
	v_cmp_ne_u32_e32 vcc, v16, v0
	v_cmp_ne_u32_e64 s[0:1], v15, v1
	;;#ASMSTART
	v_perm_b32 v6, v11, v12, v5
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v7, v13, v14, v7
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v8, v11, v12, v8
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v9, v13, v14, v9
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v2, v2, 4, v4
	;;#ASMEND
	;;#ASMSTART
	v_lshl_add_u32 v4, v3, 4, v10
	;;#ASMEND
	v_xor_b32_e32 v3, 0x80808080, v2
	v_xor_b32_e32 v5, 0x80808080, v4
	s_or_b64 s[4:5], vcc, s[0:1]
	;;#ASMSTART
	v_perm_b32 v2, v11, v12, v2
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v3, v13, v14, v3
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v4, v11, v12, v4
	;;#ASMEND
	;;#ASMSTART
	v_perm_b32 v5, v13, v14, v5
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB4_4
; %bb.2:
	s_mov_b64 s[4:5], exec
	v_mbcnt_lo_u32_b32 v10, s4, 0
	v_mbcnt_hi_u32_b32 v10, s5, v10
	v_cmp_eq_u32_e32 vcc, 0, v10
	s_and_b64 s[6:7], exec, vcc
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB4_4
; %bb.3:
	s_bcnt1_i32_b64 s4, s[4:5]
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, s4
	s_waitcnt lgkmcnt(0)
	global_atomic_add v10, v11, s[2:3]
.LBB4_4:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v6, v7, v6
	v_or_b32_e32 v7, v9, v8
	v_cmp_ne_u32_e32 vcc, v6, v0
	v_cmp_ne_u32_e64 s[0:1], v7, v1
	s_or_b64 s[4:5], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB4_7
; %bb.5:
	s_mov_b64 s[4:5], exec
	v_mbcnt_lo_u32_b32 v6, s4, 0
	v_mbcnt_hi_u32_b32 v6, s5, v6
	v_cmp_eq_u32_e32 vcc, 0, v6
	s_and_b64 s[6:7], exec, vcc
	s_mov_b64 exec, s[6:7]
	s_cbranch_execz .LBB4_7
; %bb.6:
	s_bcnt1_i32_b64 s4, s[4:5]
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, s4
	s_waitcnt lgkmcnt(0)
	global_atomic_add v6, v7, s[2:3] offset:4
.LBB4_7:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, v3, v2
	v_or_b32_e32 v3, v5, v4
	v_cmp_ne_u32_e32 vcc, v2, v0
	v_cmp_ne_u32_e64 s[0:1], v3, v1
	s_or_b64 s[0:1], vcc, s[0:1]
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB4_10
; %bb.8:
	s_mov_b64 s[0:1], exec
	v_mbcnt_lo_u32_b32 v0, s0, 0
	v_mbcnt_hi_u32_b32 v0, s1, v0
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_and_b64 s[4:5], exec, vcc
	s_mov_b64 exec, s[4:5]
	s_cbranch_execz .LBB4_10
; %bb.9:
	s_bcnt1_i32_b64 s0, s[0:1]
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, s0
	s_waitcnt lgkmcnt(0)
	global_atomic_add v0, v1, s[2:3] offset:8
.LBB4_10:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z6verifyPi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 264
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
.Lfunc_end4:
	.size	_Z6verifyPi, .Lfunc_end4-_Z6verifyPi
                                        ; -- End function
	.set _Z6verifyPi.num_vgpr, 20
	.set _Z6verifyPi.num_agpr, 0
	.set _Z6verifyPi.numbered_sgpr, 8
	.set _Z6verifyPi.private_seg_size, 0
	.set _Z6verifyPi.uses_vcc, 1
	.set _Z6verifyPi.uses_flat_scratch, 0
	.set _Z6verifyPi.has_dyn_sized_stack, 0
	.set _Z6verifyPi.has_recursion, 0
	.set _Z6verifyPi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 680
; TotalNumSgprs: 12
; NumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 12
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
	.type	__hip_cuid_87282b2f4982cd8e,@object ; @__hip_cuid_87282b2f4982cd8e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_87282b2f4982cd8e
__hip_cuid_87282b2f4982cd8e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_87282b2f4982cd8e, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_87282b2f4982cd8e
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
    .name:           _Z9bench_refPjPKji
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z9bench_refPjPKji.kd
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
    .name:           _Z7bench_hPjPKji
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z7bench_hPjPKji.kd
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
    .name:           _Z7bench_iPjPKji
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z7bench_iPjPKji.kd
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
    .name:           _Z7bench_lPjPKji
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z7bench_lPjPKji.kd
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
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z6verifyPi
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z6verifyPi.kd
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
