/*
 * Streebog GOST R 34.10-2012 CUDA implementation.
 *
 * https://tools.ietf.org/html/rfc6986
 * https://en.wikipedia.org/wiki/Streebog
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * @author   Tanguy Pruvot - 2015
 * @author   Alexis Provos - 2016
 */

// Further improved with shared memory partial utilization
// Merged implementation to decrease shared memory's bottleneck
// Tested under CUDA7.5 toolkit for cp 5.0/5.2

#include "miner.h"

#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "streebog_arrays.cuh"

__constant__ 
static uint2 keccak_round_constants[24] = {
		{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 },	{ 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
		{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 },	{ 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
		{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 },	{ 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
		{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 },	{ 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
		{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 },	{ 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
		{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 },	{ 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

__device__ __forceinline__
static void GOST_FS(const uint2 shared[8][256],const uint2 *const __restrict__ state,uint2* return_state){

	return_state[0] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44440)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44440)])
			^ shared[2][__byte_perm(state[5].x,0,0x44440)]
			^ shared[3][__byte_perm(state[4].x,0,0x44440)]
			^ shared[4][__byte_perm(state[3].x,0,0x44440)]
			^ shared[5][__byte_perm(state[2].x,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44440)])
			^ shared[6][__byte_perm(state[1].x,0,0x44440)];

	return_state[1] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44441)])
			^ shared[2][__byte_perm(state[5].x,0,0x44441)]
			^ shared[3][__byte_perm(state[4].x,0,0x44441)]
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44441)])
 			^ shared[4][__byte_perm(state[3].x,0,0x44441)]
			^ shared[5][__byte_perm(state[2].x,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44441)])
			^ shared[6][__byte_perm(state[1].x,0,0x44441)];

	return_state[2] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44442)])
			^ shared[2][__byte_perm(state[5].x,0,0x44442)]
			^ shared[3][__byte_perm(state[4].x,0,0x44442)]
			^ shared[4][__byte_perm(state[3].x,0,0x44442)]
			^ shared[5][__byte_perm(state[2].x,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44442)])
			^ shared[6][__byte_perm(state[1].x,0,0x44442)];

	return_state[3] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44443)])
			^ shared[1][__byte_perm(state[6].x,0,0x44443)]
			^ shared[2][__byte_perm(state[5].x,0,0x44443)]
			^ shared[3][__byte_perm(state[4].x,0,0x44443)]
			^ __ldg(&T42[__byte_perm(state[3].x,0,0x44443)])
			^ shared[5][__byte_perm(state[2].x,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44443)])
			^ shared[6][__byte_perm(state[1].x,0,0x44443)];

	return_state[4] =  __ldg(&T02[__byte_perm(state[7].y,0,0x44440)])
			^ shared[1][__byte_perm(state[6].y,0,0x44440)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44440)])
			^ shared[3][__byte_perm(state[4].y,0,0x44440)]
			^ shared[4][__byte_perm(state[3].y,0,0x44440)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44440)])
			^ shared[5][__byte_perm(state[2].y,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44440)]);

	return_state[5] = __ldg(&T02[__byte_perm(state[7].y,0,0x44441)])
			^ shared[2][__byte_perm(state[5].y,0,0x44441)]
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44441)])
			^ shared[3][__byte_perm(state[4].y,0,0x44441)]
			^ shared[4][__byte_perm(state[3].y,0,0x44441)]
			^ shared[5][__byte_perm(state[2].y,0,0x44441)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44441)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44441)]);

	return_state[6] =  __ldg(&T02[__byte_perm(state[7].y,0,0x44442)])
			^ shared[1][__byte_perm(state[6].y,0,0x44442)]
			^ shared[2][__byte_perm(state[5].y,0,0x44442)]
			^ shared[3][__byte_perm(state[4].y,0,0x44442)]
			^ shared[4][__byte_perm(state[3].y,0,0x44442)]
			^ shared[5][__byte_perm(state[2].y,0,0x44442)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44442)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44442)]);

	return_state[7] =  __ldg(&T02[__byte_perm(state[7].y,0,0x44443)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44443)])
			^ shared[2][__byte_perm(state[5].y,0,0x44443)]
			^ shared[3][__byte_perm(state[4].y,0,0x44443)]
			^ shared[4][__byte_perm(state[3].y,0,0x44443)]
			^ shared[5][__byte_perm(state[2].y,0,0x44443)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44443)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44443)]);
}

__device__ __forceinline__
static void GOST_FS_LDG(const uint2 shared[8][256],const uint2 *const __restrict__ state,uint2* return_state){

	return_state[0] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44440)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44440)])
			^ shared[2][__byte_perm(state[5].x,0,0x44440)]
			^ shared[3][__byte_perm(state[4].x,0,0x44440)]
			^ shared[4][__byte_perm(state[3].x,0,0x44440)]
			^ shared[5][__byte_perm(state[2].x,0,0x44440)]
			^ shared[6][__byte_perm(state[1].x,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44440)]);

	return_state[1] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44441)])
			^ shared[1][__byte_perm(state[6].x,0,0x44441)]
			^ shared[2][__byte_perm(state[5].x,0,0x44441)]
			^ shared[3][__byte_perm(state[4].x,0,0x44441)]
			^ shared[4][__byte_perm(state[3].x,0,0x44441)]
			^ shared[5][__byte_perm(state[2].x,0,0x44441)]
			^ shared[6][__byte_perm(state[1].x,0,0x44441)] 
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44441)]);

	return_state[2] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44442)])
			^ shared[2][__byte_perm(state[5].x,0,0x44442)]
			^ shared[3][__byte_perm(state[4].x,0,0x44442)]
			^ shared[4][__byte_perm(state[3].x,0,0x44442)]
			^ shared[5][__byte_perm(state[2].x,0,0x44442)]
			^ shared[6][__byte_perm(state[1].x,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44442)]);

	return_state[3] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44443)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44443)])
			^ shared[2][__byte_perm(state[5].x,0,0x44443)]
			^ shared[3][__byte_perm(state[4].x,0,0x44443)]
			^ shared[4][__byte_perm(state[3].x,0,0x44443)]
			^ shared[5][__byte_perm(state[2].x,0,0x44443)]
			^ __ldg(&T62[__byte_perm(state[1].x,0,0x44443)])
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44443)]);

	return_state[4] = __ldg(&T02[__byte_perm(state[7].y,0,0x44440)])
			^ shared[1][__byte_perm(state[6].y,0,0x44440)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44440)])
			^ shared[3][__byte_perm(state[4].y,0,0x44440)]
			^ shared[4][__byte_perm(state[3].y,0,0x44440)]
			^ shared[5][__byte_perm(state[2].y,0,0x44440)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44440)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44440)]);

	return_state[5] = __ldg(&T02[__byte_perm(state[7].y,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44441)])
			^ shared[2][__byte_perm(state[5].y,0,0x44441)]
			^ shared[3][__byte_perm(state[4].y,0,0x44441)]
			^ shared[4][__byte_perm(state[3].y,0,0x44441)]
			^ shared[5][__byte_perm(state[2].y,0,0x44441)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44441)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44441)]);

	return_state[6] =  __ldg(&T02[__byte_perm(state[7].y,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44442)])
			^ shared[2][__byte_perm(state[5].y,0,0x44442)]
			^ shared[3][__byte_perm(state[4].y,0,0x44442)]
			^ shared[4][__byte_perm(state[3].y,0,0x44442)]
			^ shared[5][__byte_perm(state[2].y,0,0x44442)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44442)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44442)]);

	return_state[7] =  __ldg(&T02[__byte_perm(state[7].y,0,0x44443)])
			^ shared[1][__byte_perm(state[6].y,0,0x44443)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44443)])
			^ shared[3][__byte_perm(state[4].y,0,0x44443)]
			^ shared[4][__byte_perm(state[3].y,0,0x44443)]
			^ shared[5][__byte_perm(state[2].y,0,0x44443)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44443)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44443)]);
}

__device__ __forceinline__
static void GOST_E12(const uint2 shared[8][256],uint2 *const __restrict__ K, uint2 *const __restrict__ state){

	uint2 t[ 8];
//	#pragma unroll 2
	for(int i=0; i<12; i++){
		GOST_FS(shared,state, t);
		
		#pragma unroll 8
		for(int j=0;j<8;j++)
			K[ j] ^= *(uint2*)&CC[i][j];
		
		#pragma unroll 8
		for(int j=0;j<8;j++)
			state[ j] = t[ j];
		
		GOST_FS_LDG(shared,K, t);

		#pragma unroll 8
		for(int j=0;j<8;j++)
			state[ j]^= t[ j];

		#pragma unroll 8
		for(int j=0;j<8;j++)
			K[ j] = t[ j];
	}
}

__device__ __forceinline__
static void keccak_kernel(uint2* s){
	uint2 u[5],t[5], v, w;

	/*theta*/
	t[ 0] = vectorize(devectorize(s[ 0])^devectorize(s[ 5]));
	t[ 1] = vectorize(devectorize(s[ 1])^devectorize(s[ 6]));
	t[ 2] = vectorize(devectorize(s[ 2])^devectorize(s[ 7]));
	t[ 3] = vectorize(devectorize(s[ 3])^devectorize(s[ 8]));
	t[ 4] = s[4];
		
	/*theta*/
	#pragma unroll 5
	for(int j=0;j<5;j++){
		u[ j] = ROL2(t[ j], 1);
	}
	
	s[ 4] = xor3x(s[ 4], t[3], u[ 0]);
	s[24] = s[19] = s[14] = s[ 9] = t[ 3] ^ u[ 0];

	s[ 0] = xor3x(s[ 0], t[4], u[ 1]);
	s[ 5] = xor3x(s[ 5], t[4], u[ 1]);
	s[20] = s[15] = s[10] = t[4] ^ u[ 1];

	s[ 1] = xor3x(s[ 1], t[0], u[ 2]);
	s[ 6] = xor3x(s[ 6], t[0], u[ 2]);
	s[21] = s[16] = s[11] = t[0] ^ u[ 2];
		
	s[ 2] = xor3x(s[ 2], t[1], u[ 3]);
	s[ 7] = xor3x(s[ 7], t[1], u[ 3]);
	s[22] = s[17] = s[12] = t[1] ^ u[ 3];
		
	s[ 3] = xor3x(s[ 3], t[2], u[ 4]);s[ 8] = xor3x(s[ 8], t[2], u[ 4]);
	s[23] = s[18] = s[13] = t[2] ^ u[ 4];
	/* rho pi: b[..] = rotl(a[..], ..) */
	v = s[1];
	s[1]  = ROL2(s[6], 44);
	s[6]  = ROL2(s[9], 20);
	s[9]  = ROL2(s[22], 61);
	s[22] = ROL2(s[14], 39);
	s[14] = ROL2(s[20], 18);
	s[20] = ROL2(s[2], 62);
	s[2]  = ROL2(s[12], 43);
	s[12] = ROL2(s[13], 25);
	s[13] = ROL8(s[19]);
	s[19] = ROR8(s[23]);
	s[23] = ROL2(s[15], 41);
	s[15] = ROL2(s[4], 27);
	s[4]  = ROL2(s[24], 14);
	s[24] = ROL2(s[21], 2);
	s[21] = ROL2(s[8], 55);
	s[8]  = ROL2(s[16], 45);
	s[16] = ROL2(s[5], 36);
	s[5]  = ROL2(s[3], 28);
	s[3]  = ROL2(s[18], 21);
	s[18] = ROL2(s[17], 15);
	s[17] = ROL2(s[11], 10);
	s[11] = ROL2(s[7], 6);
	s[7]  = ROL2(s[10], 3);
	s[10] = ROL2(v, 1);
	/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
	#pragma unroll 5
	for(int j=0;j<25;j+=5){
		v=s[j];w=s[j + 1];s[j] = chi(v,w,s[j+2]);s[j+1] = chi(w,s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
	}
	/* iota: a[0,0] ^= round constant */
	s[0] ^= keccak_round_constants[ 0];

	for (int i = 1; i < 23; i++) {
		/*theta*/
		#pragma unroll 5
		for(int j=0;j<5;j++){
			t[ j] = vectorize(xor5(devectorize(s[ j]),devectorize(s[j+5]),devectorize(s[j+10]),devectorize(s[j+15]),devectorize(s[j+20])));
		}
		/*theta*/
		#pragma unroll 5
		for(int j=0;j<5;j++){
			u[ j] = ROL2(t[ j], 1);
		}
		s[ 4] = xor3x(s[ 4], t[3], u[ 0]);s[ 9] = xor3x(s[ 9], t[3], u[ 0]);s[14] = xor3x(s[14], t[3], u[ 0]);s[19] = xor3x(s[19], t[3], u[ 0]);s[24] = xor3x(s[24], t[3], u[ 0]);
		s[ 0] = xor3x(s[ 0], t[4], u[ 1]);s[ 5] = xor3x(s[ 5], t[4], u[ 1]);s[10] = xor3x(s[10], t[4], u[ 1]);s[15] = xor3x(s[15], t[4], u[ 1]);s[20] = xor3x(s[20], t[4], u[ 1]);
		s[ 1] = xor3x(s[ 1], t[0], u[ 2]);s[ 6] = xor3x(s[ 6], t[0], u[ 2]);s[11] = xor3x(s[11], t[0], u[ 2]);s[16] = xor3x(s[16], t[0], u[ 2]);s[21] = xor3x(s[21], t[0], u[ 2]);
		s[ 2] = xor3x(s[ 2], t[1], u[ 3]);s[ 7] = xor3x(s[ 7], t[1], u[ 3]);s[12] = xor3x(s[12], t[1], u[ 3]);s[17] = xor3x(s[17], t[1], u[ 3]);s[22] = xor3x(s[22], t[1], u[ 3]);
		s[ 3] = xor3x(s[ 3], t[2], u[ 4]);s[ 8] = xor3x(s[ 8], t[2], u[ 4]);s[13] = xor3x(s[13], t[2], u[ 4]);s[18] = xor3x(s[18], t[2], u[ 4]);s[23] = xor3x(s[23], t[2], u[ 4]);

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1]  = ROL2(s[6], 44);
		s[6]  = ROL2(s[9], 20);
		s[9]  = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2]  = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL8(s[19]);
		s[19] = ROR8(s[23]);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4]  = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8]  = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5]  = ROL2(s[3], 28);
		s[3]  = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7]  = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		#pragma unroll 5
		for(int j=0;j<25;j+=5){
			v=s[j];w=s[j + 1];s[j] = chi(v,w,s[j+2]);s[j+1] = chi(w,s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
		}
		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[i];
	}
	//theta
	#pragma unroll 5
	for(int j=0;j<5;j++){
		t[ j] = xor3x(xor3x(s[j+0],s[j+5],s[j+10]),s[j+15],s[j+20]);
	}
	//theta
	#pragma unroll 5
	for(int j=0;j<5;j++){
		u[ j] = ROL2(t[ j], 1);
	}
	s[ 9] = xor3x(s[ 9], t[3], u[ 0]);
	s[24] = xor3x(s[24], t[3], u[ 0]);
	s[ 0] = xor3x(s[ 0], t[4], u[ 1]);
	s[10] = xor3x(s[10], t[4], u[ 1]);
	s[ 6] = xor3x(s[ 6], t[0], u[ 2]);
	s[16] = xor3x(s[16], t[0], u[ 2]);
	s[12] = xor3x(s[12], t[1], u[ 3]);
	s[22] = xor3x(s[22], t[1], u[ 3]);
	s[ 3] = xor3x(s[ 3], t[2], u[ 4]);
	s[18] = xor3x(s[18], t[2], u[ 4]);
	// rho pi: b[..] = rotl(a[..], ..)
	s[ 1]  = ROL2(s[ 6], 44);
	s[ 2]  = ROL2(s[12], 43);
	s[ 5]  = ROL2(s[ 3], 28);
	s[ 7]  = ROL2(s[10], 3);
	s[ 3]  = ROL2(s[18], 21);
	s[ 4]  = ROL2(s[24], 14);
	s[ 6]  = ROL2(s[ 9], 20);
	s[ 8]  = ROL2(s[16], 45);
	s[ 9]  = ROL2(s[22], 61);
	// chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2]
	v=s[ 0];w=s[ 1];s[ 0] = chi(v,w,s[ 2]);s[ 1] = chi(w,s[ 2],s[ 3]);s[ 2]=chi(s[ 2],s[ 3],s[ 4]);s[ 3]=chi(s[ 3],s[ 4],v);s[ 4]=chi(s[ 4],v,w);		
	v=s[ 5];w=s[ 6];s[ 5] = chi(v,w,s[ 7]);s[ 6] = chi(w,s[ 7],s[ 8]);s[ 7]=chi(s[ 7],s[ 8],s[ 9]);
	// iota: a[0,0] ^= round constant
	s[0] ^= keccak_round_constants[23];
}

__device__ __forceinline__
static void streebog_kernel(const uint2 shared[8][256],uint2* s){

	uint2 buf[8], t[8], temp[8],K0[8];

	K0[0] = vectorize(0x74a5d4ce2efc83b3);

	#pragma unroll 8
	for(int i=0;i<8;i++){
		buf[ i] = K0[ 0] ^ s[ i];
	}

//	#pragma unroll 11
	for(int i=0; i<12; i++){
		GOST_FS(shared,buf, temp);
		#pragma unroll 8
		for(int j=0;j<8;j++){
			buf[ j] = temp[ j] ^ *(uint2*)&precomputed_values[i][j];
		}
	}
	#pragma unroll 8
	for(int j=0;j<8;j++){
		buf[ j]^= s[ j];
	}
	#pragma unroll 8
	for(int j=0;j<8;j++){
		K0[ j] = buf[ j];
	}
	
	K0[7].y ^= 0x00020000;
	
	GOST_FS(shared,K0, t);

	#pragma unroll 8
	for(int i=0;i<8;i++)
		K0[ i] = t[ i];
		
	t[7].y ^= 0x01000000;

	GOST_E12(shared,K0, t);

	#pragma unroll 8
	for(int j=0;j<8;j++)
		buf[ j] ^= t[ j];

	buf[7].y ^= 0x01000000;

	GOST_FS(shared,buf,K0);

	buf[7].y ^= 0x00020000;

	#pragma unroll 8
	for(int j=0;j<8;j++)
		t[ j] = K0[ j];
		
	t[7].y ^= 0x00020000;

	GOST_E12(shared,K0, t);

	#pragma unroll 8
	for(int j=0;j<8;j++)
		buf[ j] ^= t[ j];
		
	GOST_FS(shared,buf,K0); // K = F(h)

	s[7]+= vectorize(0x0100000000000000);

	#pragma unroll 8
	for(int j=0;j<8;j++)
		t[ j] = K0[ j] ^ s[ j];

	GOST_E12(shared,K0, t);
	
	#pragma unroll 8
	for(int i=0;i<8;i++)
		s[i] = s[ i] ^ buf[ i] ^ t[ i];
}

#define TPB 256
__global__
__launch_bounds__(TPB,3)
void keccak_streebog_gpu_hash_64(const uint32_t threads,uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint2 s[25];
	
	__shared__ uint2 shared[8][256];
	shared[0][threadIdx.x] = __ldg(&T02[threadIdx.x]);
	shared[1][threadIdx.x] = __ldg(&T12[threadIdx.x]);
	shared[2][threadIdx.x] = __ldg(&T22[threadIdx.x]);
	shared[3][threadIdx.x] = __ldg(&T32[threadIdx.x]);
	shared[4][threadIdx.x] = __ldg(&T42[threadIdx.x]);
	shared[5][threadIdx.x] = __ldg(&T52[threadIdx.x]);
	shared[6][threadIdx.x] = __ldg(&T62[threadIdx.x]);
	shared[7][threadIdx.x] = __ldg(&T72[threadIdx.x]);

//	shared[threadIdx.x] = __ldg(&T02[threadIdx.x]);
//	shared[256+threadIdx.x] = __ldg(&T12[threadIdx.x]);
//	shared[512+threadIdx.x] = __ldg(&T22[threadIdx.x]);
//	shared[768+threadIdx.x] = __ldg(&T32[threadIdx.x]);
//	shared[1024+threadIdx.x] = __ldg(&T42[threadIdx.x]);

//	shared[1280+threadIdx.x] = T52[threadIdx.x];
//	shared[1536+threadIdx.x] = T62[threadIdx.x];
//	shared[1792+threadIdx.x] = T72[threadIdx.x];

	uint64_t* inout = &g_hash[thread<<3];

	__threadfence_block();
	
	*(uint2x4*)&s[ 0] = __ldg4((uint2x4 *)&inout[ 0]);
	*(uint2x4*)&s[ 4] = __ldg4((uint2x4 *)&inout[ 4]);
	s[8] = make_uint2(1,0x80000000);

	keccak_kernel(s);
	streebog_kernel(shared,s);
	
	*(uint2x4*)&inout[ 0] = *(uint2x4*)&s[ 0];
	*(uint2x4*)&inout[ 4] = *(uint2x4*)&s[ 4];
}

__host__
void keccak_streebog_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);
	
	keccak_streebog_gpu_hash_64<<<grid, block>>>(threads,(uint64_t*)d_hash);
}
