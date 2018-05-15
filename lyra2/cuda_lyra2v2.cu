/*
*	Lyra2v2 kernel merged from nicehash repository
*	Based on djm34/VTC sources and incredible 2x boost by Nanashi Meiyo-Meijin (May 2016)
*/

#include <stdio.h>
#include <memory.h>
#include "cuda_vectors.h"
#include "miner.h"

#define TPB1 32
#define TPB2 128

#define Nrow 4
#define Ncol 4
#define u64type uint4
#define vectype uint48
#define memshift 3

__device__ __forceinline__ uint2 LD4S(const int index,const uint2* shared_mem)
{
	return shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void ST4S(const int index, const uint2 data, uint2* shared_mem)
{
	shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data;
}

__device__ __forceinline__
void Gfunc_v35(uint2 &a, uint2 &b, uint2 &c, uint2 &d){
	a += b;d = SWAPUINT2(d^a);
	c += d;b = ROR24(b^c);
	a += b;d = ROR16(d^a);
	c += d;b = ROR2(b^c, 63);
}

__device__ __forceinline__
void round_lyra_v35(uint2x4 s[4])
{
	Gfunc_v35(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc_v35(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc_v35(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc_v35(s[0].w, s[1].w, s[2].w, s[3].w);

	Gfunc_v35(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc_v35(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc_v35(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc_v35(s[0].w, s[1].x, s[2].y, s[3].z);
}

__device__ __forceinline__ uint2 __shfl(uint2 a, uint32_t b, uint32_t c)
{
	return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
}

__device__ __forceinline__
void round_lyra_v35(uint2 s[4])
{
	Gfunc_v35(s[0], s[1], s[2], s[3]);
	s[1] = __shfl(s[1], threadIdx.x + 1, 4);
	s[2] = __shfl(s[2], threadIdx.x + 2, 4);
	s[3] = __shfl(s[3], threadIdx.x + 3, 4);
	Gfunc_v35(s[0], s[1], s[2], s[3]);
	s[1] = __shfl(s[1], threadIdx.x + 3, 4);
	s[2] = __shfl(s[2], threadIdx.x + 2, 4);
	s[3] = __shfl(s[3], threadIdx.x + 1, 4);
}

__device__ __forceinline__
void reduceDuplexRowSetupV2(uint2 state[4],uint2* shared){

	uint2 state1[Ncol][3], state0[Ncol][3], state2[3];

	#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			state0[Ncol - i - 1][j] = state[j];
		round_lyra_v35(state);
	}

	#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state0[i][j];

		round_lyra_v35(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[Ncol - i - 1][j] = state0[i][j];

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[Ncol - i - 1][j] ^= state[j];
	}

	for (int i = 0; i < Ncol; i++){
		const uint32_t s0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s2 = memshift * Ncol * 2 + memshift * (Ncol - 1) - i*memshift;
		
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[i][j] + state0[i][j];

		round_lyra_v35(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = state1[i][j];

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] ^= state[j];

		#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s2 + j, state2[j],shared);

		//êÂèOÌXbh©çf[^ðá€(¯ÉêÂæÌXbhÉf[^ðé)
		uint2 Data0 = __shfl(state[0], threadIdx.x - 1, 4);
		uint2 Data1 = __shfl(state[1], threadIdx.x - 1, 4);
		uint2 Data2 = __shfl(state[2], threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state0[i][0] ^= Data2;
			state0[i][1] ^= Data0;
			state0[i][2] ^= Data1;
		}
		else
		{
			state0[i][0] ^= Data0;
			state0[i][1] ^= Data1;
			state0[i][2] ^= Data2;
		}

		#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s0 + j, state0[i][j],shared);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state0[i][j] = state2[j];

	}
	
//	#pragma nounroll
	for (int i = 0; i < Ncol; i++)
	{
		const uint32_t s1 = memshift * Ncol * 1 + i*memshift;
		const uint32_t s3 = memshift * Ncol * 3 + memshift * (Ncol - 1) - i*memshift;

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[i][j] + state0[Ncol - i - 1][j];

		round_lyra_v35(state);

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			state0[Ncol - i - 1][j] ^= state[j];

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			ST4S(s3 + j, state0[Ncol - i - 1][j],shared);

		//êÂèOÌXbh©çf[^ðá€(¯ÉêÂæÌXbhÉf[^ðé)
		uint2 Data0 = __shfl(state[0], threadIdx.x - 1, 4);
		uint2 Data1 = __shfl(state[1], threadIdx.x - 1, 4);
		uint2 Data2 = __shfl(state[2], threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state1[i][0] ^= Data2;
			state1[i][1] ^= Data0;
			state1[i][2] ^= Data1;
		}
		else
		{
			state1[i][0] ^= Data0;
			state1[i][1] ^= Data1;
			state1[i][2] ^= Data2;
		}

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			ST4S(s1 + j, state1[i][j],shared);
	}
}

__device__
void reduceDuplexRowtV2(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4],uint2 *shared)
{
	uint2 state1[3];
	uint32_t ps1 = memshift * Ncol * rowIn;
	uint32_t ps2 = memshift * Ncol * rowInOut;
	uint32_t ps3 = memshift * Ncol * rowOut;

	#pragma unroll 1
	for (int i = 0; i < Ncol; i++){
		uint32_t s1 = ps1 + i*memshift;
		uint32_t s2 = ps2 + i*memshift;
		uint32_t s3 = ps3 + i*memshift;

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			state1[j] = LD4S(s2 + j,shared);

		#pragma unroll 3
		for (int j = 0; j < 3; j++){
			state[j] = state[j] ^ (state1[j] + LD4S(s1 + j,shared));
		}
		round_lyra_v35(state);

		//�����O�̃X���b�h�����f�[�^���Ⴄ(�����Ɉ����̃X���b�h�Ƀf�[�^�𑗂�)
		uint2 Data0 = __shfl(state[0], threadIdx.x - 1, 4);
		uint2 Data1 = __shfl(state[1], threadIdx.x - 1, 4);
		uint2 Data2 = __shfl(state[2], threadIdx.x - 1, 4);
		if (threadIdx.x != 0)
		{
			state1[0] ^= Data0;
			state1[1] ^= Data1;
			state1[2] ^= Data2;
		}
		else
		{
			state1[0] ^= Data2;
			state1[1] ^= Data0;
			state1[2] ^= Data1;
		}

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			ST4S(s2 + j, state1[j],shared);

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			state1[j] = LD4S(s3 + j, shared);
		
		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			ST4S(s3 + j, state1[j] ^ state[j],shared);
	}
}

__device__
void reduceDuplexRowtV2_4(const int rowInOut, uint2 state[4], uint2* shared)
{
	int i, j;
	uint2 last[3],last2[3];
	const uint32_t ps1 = memshift * Ncol * 2;
	const uint32_t ps2 = memshift * Ncol * rowInOut;
	
	#pragma unroll
	for (int j = 0; j < 3; j++)
		last[j] = LD4S(ps2 + j,shared);

	#pragma unroll 
	for (int j = 0; j < 3; j++)
		state[j] ^= LD4S(ps1 + j,shared) + last[j];

	round_lyra_v35(state);

	//�����O�̃X���b�h�����f�[�^���Ⴄ(�����Ɉ����̃X���b�h�Ƀf�[�^�𑗂�)
	uint2 Data0 = __shfl(state[0], threadIdx.x - 1, 4);
	uint2 Data1 = __shfl(state[1], threadIdx.x - 1, 4);
	uint2 Data2 = __shfl(state[2], threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	}
	else
	{
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == 3)
	{
		#pragma unroll 
		for (j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	#pragma unroll
	for (i = 1; i < Ncol; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;

		#pragma unroll
		for (j = 0; j < 3; j++)
			last2[j] = LD4S(s2 + j,shared);
			
		#pragma unroll 
		for (j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j,shared) + last2[j];

		round_lyra_v35(state);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

__constant__ const uint2x4 blake2b_IV[2] = {
	0xf3bcc908lu, 0x6a09e667lu,
	0x84caa73blu, 0xbb67ae85lu,
	0xfe94f82blu, 0x3c6ef372lu,
	0x5f1d36f1lu, 0xa54ff53alu,
	0xade682d1lu, 0x510e527flu,
	0x2b3e6c1flu, 0x9b05688clu,
	0xfb41bd6blu, 0x1f83d9ablu,
	0x137e2179lu, 0x5be0cd19lu
};

__constant__ const uint2x4 Mask[2] = {
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000001lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000080lu, 0x00000000lu,
	0x00000000lu, 0x01000000lu
};

__global__
__launch_bounds__(TPB2)
void lyra2v2_gpu_hash_32_1(uint32_t threads,uint2* DMatrix,const uint2* __restrict__ outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint2x4 state[4];

	uint2x4 *DState = (uint2x4*)DMatrix;

	if (thread < threads)
	{
		state[0].x = state[1].x = __ldg(&outputHash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&outputHash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&outputHash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&outputHash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		#pragma unroll 1 
		for (int i = 0; i<12; i++)
			round_lyra_v35(state);

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);

		DState[blockDim.x * gridDim.x * 0 + thread] = state[0];
		DState[blockDim.x * gridDim.x * 1 + thread] = state[1];
		DState[blockDim.x * gridDim.x * 2 + thread] = state[2];
		DState[blockDim.x * gridDim.x * 3 + thread] = state[3];

	} //thread
}

__global__
__launch_bounds__(TPB1)
void lyra2v2_gpu_hash_32_2(uint32_t threads,uint2* DState)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	__shared__ uint2 shared[48 * TPB1];

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = ((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[1] = ((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[2] = ((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[3] = ((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];

		reduceDuplexRowSetupV2(state,shared);

		uint32_t rowa;
		int prev = 3;

		#pragma unroll 2
		for (int i = 0; i < 3; i++)
		{
			rowa = __shfl(state[0].x, 0, 4) & 3;
			reduceDuplexRowtV2(prev, rowa, i, state,shared);
			prev = i;
		}

		rowa = __shfl(state[0].x, 0, 4) & 3;
		reduceDuplexRowtV2_4(rowa, state,shared);

		((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[0];
		((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[1];
		((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[2];
		((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[3];
	} //thread
}

__global__
__launch_bounds__(TPB2)
void lyra2v2_gpu_hash_32_3(uint32_t threads,const uint2* __restrict__ DMatrix,uint2 *outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint2x4 state[4];

	uint2x4 *DState = (uint2x4*)DMatrix;

	if (thread < threads)
	{
		state[0] = __ldg4(&DState[blockDim.x * gridDim.x * 0 + thread]);
		state[1] = __ldg4(&DState[blockDim.x * gridDim.x * 1 + thread]);
		state[2] = __ldg4(&DState[blockDim.x * gridDim.x * 2 + thread]);
		state[3] = __ldg4(&DState[blockDim.x * gridDim.x * 3 + thread]);

		#pragma unroll 1
		for (int i = 0; i < 12; i++)
			round_lyra_v35(state);

		outputHash[thread + threads * 0] = state[0].x;
		outputHash[thread + threads * 1] = state[0].y;
		outputHash[thread + threads * 2] = state[0].z;
		outputHash[thread + threads * 3] = state[0].w;

	} //thread
}

__host__
void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads,uint2* DMatrix, uint2 *d_outputHash)
{
	dim3 grid1((threads * 4 + TPB1 - 1) / TPB1);
	dim3 block1(4, TPB1>>2);

	dim3 grid2((threads + TPB2 - 1) / TPB2);
	dim3 block2(TPB2);

	lyra2v2_gpu_hash_32_1 <<<grid2, block2 >>>(threads,DMatrix,d_outputHash);

	lyra2v2_gpu_hash_32_2 <<<grid1, block1 >>>(threads,DMatrix);

	lyra2v2_gpu_hash_32_3 <<<grid2, block2 >>>(threads,DMatrix,d_outputHash);

}

