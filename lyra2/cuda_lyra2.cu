/**
* Lyra2 (v1) cuda implementation based on djm34 work - SM 5/5.2
* Improved by Nanashi Meiyo-Meijin - 2016
* tpruvot@github 2015

* Further improved on 970 by utilizing for loops in order to 
* reduce the code size.
* Provos Alexis - 2016
*
*/

#include <stdio.h>
#include <memory.h>
#include "cuda_vectors.h"
#include "cuda_helper.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 520
#endif

#define TPB52 32
#define TPB50 16

#define Nrow 8
#define Ncol 8
#define memshift 3

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

//#include "cuda_lyra2_sm2.cuh"
#include "cuda_lyra2_sm5.cuh"

#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ > 500

#include "cuda_vectors.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
__device__ uint32_t __shfl(uint32_t a, uint32_t b, uint32_t c);
#endif

__device__ uint2 *DMatrix;

__device__
void LD4S(uint2 res[3], const uint32_t row, const uint32_t col, const uint2* shared_mem)
{
	const uint32_t s0 = (Ncol * row + col) * memshift;

	#pragma unroll 3
	for (uint32_t j = 0; j < 3; j++)
		res[j] = shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
}

__device__
void ST4S(const uint32_t row, const uint32_t col, const uint2 data[3], uint2* shared_mem)
{
	const uint32_t s0 = (Ncol * row + col) * memshift;

	#pragma unroll 3
	for (uint32_t j = 0; j < 3; j++)
		shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];
}

__device__ __forceinline__ uint2 __shfl(uint2 a, uint32_t b, uint32_t c)
{
	return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
}

__device__ __forceinline__ void __shfl3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	a1 = __shfl(a1, b1, c);
	a2 = __shfl(a2, b2, c);
	a3 = __shfl(a3, b3, c);
}

static __device__
void Gfunc(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	a+= b;
	d = SWAPUINT2(d^a);
	c+= d;
	b = ROR24(b^c);
	a+= b;
	d = ROR16(d^a);
	c+= d;
	b = ROR2(b^c, 63);
}

static __device__
void round_lyra(uint2 s[4]){
	Gfunc(s[0], s[1], s[2], s[3]);
	__shfl3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	Gfunc(s[0], s[1], s[2], s[3]);
	__shfl3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

static __device__
void round_lyra(uint2x4 *const __restrict__ s)
{
	Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
}

static __device__
void reduceDuplex(uint2 state[4], uint2* shared)
{
	uint2 state1[3];


	#pragma unroll 8
	for (int i = 0; i < Nrow; i++)
	{
		ST4S(0, Ncol - i - 1, state, shared);

		round_lyra(state);
	}

	#pragma unroll 8
	for (int i = 0; i < Nrow; i++)
	{
		LD4S(state1, 0, i, shared);
		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(1, Ncol - i - 1, state1, shared);
	}
}

static __device__
void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint2* shared)
{
	uint2 state1[3], state2[3];

	#pragma unroll 1
	for (uint32_t i = 0; i < Nrow; i++)
	{
		LD4S(state1, rowIn, i, shared);
		LD4S(state2, rowInOut, i, shared);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, Ncol - i - 1, state1, shared);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		__shfl3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, shared);
	}
}

static __device__
void reduceDuplexRowt(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint2* shared)
{
	for (uint32_t i = 0; i < Nrow; i++)
	{
		uint2 state1[3], state2[3];

		LD4S(state1, rowIn, i, shared);
		LD4S(state2, rowInOut, i, shared);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		__shfl3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, shared);

		LD4S(state1, rowOut, i, shared);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, i, state1, shared);
	}
}

static __device__
void reduceDuplexRowt_8(const int rowInOut, uint2* state, uint2* shared)
{

	uint2 state1[3], state2[3], last[3];

	LD4S(state1, 2, 0, shared);
	LD4S(last, rowInOut, 0, shared);

	#pragma unroll 3
	for (int j = 0; j < 3; j++)
		state[j] ^= state1[j] + last[j];

	round_lyra(state);

	//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	__shfl3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

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

	if (rowInOut == 5)
	{
		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	#pragma unroll 1
	for (uint32_t i = 1; i < Nrow; i++)
	{
		LD4S(state1, 2, i, shared);
		LD4S(state2, rowInOut, i, shared);

		#pragma unroll 3
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);
	}


	#pragma unroll 3
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

__global__ __launch_bounds__(512, 1)
void lyra2_gpu_hash_32_1(uint32_t threads,const uint2* __restrict__ g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint2x4 state[4];

		state[0].x = state[1].x = __ldg(&g_hash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&g_hash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&g_hash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&g_hash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (uint32_t i = 0; i<24; i++)
			round_lyra(state); //because 12 is not enough

		((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
		((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
		((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
		((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
	}
}

__global__ __launch_bounds__(TPB52, 1)
void lyra2_gpu_hash_32_2(uint32_t threads)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	__shared__ uint2 shared[192*TPB52];

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = __ldg(&DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x]);
		state[1] = __ldg(&DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x]);
		state[2] = __ldg(&DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x]);
		state[3] = __ldg(&DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x]);

		reduceDuplex(state,shared);

		reduceDuplexRowSetup(1, 0, 2, state,shared);
		reduceDuplexRowSetup(2, 1, 3, state,shared);

		for(int i=3;i<7;i++){
			reduceDuplexRowSetup(i, 8%(i+1), i+1, state,shared);
		}

		uint32_t rowa = __shfl(state[0].x, 0, 4) & 7;
		uint32_t prev = 7;
		for(int i=0;i<21;i+=3){
			reduceDuplexRowt(prev, rowa, i&7, state,shared);
			prev = i&7;
			rowa = __shfl(state[0].x, 0, 4) & 7;
		}
		reduceDuplexRowt_8(rowa, state,shared);

		DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x] = state[0];
		DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x] = state[1];
		DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x] = state[2];
		DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x] = state[3];
	}
}

__global__ __launch_bounds__(512, 1)
void lyra2_gpu_hash_32_3(uint32_t threads, uint2 *g_hash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint2x4 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);

		for (uint32_t i = 0; i < 12; i++)
			round_lyra(state);

		g_hash[thread + threads * 0] = state[0].x;
		g_hash[thread + threads * 1] = state[0].y;
		g_hash[thread + threads * 2] = state[0].z;
		g_hash[thread + threads * 3] = state[0].w;

	} //thread
}
#else
#if __CUDA_ARCH__ < 500

/* for unsupported SM arch */
__device__ void* DMatrix;
#endif
__global__ void lyra2_gpu_hash_32_1(uint32_t threads, const uint2* __restrict__ g_hash) {}
__global__ void lyra2_gpu_hash_32_2(uint32_t threads) {}
__global__ void lyra2_gpu_hash_32_3(uint32_t threads, uint2 *g_hash) {}
#endif

__host__
void lyra2_cpu_init(int thr_id, uint32_t threads, uint2* d_matrix)
{
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint2* d_hash)
{
	int dev_id = device_map[thr_id];

	if (device_sm[dev_id] >= 520)
	{
		dim3 grid1((threads * 4 + TPB52 - 1) / TPB52);
		dim3 block1(4, TPB52 >> 2);

		dim3 grid2((threads + 512 - 1) / 64);
		dim3 block2(512);

		lyra2_gpu_hash_32_1 << <grid2, block2 >> > (threads, d_hash);

		lyra2_gpu_hash_32_2 << <grid1, block1>> > (threads);

		lyra2_gpu_hash_32_3 << <grid2, block2 >> > (threads, d_hash);
	}
	else{
		dim3 grid((threads + TPB50 - 1) / TPB50);
		dim3 block(TPB50);

		lyra2_gpu_hash_32 <<< grid, block >>> (threads, d_hash);
	}
}
