/*
	Based upon Tanguy Pruvot's repo
	Provos Alexis - 2016
*/

#include "cuda_helper.h"
#include "miner.h"
#include "cuda_vectors.h"

//#define SHUFFLE //<-- 2-way implementation of cubehash kernel, seems to be slightly better on certain gpus (or can be done), the rest will throttle

#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }
//#define SWAP(a,b) { a ^= b; b ^= a; a ^= b; }

#if defined(SHUFFLE)

#define TPB 1024

__device__ __forceinline__
void rrounds(uint32_t *x){
	#pragma unroll 4
	for (int r = 0; r < 16; ++r) {
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[ 8] += x[ 0]; x[ 9] += x[ 1];
		x[10] += x[ 2]; x[11] += x[ 3];
		x[12] += x[ 4]; x[13] += x[ 5];
		x[14] += x[ 6]; x[15] += x[ 7];
		x[ 0] = ROTL32(x[ 0], 7);
		x[ 1] = ROTL32(x[ 1], 7);
		x[ 2] = ROTL32(x[ 2], 7);
		x[ 3] = ROTL32(x[ 3], 7);
		x[ 4] = ROTL32(x[ 4], 7);
		x[ 5] = ROTL32(x[ 5], 7);
		x[ 6] = ROTL32(x[ 6], 7);
		x[ 7] = ROTL32(x[ 7], 7);
		/* "swap x_00klm with x_01klm" */
		SWAP(x[ 0], x[ 4]); SWAP(x[ 1], x[ 5]);
		SWAP(x[ 2], x[ 6]); SWAP(x[ 3], x[ 7]);
		
		x[ 0] ^= x[ 8]; x[ 4] ^= x[12];
		x[ 1] ^= x[ 9]; x[ 5] ^= x[13];
		x[ 2] ^= x[10]; x[ 6] ^= x[14];
		x[ 3] ^= x[11]; x[ 7] ^= x[15];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[ 8],x[10]); SWAP(x[ 9],x[11]);
		SWAP(x[12],x[14]); SWAP(x[13],x[15]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[ 8] += x[ 0]; x[ 9] += x[ 1];
		x[10] += x[ 2]; x[11] += x[ 3];
		x[12] += x[ 4]; x[13] += x[ 5];
		x[14] += x[ 6]; x[15] += x[ 7];
		x[ 0] = ROTL32(x[ 0],11);
		x[ 1] = ROTL32(x[ 1],11);
		x[ 2] = ROTL32(x[ 2],11);
		x[ 3] = ROTL32(x[ 3],11);
		x[ 4] = ROTL32(x[ 4],11);
		x[ 5] = ROTL32(x[ 5],11);
		x[ 6] = ROTL32(x[ 6],11);
		x[ 7] = ROTL32(x[ 7],11);
		/* "swap x_0j0lm with x_0j1lm" */
		x[ 0] = __shfl(x[ 0],threadIdx.x^1);
		x[ 1] = __shfl(x[ 1],threadIdx.x^1);
		x[ 2] = __shfl(x[ 2],threadIdx.x^1);
		x[ 3] = __shfl(x[ 3],threadIdx.x^1);
		x[ 4] = __shfl(x[ 4],threadIdx.x^1);
		x[ 5] = __shfl(x[ 5],threadIdx.x^1);
		x[ 6] = __shfl(x[ 6],threadIdx.x^1);
		x[ 7] = __shfl(x[ 7],threadIdx.x^1);
	
		x[ 0] ^= x[ 8]; x[ 1] ^= x[ 9];
		x[ 2] ^= x[10]; x[ 3] ^= x[11];
		x[ 4] ^= x[12]; x[ 5] ^= x[13];
		x[ 6] ^= x[14]; x[ 7] ^= x[15];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[ 8],x[ 9]); SWAP(x[10],x[11]);
		SWAP(x[12],x[13]); SWAP(x[14],x[15]);
	}
}
__global__ __launch_bounds__(TPB, 1)
void cubehash256_gpu_hash_32(uint32_t threads, uint2* g_hash){
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x)>>1;

	const uint32_t even = (threadIdx.x & 1);

	if (thread < threads){
		uint2 Hash[ 2];

		Hash[ 0] = __ldg(&g_hash[thread + (2*even+0) * threads]); //((2*lid)+1)
		Hash[ 1] = __ldg(&g_hash[thread + (2*even+1) * threads]);

		uint32_t x[16];
		if(even==0){
			x[ 0] = 0xEA2BD4B4;	x[ 1] = 0xCCD6F29F;	x[ 2] = 0x63117E71;	x[ 3] = 0x35481EAE;
			x[ 4] = 0xC2D0B696;	x[ 5] = 0x42AF2070;	x[ 6] = 0xD0720C35;	x[ 7] = 0x3361DA8C;
			x[ 8] = 0xD89041C3;	x[ 9] = 0x6107FBD5;	x[10] = 0x6C859D41;	x[11] = 0xF0B26679;
			x[12] = 0x2AF2B5AE;	x[13] = 0x9E4B4E60;	x[14] = 0x774ABFDD;	x[15] = 0x85254725;
		}else{
			x[ 0] = 0x22512D5B;	x[ 1] = 0xE5D94E63;	x[ 2] = 0x7E624131;	x[ 3] = 0xF4CC12BE;
			x[ 4] = 0x28CCECA4;	x[ 5] = 0x8EF8AD83;	x[ 6] = 0x4680AC00;	x[ 7] = 0x40E5FBAB;
			x[ 8] = 0x09392549;	x[ 9] = 0x5FA25603;	x[10] = 0x65C892FD;	x[11] = 0x93CB6285;
			x[12] = 0x15815AEB;	x[13] = 0x4AB6AAD6; 	x[14] = 0x9CDAF8AF;	x[15] = 0xD6032C0A;
		}
		x[ 0] ^= Hash[ 0].x; x[ 1] ^= Hash[ 0].y;
		x[ 2] ^= Hash[ 1].x; x[ 3] ^= Hash[ 1].y;

		rrounds(x);

		if(!even)
			x[ 0] ^= 0x80U;

		rrounds(x);
		/* "the integer 1 is xored into the last state word x_11111" */
		if(even)
			x[15] ^= 1U;
	
		#pragma unroll 10
		for (int i = 0; i < 10; ++i)
			rrounds(x);
	
		g_hash[thread + (2*even+0) * threads]	= *(uint2*)&x[ 0];
		g_hash[thread + (2*even+1) * threads]	= *(uint2*)&x[ 2];
	}
}
__host__
void cubehash256_cpu_hash_32(const uint32_t threads, uint2* d_hash){
	dim3 grid(((threads * 2 + TPB-1)/TPB));
	dim3 block(TPB);
	cubehash256_gpu_hash_32 <<<grid, block>>> (threads, d_hash);
}

#else

#define TPB 768

__device__ __forceinline__
void rrounds(uint32_t *x){
	#pragma unroll 2
	for (int r = 0; r < 16; r++) {
		/* "add x_0jklm into x_1jklmn modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0], 7);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1], 7);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2], 7);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3], 7);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4], 7);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5], 7);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6], 7);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7], 7);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8], 7);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9], 7);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10], 7);x[27] = x[27] + x[11];x[11] = ROTL32(x[11], 7);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12], 7);x[29] = x[29] + x[13];x[13] = ROTL32(x[13], 7);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14], 7);x[31] = x[31] + x[15];x[15] = ROTL32(x[15], 7);
		/* "swap x_00klm with x_01klm" */
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[ 0], x[ 8]);SWAP(x[ 1], x[ 9]);x[ 0] ^= x[16];x[ 8] ^= x[24];x[ 1] ^= x[17];x[ 9] ^= x[25];
		SWAP(x[ 2], x[10]);SWAP(x[ 3], x[11]);x[ 2] ^= x[18];x[10] ^= x[26];x[ 3] ^= x[19];x[11] ^= x[27];
		SWAP(x[ 4], x[12]);SWAP(x[ 5], x[13]);x[ 4] ^= x[20];x[12] ^= x[28];x[ 5] ^= x[21];x[13] ^= x[29];
		SWAP(x[ 6], x[14]);SWAP(x[ 7], x[15]);x[ 6] ^= x[22];x[14] ^= x[30];x[ 7] ^= x[23];x[15] ^= x[31];
		SWAP(x[16], x[18]);
		SWAP(x[17], x[19]);
		SWAP(x[24], x[26]);
		SWAP(x[25], x[27]);
		SWAP(x[20], x[22]);
		SWAP(x[21], x[23]);
		SWAP(x[28], x[30]);
		SWAP(x[29], x[31]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0],11);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1],11);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2],11);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3],11);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4],11);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5],11);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6],11);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7],11);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8],11);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9],11);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10],11);x[27] = x[27] + x[11];x[11] = ROTL32(x[11],11);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12],11);x[29] = x[29] + x[13];x[13] = ROTL32(x[13],11);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14],11);x[31] = x[31] + x[15];x[15] = ROTL32(x[15],11);
		/* "swap x_0j0lm with x_0j1lm" */
		SWAP(x[ 0], x[ 4]);SWAP(x[ 1], x[ 5]); x[ 0] ^= x[16]; x[ 4] ^= x[20]; x[ 1] ^= x[17]; x[ 5] ^= x[21];
		SWAP(x[ 2], x[ 6]);SWAP(x[ 3], x[ 7]); x[ 2] ^= x[18]; x[ 6] ^= x[22]; x[ 3] ^= x[19]; x[ 7] ^= x[23];
		SWAP(x[ 8], x[12]);SWAP(x[ 9], x[13]); x[ 8] ^= x[24]; x[12] ^= x[28]; x[ 9] ^= x[25]; x[13] ^= x[29];
		SWAP(x[10], x[14]);SWAP(x[11], x[15]); x[10] ^= x[26]; x[14] ^= x[30]; x[11] ^= x[27]; x[15] ^= x[31];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[16], x[17]);
		SWAP(x[18], x[19]); SWAP(x[20], x[21]); SWAP(x[22], x[23]);SWAP(x[24], x[25]); SWAP(x[26], x[27]); SWAP(x[28], x[29]); SWAP(x[30], x[31]);
	}
}
__global__ __launch_bounds__(TPB,1)
void cubehash256_gpu_hash_32(const uint32_t threads, uint2* g_hash){
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	
	if (thread >= threads)return;
	
	uint2 Hash[ 4];

	Hash[ 0] = __ldg(&g_hash[thread]);
	Hash[ 1] = __ldg(&g_hash[thread + 1 * threads]);
	Hash[ 2] = __ldg(&g_hash[thread + 2 * threads]);
	Hash[ 3] = __ldg(&g_hash[thread + 3 * threads]);

	uint32_t x[32] = {
		0xEA2BD4B4, 0xCCD6F29F, 0x63117E71, 0x35481EAE,	0x22512D5B, 0xE5D94E63, 0x7E624131, 0xF4CC12BE,
		0xC2D0B696, 0x42AF2070, 0xD0720C35, 0x3361DA8C,	0x28CCECA4, 0x8EF8AD83, 0x4680AC00, 0x40E5FBAB,
		0xD89041C3, 0x6107FBD5, 0x6C859D41, 0xF0B26679,	0x09392549, 0x5FA25603, 0x65C892FD, 0x93CB6285,
		0x2AF2B5AE, 0x9E4B4E60, 0x774ABFDD, 0x85254725,	0x15815AEB, 0x4AB6AAD6, 0x9CDAF8AF, 0xD6032C0A
	};

	*(uint2x4*)&x[ 0] ^= *(uint2x4*)&Hash[ 0];

	rrounds(x);
	x[ 0] ^= 0x80U;
	rrounds(x);
	
	/* "the integer 1 is xored into the last state word x_11111" */
	x[31] ^= 1U;

	#pragma unroll
	for (int i = 0; i < 10; ++i)rrounds(x);

	g_hash[thread]			= make_uint2(x[ 0],x[ 1]);
	g_hash[1 * threads + thread]	= make_uint2(x[ 2],x[ 3]);
	g_hash[2 * threads + thread]	= make_uint2(x[ 4],x[ 5]);
	g_hash[3 * threads + thread]	= make_uint2(x[ 6],x[ 7]);
}

__host__
void cubehash256_cpu_hash_32(const uint32_t threads, uint2* d_hash){
	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);
	cubehash256_gpu_hash_32 <<<grid, block>>> (threads, d_hash);
}

#endif
