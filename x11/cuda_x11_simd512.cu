/***************************************************************************************************
 * SIMD512 SM3+ CUDA IMPLEMENTATION (require cuda_x11_simd512_func.cuh)
 */

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#define TPB50_1 128
#define TPB50_2 128
#define TPB52_1 128
#define TPB52_2 128

static uint4 *d_temp4[MAX_GPUS];
#include "cuda_x11_simd512_func.cuh"

__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52_2,1)
#else
__launch_bounds__(TPB50_2,4)
#endif
static void x11_simd512_gpu_compress_64_maxwell(uint32_t threads, uint32_t *g_hash,const uint4 *const __restrict__ g_fft4)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)
	uint32_t IV[32];
	if (thread < threads){

		uint32_t *Hash = &g_hash[thread<<4];
//		Compression1(Hash, thread, g_fft4, g_state);
		uint32_t A[32];

		*(uint2x4*)&IV[ 0] = *(uint2x4*)&c_IV_512[ 0];
		*(uint2x4*)&IV[ 8] = *(uint2x4*)&c_IV_512[ 8];
		*(uint2x4*)&IV[16] = *(uint2x4*)&c_IV_512[16];
		*(uint2x4*)&IV[24] = *(uint2x4*)&c_IV_512[24];

		*(uint2x4*)&A[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&A[ 8] = __ldg4((uint2x4*)&Hash[ 8]);

		#pragma unroll 16
		for(uint32_t i=0;i<16;i++)
			A[ i] = A[ i] ^ IV[ i];

		#pragma unroll 16
		for(uint32_t i=16;i<32;i++)
			A[ i] = IV[ i];

		Round8(A, thr_offset, g_fft4);
		
		STEP8_IF(&IV[ 0],32, 4,13,&A[ 0],&A[ 8],&A[16],&A[24]);
		STEP8_IF(&IV[ 8],33,13,10,&A[24],&A[ 0],&A[ 8],&A[16]);
		STEP8_IF(&IV[16],34,10,25,&A[16],&A[24],&A[ 0],&A[ 8]);
		STEP8_IF(&IV[24],35,25, 4,&A[ 8],&A[16],&A[24],&A[ 0]);

		#pragma unroll 32
		for(uint32_t i=0;i<32;i++){
			IV[ i] = A[ i];
		}
		
		A[ 0] ^= 512;

		Round8_0_final(A, 3,23,17,27);
		Round8_1_final(A,28,19,22, 7);
		Round8_2_final(A,29, 9,15, 5);
		Round8_3_final(A, 4,13,10,25);
		STEP8_IF(&IV[ 0],32, 4,13, &A[ 0], &A[ 8], &A[16], &A[24]);
		STEP8_IF(&IV[ 8],33,13,10, &A[24], &A[ 0], &A[ 8], &A[16]);
		STEP8_IF(&IV[16],34,10,25, &A[16], &A[24], &A[ 0], &A[ 8]);
		STEP8_IF(&IV[24],35,25, 4, &A[ 8], &A[16], &A[24], &A[ 0]);

		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&A[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&A[ 8];
	}
}

__host__
void x11_simd512_cpu_init(int thr_id, uint32_t threads){
	cudaMalloc(&d_temp4[thr_id], 64*sizeof(uint4)*threads);
}

__host__
void x11_simd512_cpu_free(int thr_id){
	cudaFree(d_temp4[thr_id]);
}

__host__
void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

	int dev_id = device_map[thr_id];

	uint32_t tpb = TPB52_1;
	if (device_sm[dev_id] <= 500) tpb = TPB50_1;
	const dim3 grid1((8*threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	tpb = TPB52_2;
	if (device_sm[dev_id] <= 500) tpb = TPB50_2;
	const dim3 grid2((threads + tpb - 1) / tpb);
	const dim3 block2(tpb);
	
	x11_simd512_gpu_expand_64 <<<grid1, block1>>> (threads, d_hash, d_temp4[thr_id]);
	x11_simd512_gpu_compress_64_maxwell <<< grid2, block2 >>> (threads, d_hash, d_temp4[thr_id]);
}
