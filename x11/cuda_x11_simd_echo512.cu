/***************************************************************************************************
 * SIMD512 SM3+ CUDA IMPLEMENTATION (require cuda_x11_simd512_func.cuh)
 * Forked from Tanguy Pruvot's repo
 * Merged with echo512 kernel
 * Provos Alexis - 2016
 */

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "cuda_x11_aes.cuh"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#define TPB50_1 128
#define TPB50_2 128
#define TPB52_1 128
#define TPB52_2 128

//SIMD512 MACROS ---------------------------- 
static uint4 *d_temp4[MAX_GPUS];
#include "cuda_x11_simd512_func.cuh"
//END OF SIMD512 MACROS ---------------------

//ECHO MACROS--------------------------------
#define SHIFT_ROW1(a, b, c, d)   do { \
		tmp0 = W[a+0]; \
		W[a+0] = W[b+0]; \
		W[b+0] = W[c+0]; \
		W[c+0] = W[d+0]; \
		W[d+0] = tmp0; \
\
		tmp0 = W[a+1]; \
		W[a+1] = W[b+1]; \
		W[b+1] = W[c+1]; \
		W[c+1] = W[d+1]; \
		W[d+1] = tmp0; \
\
		tmp0 = W[a+2]; \
		W[a+2] = W[b+2]; \
		W[b+2] = W[c+2]; \
		W[c+2] = W[d+2]; \
		W[d+2] = tmp0; \
\
		tmp0 = W[a+3]; \
		W[a+3] = W[b+3]; \
		W[b+3] = W[c+3]; \
		W[c+3] = W[d+3]; \
		W[d+3] = tmp0; \
	} while (0)

#define SHIFT_ROW2(a, b, c, d)   do { \
		tmp0 = W[a+0]; \
		W[a+0] = W[c+0]; \
		W[c+0] = tmp0; \
\
		tmp0 = W[a+1]; \
		W[a+1] = W[c+1]; \
		W[c+1] = tmp0; \
\
		tmp0 = W[a+2]; \
		W[a+2] = W[c+2]; \
		W[c+2] = tmp0; \
\
		tmp0 = W[a+3]; \
		W[a+3] = W[c+3]; \
		W[c+3] = tmp0; \
\
		tmp0 = W[b+0]; \
		W[b+0] = W[d+0]; \
		W[d+0] = tmp0; \
\
		tmp0 = W[b+1]; \
		W[b+1] = W[d+1]; \
		W[d+1] = tmp0; \
\
		tmp0 = W[b+2]; \
		W[b+2] = W[d+2]; \
		W[d+2] = tmp0; \
\
		tmp0 = W[b+3]; \
		W[b+3] = W[d+3]; \
		W[d+3] = tmp0; \
	} while (0)

#define MIX_COLUMN1(ia, ib, ic, id, n)   do { \
		tmp0 = W[ia+n]; \
		unsigned int tmp1 = W[ic+n]; \
		unsigned int tmp2 = tmp0 ^ W[ib+n]; \
		unsigned int tmp3 = W[ib+n] ^ tmp1; \
		unsigned int tmp4 = tmp1 ^ W[id+n]; \
		unsigned int tmp5 = (((tmp2 & (0x80808080)) >> 7) * 27 ^ ((tmp2 & (0x7F7F7F7F)) << 1));\
		unsigned int tmp6 = (((tmp3 & (0x80808080)) >> 7) * 27 ^ ((tmp3 & (0x7F7F7F7F)) << 1));\
		unsigned int tmp7 = (((tmp4 & (0x80808080)) >> 7) * 27 ^ ((tmp4 & (0x7F7F7F7F)) << 1));\
		W[ia+n] = tmp5 ^ tmp3 ^ W[id+n]; \
		W[ib+n] = tmp6 ^ tmp0 ^ tmp4; \
		W[ic+n] = tmp7 ^ tmp2 ^ W[id+n]; \
		W[id+n] = tmp5^tmp6^tmp7^tmp2^tmp1; \
	} while (0)

#define MIX_COLUMN(a, b, c, d)   do { \
		MIX_COLUMN1(a, b, c, d, 0); \
		MIX_COLUMN1(a, b, c, d, 1); \
		MIX_COLUMN1(a, b, c, d, 2); \
		MIX_COLUMN1(a, b, c, d, 3); \
	} while (0)
//END OF ECHO MACROS-------------------------

__device__ __forceinline__
static void SIMD_Compress(uint32_t *A,const uint32_t thr_offset,const uint4 *const __restrict__ g_fft4){

	uint32_t IV[32];

	*(uint2x4*)&IV[ 0] = *(uint2x4*)&c_IV_512[ 0];
	*(uint2x4*)&IV[ 8] = *(uint2x4*)&c_IV_512[ 8];
	*(uint2x4*)&IV[16] = *(uint2x4*)&c_IV_512[16];
	*(uint2x4*)&IV[24] = *(uint2x4*)&c_IV_512[24];
	
	Round8(A, thr_offset, g_fft4);
	
	const uint32_t a[4] = {4,13,10,25};
	
	for(int i=0;i<4;i++)
		STEP8_IF(&IV[i*8],32+i,a[i],a[(i+1)&3],&A[(0+i*24)&31],&A[(8+i*24)&31],&A[(16+i*24)&31],&A[(24+i*24)&31]);
		
	#pragma unroll 32
	for(uint32_t i=0;i<32;i++){
		IV[ i] = A[ i];
	}
		
	A[ 0] ^= 512;

	Round8_0_final(A, 3,23,17,27);
	Round8_1_final(A,28,19,22, 7);
	Round8_2_final(A,29, 9,15, 5);
	Round8_3_final(A, 4,13,10,25);

	for(int i=0;i<4;i++)
		STEP8_IF(&IV[i*8],32+i,a[i],a[(i+1)&3],&A[(0+i*24)&31],&A[(8+i*24)&31],&A[(16+i*24)&31],&A[(24+i*24)&31]);
		
}

__device__
static void echo_round(const uint32_t sharedMemory[4][256], uint32_t* W, uint32_t &k0){
	// Big Sub Words
	#pragma unroll 4
	for (int idx = 0; idx < 16; idx++){
		AES_2ROUND(sharedMemory,W[(idx<<2) + 0], W[(idx<<2) + 1], W[(idx<<2) + 2], W[(idx<<2) + 3], k0);
		idx++;
		AES_2ROUND_LDG(sharedMemory,W[(idx<<2) + 0], W[(idx<<2) + 1], W[(idx<<2) + 2], W[(idx<<2) + 3], k0);
		idx++;
		AES_2ROUND_LDG(sharedMemory,W[(idx<<2) + 0], W[(idx<<2) + 1], W[(idx<<2) + 2], W[(idx<<2) + 3], k0);
		idx++;
		AES_2ROUND(sharedMemory,W[(idx<<2) + 0], W[(idx<<2) + 1], W[(idx<<2) + 2], W[(idx<<2) + 3], k0);
	}
	uint32_t tmp0;
	SHIFT_ROW1(4, 20, 36, 52);
	SHIFT_ROW2(8, 24, 40, 56);
	SHIFT_ROW1(60,44, 28, 12);
	// Mix Columns
	#pragma unroll 4
	for (int i = 0; i < 4; i++){ // Schleife über je 2*uint32_t
		#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16){ // Schleife über die elemnte
			uint32_t a[4];
			a[0] = W[idx + i];
			a[1] = W[idx + i + 4];
			a[2] = W[idx + i + 8];
			a[3] = W[idx + i +12];

			uint32_t ab = a[0] ^ a[1];
			uint32_t bc = a[1] ^ a[2];
			uint32_t cd = a[2] ^ a[3];

			uint32_t t, t2, t3;
			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[idx + i] = bc ^ a[3] ^ abx;
			W[idx + i + 4] = a[0] ^ cd ^ bcx;
			W[idx + i + 8] = ab ^ a[3] ^ cdx;
			W[idx + i +12] = ab ^ a[2] ^ (abx ^ bcx ^ cdx);
		}
	}
}

__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52_2,5)
#else
__launch_bounds__(TPB50_2,5)
#endif
static void x11_simd512_gpu_compress_64_maxwell_echo512_final(const uint32_t* __restrict__ g_hash,const uint4 *const __restrict__ g_fft4,uint32_t* resNonce, const uint64_t target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)
	
	__shared__ uint32_t sharedMemory[4][256];
	
	aes_gpu_init128(sharedMemory);

	const uint32_t P[48] = {
		0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751,0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7,0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
		0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
	};
	uint32_t k0;
	uint32_t h[16];

//	if (thread < threads){

		const uint32_t* __restrict__ Hash = &g_hash[thread<<4];

		uint32_t A[32];

		*(uint2x4*)&A[ 0] = *(uint2x4*)&c_IV_512[ 0] ^ __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&A[ 8] = *(uint2x4*)&c_IV_512[ 8] ^ __ldg4((uint2x4*)&Hash[ 8]);
		*(uint2x4*)&A[16] = *(uint2x4*)&c_IV_512[16];
		*(uint2x4*)&A[24] = *(uint2x4*)&c_IV_512[24];

		SIMD_Compress(A, thr_offset, g_fft4);

		#pragma unroll 16
		for(int i=0;i<16;i++){
			h[i] = A[i];
		}
		
		uint64_t backup = *(uint64_t*)&h[ 6];

		k0 = 512 + 8;

		#pragma unroll 2
		for (uint32_t idx = 0; idx < 16; idx += 4){
			AES_2ROUND(sharedMemory,h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);
			idx+=4;
			AES_2ROUND_LDG(sharedMemory,h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);
		}
		k0 += 4;

		uint32_t W[64];

//		#pragma unroll 4
		for (int i = 0; i < 4; i++){
			uint32_t a = P[i];
			uint32_t b = P[i + 4];
			uint32_t c = h[i + 8];
			uint32_t d = P[i + 8];

			uint32_t ab = a ^ b;
			uint32_t bc = b ^ c;
			uint32_t cd = c ^ d;


			uint32_t t =  ((a ^ b) & 0x80808080);
			uint32_t t2 = ((b ^ c) & 0x80808080);
			uint32_t t3 = ((c ^ d) & 0x80808080);

			uint32_t abx = ((t  >> 7) * 27U) ^ ((ab^t) << 1);
			uint32_t bcx = ((t2 >> 7) * 27U) ^ ((bc^t2) << 1);
			uint32_t cdx = ((t3 >> 7) * 27U) ^ ((cd^t3) << 1);

			W[0U + i] = bc ^ d ^ abx;
			W[4U + i] = a ^ cd ^ bcx;
			W[8U + i] = ab ^ d ^ cdx;
			W[12U+ i] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[12U + i];
			b = h[i + 4U];
			c = P[12U + i + 4U];
			d = P[12U + i + 8U];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t  = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[16U + i] = abx ^ bc ^ d;
			W[16U + i + 4U] = bcx ^ a ^ cd;
			W[16U + i + 8U] = cdx ^ ab ^ d;
			W[16U + i +12U] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = h[i];
			b = P[24U + i + 0U];
			c = P[24U + i + 4U];
			d = P[24U + i + 8U];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[32U + i] = abx ^ bc ^ d;
			W[32U + i + 4U] = bcx ^ a ^ cd;
			W[32U + i + 8U] = cdx ^ ab ^ d;
			W[32U + i +12U] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[36U + i ];
			b = P[36U + i + 4U];
			c = P[36U + i + 8U];
			d = h[i + 12U];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;

			t  = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[48U + i] = abx ^ bc ^ d;
			W[48U + i + 4U] = bcx ^ a ^ cd;
			W[48U + i + 8U] = cdx ^ ab ^ d;
			W[48U + i +12U] = abx ^ bcx ^ cdx ^ ab ^ c;
		}
		
		for (int k = 1; k < 9; k++){
			echo_round(sharedMemory,W,k0);
		}
		
		// Big Sub Words
		uint32_t y[4];
		aes_round(sharedMemory, W[ 0], W[ 1], W[ 2], W[ 3], k0, y[0], y[1], y[2], y[3]);
		aes_round(sharedMemory, y[ 0], y[ 1], y[ 2], y[ 3], W[ 0], W[ 1], W[ 2], W[ 3]);
		aes_round(sharedMemory, W[ 4], W[ 5], W[ 6], W[ 7], k0, y[0], y[1], y[2], y[3]);
		aes_round(sharedMemory, y[ 0], y[ 1], y[ 2], y[ 3], W[ 4], W[ 5], W[ 6], W[ 7]);
		aes_round(sharedMemory, W[ 8], W[ 9], W[10], W[11], k0, y[0], y[1], y[2], y[3]);
		aes_round(sharedMemory, y[ 0], y[ 1], y[ 2], y[ 3], W[ 8], W[ 9], W[10], W[11]);
		aes_round(sharedMemory, W[20], W[21], W[22], W[23], k0, y[0], y[1], y[2], y[3]);
		aes_round(sharedMemory, y[ 0], y[ 1], y[ 2], y[ 3], W[20], W[21], W[22], W[23]);
		aes_round(sharedMemory, W[28], W[29], W[30], W[31], k0, y[0], y[1], y[2], y[3]);
		aes_round(sharedMemory, y[ 0], y[ 1], y[ 2], y[ 3], W[28], W[29], W[30], W[31]);
		aes_round(sharedMemory, W[32], W[33], W[34], W[35], k0, y[0], y[1], y[2], y[3]);
		aes_round(sharedMemory, y[ 0], y[ 1], y[ 2], y[ 3], W[32], W[33], W[34], W[35]);
		aes_round(sharedMemory, W[40], W[41], W[42], W[43], k0, y[0], y[1], y[2], y[3]);
		aes_round(sharedMemory, y[ 0], y[ 1], y[ 2], y[ 3], W[40], W[41], W[42], W[43]);
		aes_round(sharedMemory, W[52], W[53], W[54], W[55], k0, y[0], y[1], y[2], y[3]);
		aes_round(sharedMemory, y[ 0], y[ 1], y[ 2], y[ 3], W[52], W[53], W[54], W[55]);
		aes_round(sharedMemory, W[60], W[61], W[62], W[63], k0, y[0], y[1], y[2], y[3]);
		aes_round(sharedMemory, y[ 0], y[ 1], y[ 2], y[ 3], W[60], W[61], W[62], W[63]);
		
		uint32_t bc = W[22] ^ W[42];
		uint32_t t2 = (bc & 0x80808080);
		W[ 6] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

		bc = W[23] ^ W[43];
		t2 = (bc & 0x80808080);
		W[ 7] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

		bc = W[10] ^ W[54];
		t2 = (bc & 0x80808080);
		W[38] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

		bc = W[11] ^ W[55];
		t2 = (bc & 0x80808080);
		W[39] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

		uint64_t check = backup ^ *(uint64_t*)&W[2] ^ *(uint64_t*)&W[6] ^ *(uint64_t*)&W[10] ^ *(uint64_t*)&W[30] ^ *(uint64_t*)&W[34] ^ *(uint64_t*)&W[38] ^ *(uint64_t*)&W[42] ^ *(uint64_t*)&W[62];
		if(check <= target){
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}		
//	}
}

__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52_2,6)
#else
__launch_bounds__(TPB50_2,6)
#endif
static void x11_simd512_gpu_compress_64_maxwell_echo512(uint32_t *g_hash,const uint4 *const __restrict__ g_fft4)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)
	
	__shared__ uint32_t sharedMemory[4][256];
	
	aes_gpu_init128(sharedMemory);

	const uint32_t P[48] = {
		0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751,0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7,0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
		0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
	};
	uint32_t k0;
	uint32_t h[16];

//	if (thread < threads){

		uint32_t *Hash = &g_hash[thread<<4];

		uint32_t A[32];

		*(uint2x4*)&A[ 0] = *(uint2x4*)&c_IV_512[ 0] ^ __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&A[ 8] = *(uint2x4*)&c_IV_512[ 8] ^ __ldg4((uint2x4*)&Hash[ 8]);
		*(uint2x4*)&A[16] = *(uint2x4*)&c_IV_512[16];
		*(uint2x4*)&A[24] = *(uint2x4*)&c_IV_512[24];
		
		__syncthreads();

		SIMD_Compress(A, thr_offset, g_fft4);

		#pragma unroll 16
		for(int i=0;i<16;i++){
			h[i] = A[i];
		}
		
		k0 = 512 + 8;

		#pragma unroll 4
		for (uint32_t idx = 0; idx < 16; idx += 4)
			AES_2ROUND(sharedMemory,h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);

		k0 += 4;

		uint32_t W[64];

//		#pragma unroll 4
		for (int i = 0; i < 4; i++){
			uint32_t a = P[i];
			uint32_t b = P[i + 4];
			uint32_t c = h[i + 8];
			uint32_t d = P[i + 8];

			uint32_t ab = a ^ b;
			uint32_t bc = b ^ c;
			uint32_t cd = c ^ d;


			uint32_t t =  ((a ^ b) & 0x80808080);
			uint32_t t2 = ((b ^ c) & 0x80808080);
			uint32_t t3 = ((c ^ d) & 0x80808080);

			uint32_t abx = ((t  >> 7) * 27U) ^ ((ab^t) << 1);
			uint32_t bcx = ((t2 >> 7) * 27U) ^ ((bc^t2) << 1);
			uint32_t cdx = ((t3 >> 7) * 27U) ^ ((cd^t3) << 1);

			W[0U + i] = bc ^ d ^ abx;
			W[4U + i] = a ^ cd ^ bcx;
			W[8U + i] = ab ^ d ^ cdx;
			W[12U+ i] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[12U + i];
			b = h[i + 4U];
			c = P[12U + i + 4U];
			d = P[12U + i + 8U];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t  = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[16U + i] = abx ^ bc ^ d;
			W[16U + i + 4U] = bcx ^ a ^ cd;
			W[16U + i + 8U] = cdx ^ ab ^ d;
			W[16U + i +12U] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = h[i];
			b = P[24U + i + 0U];
			c = P[24U + i + 4U];
			d = P[24U + i + 8U];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[32U + i] = abx ^ bc ^ d;
			W[32U + i + 4U] = bcx ^ a ^ cd;
			W[32U + i + 8U] = cdx ^ ab ^ d;
			W[32U + i +12U] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[36U + i ];
			b = P[36U + i + 4U];
			c = P[36U + i + 8U];
			d = h[i + 12U];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;

			t  = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[48U + i] = abx ^ bc ^ d;
			W[48U + i + 4U] = bcx ^ a ^ cd;
			W[48U + i + 8U] = cdx ^ ab ^ d;
			W[48U + i +12U] = abx ^ bcx ^ cdx ^ ab ^ c;
		}
		
		for (int k = 1; k < 10; k++){
			echo_round(sharedMemory,W,k0);
		}
		#pragma unroll 4
		for (uint32_t i = 0; i < 16; i += 4)
		{
			W[i] ^= W[32 + i] ^ 512;
			W[i + 1] ^= W[32 + i + 1];
			W[i + 2] ^= W[32 + i + 2];
			W[i + 3] ^= W[32 + i + 3];
		}
		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&A[ 0] ^ *(uint2x4*)&W[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&A[ 8] ^ *(uint2x4*)&W[ 8];		
//	}
}

__host__
void x11_simd_echo_512_cpu_init(int thr_id, uint32_t threads){
	cudaMalloc(&d_temp4[thr_id], 64*sizeof(uint4)*threads);
}

__host__
void x11_simd_echo_512_cpu_free(int thr_id){
	cudaFree(d_temp4[thr_id]);
}

__host__
void x11_simd_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target){

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
	x11_simd512_gpu_compress_64_maxwell_echo512_final <<< grid2, block2 >>> (d_hash, d_temp4[thr_id],d_resNonce,target);
}

__host__
void x11_simd_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

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
	x11_simd512_gpu_compress_64_maxwell_echo512 <<< grid2, block2 >>> (d_hash, d_temp4[thr_id]);
}
