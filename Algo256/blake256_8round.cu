/**
 * Blake-256 Cuda Kernel (Tested on SM 5/5.2)
 * Tanguy Pruvot / SP - Jan 2016
 *
 * Provos Alexis (Tested on SM5.2) - Jan. 2016
 * Reviewed by tpruvot - Feb 2016
 * 
 * Fixed CUDA 7.5 flaw
 * minor code changes
 * code cleanup
 * replaced SSE2 midstate computation with SPH
 * Provos Alexis - Mar 2016
 *
 * Minor boost
 * Provos Alexis - Apr 2016 
 */

#include <stdint.h>
#include <memory.h>

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
}

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

/* threads per block and nonces per thread */
#define TPB 768
#define NPT 384
#define maxResults 8

__constant__ uint32_t _ALIGN(16) c_data[20];

/* 8 adapters max */
static uint32_t		*d_resNonce[MAX_GPUS];
static uint32_t		*h_resNonce[MAX_GPUS];

/* hash by cpu with blake 256 */
extern "C" void blake256_8roundHash(void *output, const void *input){
	uchar hash[64];
	sph_blake256_context ctx;

	sph_blake256_set_rounds(8);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);

	memcpy(output, hash, 32);
}

#define GSn4(a,b,c,d,x,y,a1,b1,c1,d1,x1,y1,a2,b2,c2,d2,x2,y2,a3,b3,c3,d3,x3,y3) { \
	v[ a] = v[ a] + v[ b] + x;		v[a1] = v[a1] + v[b1] + x1;		v[a2] = v[a2] + v[b2] + x2;	 	v[a3] = v[a3] + v[b3] + x3;\
	v[ d] = ROL16(v[ d] ^ v[ a]);		v[d1] = ROL16(v[d1] ^ v[a1]);		v[d2] = ROL16(v[d2] ^ v[a2]);		v[d3] = ROL16(v[d3] ^ v[a3]); \
	v[ c] = v[ c] + v[ d];			v[c1] = v[c1] + v[d1];			v[c2] = v[c2] + v[d2];			v[c3] = v[c3] + v[d3]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);	v[b1] = ROTR32(v[b1] ^ v[c1], 12);	v[b2] = ROTR32(v[b2] ^ v[c2], 12);	v[b3] = ROTR32(v[b3] ^ v[c3], 12); \
	v[ a] = v[ a] + v[ b] + y;		v[a1] = v[a1] + v[b1] + y1;		v[a2] = v[a2] + v[b2] + y2;		v[a3] = v[a3] + v[b3] + y3; \
	v[ d] = ROR8(v[ d] ^ v[ a]);		v[d1] = ROR8(v[d1] ^ v[a1]);		v[d2] = ROR8(v[d2] ^ v[a2]);		v[d3] = ROR8(v[d3] ^ v[a3]); \
	v[ c] = v[ c] + v[ d];			v[c1] = v[c1] + v[d1];			v[c2] = v[c2] + v[d2];			v[c3] = v[c3] + v[d3]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);	v[b1] = ROTR32(v[b1] ^ v[c1], 7);	v[b2] = ROTR32(v[b2] ^ v[c2], 7);	v[b3] = ROTR32(v[b3] ^ v[c3], 7); \
}

#define GS(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ z[y]) + v[b]; \
	v[d] = ROL16(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c],12); \
	v[a] += (m[y] ^ z[x]) + v[b]; \
	v[d] = ROR8(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
}
	
__global__ __launch_bounds__(TPB)
void blake256_8round_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce){
	      uint64_t m3	= startNonce + blockDim.x * blockIdx.x + threadIdx.x;
	const uint64_t step     = gridDim.x * blockDim.x;
	const uint64_t maxNonce = startNonce + threads;

	uint32_t v[16];
	uint32_t z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	uint32_t m[16] = {
				  c_data[16],	c_data[17],	c_data[18],	0,	0x80000000,	0,		0,		0,
				  0,		0,		0,		0,	0,		1,		0,		640
	};

	uint32_t h7 = c_data[19];

	uint32_t xors[16];
	for(; m3<maxNonce;m3+=step){

		m[3]  = _LODWORD(m3);
		
		#pragma unroll 16
		for(int i=0;i<16;i++){
			v[i] = c_data[i];
		}
		v[ 1] = v[ 1] + (m[3] ^ z[2]);		v[13] = ROR8(v[13] ^ v[ 1]);
		v[ 9] = v[ 9] + v[13];			v[ 5] = ROTR32(v[ 5] ^ v[ 9], 7);
		
		v[ 0] = v[ 0] + v[ 5];			v[15] = ROL16(v[15] ^ v[ 0]);
		v[10] = v[10] + v[15];			v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
		v[ 0] = v[ 0] + v[ 5] + z[ 8];		v[15] = ROR8(v[15] ^ v[ 0]);
		v[10] = v[10] + v[15];			v[ 5] = ROTR32(v[ 5] ^ v[10], 7);

		v[ 1] = v[ 1] + v[ 6] + z[11];
		v[12] = ROL16(v[12] ^ v[ 1]);		v[13] = ROL16(v[13] ^ v[ 2]);
		v[11] = v[11] + v[12];			v[ 8] = v[ 8] + v[13];			v[ 9] = v[ 9] + v[14];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 12);	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 12);
		v[ 1] = v[ 1] + v[ 6] + z[10];		v[ 2] = v[ 2] + v[ 7] + (m[13] ^ z[12]);v[ 3] = v[ 3] + v[ 4] + (m[15] ^ z[14]);
		v[12] = ROR8(v[12] ^ v[ 1]);		v[13] = ROR8(v[13] ^ v[ 2]);		v[14] = ROR8(v[14] ^ v[ 3]);
		v[11] = v[11] + v[12];			v[ 8] = v[ 8] + v[13];			v[ 9] = v[ 9] + v[14];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 7);	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 7);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 7);		

		// 1{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
		xors[ 0] = z[10];		xors[ 1] = m[ 4] ^ z[ 8];	xors[ 2] = z[15];		xors[ 3] = m[13] ^ z[ 6];
		xors[ 4] = z[14];		xors[ 5] = z[ 4];		xors[ 6] = z[ 9] ^ m[15];	xors[ 7] = z[13];

		xors[ 8] = m[ 1] ^ z[12];	xors[ 9] = m[ 0] ^ z[ 2];	xors[10] = z[ 7];		xors[11] = z[ 3];
		xors[12] = z[ 1];		xors[13] = z[ 0] ^ m[ 2];	xors[14] = z[11];		xors[15] = z[ 5] ^ m[ 3];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 2{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 }
		xors[ 0] = z[ 8];		xors[ 1] = z[ 0];		xors[ 2] = z[ 2];		xors[ 3] = m[15] ^ z[13];
		xors[ 4] = z[11];		xors[ 5] = z[12] ^ m[ 0];	xors[ 6] = z[ 5] ^ m[ 2];	xors[ 7] = z[15] ^ m[13];

		xors[ 8] = z[14];		xors[ 9] = m[ 3] ^ z[ 6];	xors[10] = z[ 1];		xors[11] = z[ 4];
		xors[12] = z[10];		xors[13] = z[ 3];		xors[14] = z[ 7] ^ m[ 1];	xors[15] = z[ 9] ^ m[ 4];

		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 3{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 }
		xors[ 0] = z[ 9];		xors[ 1] = m[ 3] ^ z[ 1];	xors[ 2] = m[13] ^ z[12];	xors[ 3] = z[14];
		xors[ 4] = z[ 7];		xors[ 5] = z[ 3] ^ m[ 1];	xors[ 6] = z[13];		xors[ 7] = z[11];

		xors[ 8] = m[ 2] ^ z[ 6];	xors[ 9] = z[10];		xors[10] = m[ 4] ^ z[ 0];	xors[11] = m[15] ^ z[ 8];
		xors[12] = z[ 2];		xors[13] = z[ 5];		xors[14] = z[ 4] ^ m[ 0];	xors[15] = z[15];

		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
		
		// 4{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 }
		xors[ 0] = z[ 0];		xors[ 1] = z[ 7];	xors[ 2] = m[ 2] ^ z[ 4];	xors[ 3] = z[15];
		xors[ 4] = z[ 9] ^ m[ 0];	xors[ 5] = z[ 5];	xors[ 6] = z[ 2] ^ m[ 4];	xors[ 7] = z[10] ^ m[15];

		xors[ 8] = z[ 1];		xors[ 9] = z[12];	xors[10] = z[ 8];		xors[11] = m[ 3] ^ z[13];
		xors[12] = z[14] ^ m[ 1];	xors[13] = z[11];	xors[14] = z[ 6];		xors[15] = z[ 3] ^ m[13];

		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 5{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 }
		xors[ 0] = m[ 2] ^ z[12];	xors[ 1] = z[10];	xors[ 2] = m[ 0] ^ z[11];	xors[ 3] = z[ 3];
		xors[ 4] = z[ 2];		xors[ 5] = z[ 6];	xors[ 6] = z[ 0];		xors[ 7] = z[ 8] ^ m[ 3];

		xors[ 8] = m[ 4] ^ z[13];	xors[ 9] = z[ 5];	xors[10] = m[15] ^ z[14];	xors[11] = m[ 1] ^ z[ 9];
		xors[12] = z[ 4] ^ m[13];	xors[13] = z[ 7];	xors[14] = z[15];		xors[15] = z[ 1];

		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 6{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 }
		xors[ 0] = z[ 5];		xors[ 1] = m[ 1] ^ z[15];	xors[ 2] = z[13];		xors[ 3] = m[ 4] ^ z[10];
		xors[ 4] = z[12];		xors[ 5] = z[ 1] ^ m[15];	xors[ 6] = z[14] ^ m[13];	xors[ 7] = z[ 4];

		xors[ 8] = m[ 0] ^ z[ 7];	xors[ 9] = z[ 3];		xors[10] = z[ 2];		xors[11] = z[11];
		xors[12] = z[ 0];		xors[13] = z[ 6] ^ m[ 3];	xors[14] = z[ 9] ^ m[ 2];	xors[15] = z[ 8];

		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 7{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 }
		xors[ 0] = m[13] ^ z[11];	xors[ 1] = z[14];		xors[ 2] = z[ 1];		xors[ 3] = m[ 3] ^ z[ 9];
		xors[ 4] = z[13];		xors[ 5] = z[ 7];		xors[ 6] = z[12] ^ m[ 1];	xors[ 7] = z[ 3];

		xors[ 8] = z[ 0];						xors[10] = z[ 6];
		xors[12] = z[ 5] ^ m[ 0];					xors[14] = z[ 8];

		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		
		v[ 0] = v[ 0] + v[ 5] + xors[ 8];
		v[ 2] = v[ 2] + v[ 7] + xors[10];
		v[15] = ROL16(v[15] ^ v[ 0]);
		v[13] = ROL16(v[13] ^ v[ 2]);
		v[10] = v[10] + v[15];
		v[ 8] = v[ 8] + v[13];
		v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
		v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);
		v[ 0] = v[ 0] + v[ 5] + xors[12];
		v[ 2] = v[ 2] + v[ 7] + xors[14];
		v[13] = ROR8(v[13] ^ v[ 2]);
		v[15] = ROTR32(v[15] ^ v[ 0],1);
				
		v[ 8] += v[13];
		// only compute h7
		if(xor3x(v[ 7],h7,v[ 8])==v[15]){
			uint32_t pos = atomicInc(&resNonce[0],0xffffffff)+1;
			if(pos<maxResults)
				resNonce[pos]=m[ 3];
			return;
		}
	}
}

__host__
void blake256_8round_cpu_setBlock_16(const int thr_id,const uint32_t* endiandata, uint32_t *penddata){

	const uint32_t _ALIGN(64) z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};
	uint32_t _ALIGN(64) h[22];

	sph_blake256_context ctx;

	sph_blake256_set_rounds(8);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, endiandata, 64);
	
	h[ 0] = ctx.H[0];	h[ 1] = ctx.H[1];
	h[ 2] = ctx.H[2];	h[21] = ctx.H[3];
	h[ 4] = ctx.H[4];	h[20] = ctx.H[5];
	h[19] = ctx.H[6];	h[16] = ctx.H[7];

	uint32_t tmp = h[20];
	h[20] = h[19];
	h[19] = h[16];
	h[16] = penddata[ 0];
	h[17] = penddata[ 1];
	h[18] = penddata[ 2];
	h[12] = z[ 4] ^ 640;
	h[ 8] = z[ 0];

	h[ 0] += (h[16] ^ z[ 1]) + h[ 4];
	h[12]  = SPH_ROTR32(h[12] ^ h[0],16);
	h[ 8] += h[12];
	h[ 4]  = SPH_ROTR32(h[ 4] ^ h[ 8], 12);
	h[ 0] += (h[17] ^ z[ 0]) + h[ 4];
	h[12]  = SPH_ROTR32(h[12] ^ h[0],8);
	h[ 8] += h[12];
	h[ 4]  = SPH_ROTR32(h[ 4] ^ h[ 8], 7);

	h[1] += (h[18] ^ z[ 3]) + tmp;

	h[13] = SPH_ROTR32(z[ 5] ^ 640 ^ h[1],16);
	h[ 5] = ROTR32(tmp ^ (z[ 1] + h[13]), 12);

	h[ 1] += h[ 5];
	h[ 2] += (0x80000000UL ^ z[ 5]) + h[20];

	h[14]  = SPH_ROTR32(z[ 6] ^ h[2], 16);
	h[ 6]  = z[ 2] + h[14];
	h[ 6]  = SPH_ROTR32(h[20] ^ h[ 6], 12);

	h[21] += z[ 7] + h[19];
	h[ 0] += z[ 9];

	h[ 2] += z[ 4] + h[ 6];

	h[ 9] = z[ 1] + h[13];
	h[10] = z[ 2] + h[14];
	
	h[14] = SPH_ROTR32(h[14] ^ h[2],8); //0x0321
	h[10]+=h[14];
	
	h[ 6] = SPH_ROTR32(h[ 6] ^ h[10],7);
	h[15] = SPH_ROTR32(z[ 7] ^ h[21],16);
	
	h[11] = z[ 3] + h[15];
	h[ 7] = SPH_ROTR32(h[19] ^ h[11], 12);
	h[ 3] = h[21] + h[ 7] + z[ 6];
	
	h[15] = SPH_ROTR32(h[15] ^ h[ 3],8);
	h[11]+= h[15];
	h[ 7] = ROTR32(h[ 7] ^ h[11],7);

	h[ 2]+= z[13];
	h[ 3]+= z[15];	
	h[ 2]+= h[ 7];
	h[ 3]+= h[ 4];
	h[14] = SPH_ROTR32(h[14] ^ h[ 3], 16);
	
	h[19] = SPH_ROTL32(h[19],7); //align the rotation with v[7] v[15];
	cudaMemcpyToSymbol(c_data, h, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_blake256_8round(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce  = pdata[19];
	int dev_id = device_map[thr_id];

	int intensity = 31;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], maxResults * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*) malloc(maxResults * sizeof(uint32_t));
		if(h_resNonce[thr_id] == NULL){
			gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
			exit(EXIT_FAILURE);
		}
		
		CUDA_LOG_ERROR();
		
		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) endiandata[20];

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);
		
	blake256_8round_cpu_setBlock_16(thr_id,endiandata,&pdata[16]);

	const dim3 grid((throughput + (NPT*TPB)-1)/(NPT*TPB));
	const dim3 block(TPB);
	int rc = 0;
	cudaMemset(d_resNonce[thr_id], 0x00, maxResults*sizeof(uint32_t));
	do {		
		blake256_8round_gpu_hash<<<grid,block>>>(throughput, pdata[19], d_resNonce[thr_id]);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (h_resNonce[thr_id][0] != 0){
			cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], maxResults*sizeof(uint32_t), cudaMemcpyDeviceToHost);
			cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));
			if(h_resNonce[thr_id][0]>(maxResults-1)){
				gpulog(LOG_WARNING,dev_id,"Candidate flood: %u",h_resNonce[thr_id][0]);
				h_resNonce[thr_id][0]=maxResults-1;
			}
			uint32_t i;
			for(i=1;i<h_resNonce[thr_id][0]+1;i++){
				uint32_t vhashcpu[8];
				be32enc(&endiandata[19], h_resNonce[thr_id][i]);
				blake256_8roundHash(vhashcpu, endiandata);
				if (vhashcpu[ 6] <= ptarget[ 6] && fulltest(vhashcpu, ptarget)){
					work_set_target_ratio(work, vhashcpu);
					*hashes_done = pdata[19] - first_nonce + throughput;
					pdata[19] = h_resNonce[thr_id][i];
					rc =1;
					//search for 2nd nonce
					for(uint32_t j=i+1;j<h_resNonce[thr_id][0]+1;j++){
						be32enc(&endiandata[19], h_resNonce[thr_id][j]);
						blake256_8roundHash(vhashcpu, endiandata);
						if(vhashcpu[ 6]<=ptarget[6] && fulltest(vhashcpu, ptarget)){
							pdata[21] = h_resNonce[thr_id][j];
//							if(!opt_quiet)
//								gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %u/%08X - %u/%08X",i,pdata[19],j,pdata[21]);
							if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio[0]) {
								work_set_target_ratio(work, vhashcpu);
								xchg(pdata[21], pdata[19]);
							}
							rc=2;
							break;
						}
					}
					return rc;
				}
			}
		}
		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce;

	return rc;
}

// cleanup
extern "C" void free_blake256_8round(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
