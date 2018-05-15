/**
 * Blake-256 Cuda Kernel (Tested on SM 5.2)
 *
 * Based upon Tanguy Pruvot - Nov. 2014
 * Provos Alexis - Apr. 2016
 */

extern "C" {
#include "sph/sph_blake.h"
}

#include "cuda_helper.h"
#include "miner.h"
#include <memory.h>

#define TPB 768

static const uint32_t  c_IV256[8] = {
	0x6A09E667, 0xBB67AE85,
	0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C,
	0x1F83D9AB, 0x5BE0CD19
};

__device__ __constant__ uint32_t _ALIGN(16) c_h[ 8];
__device__ __constant__ uint32_t _ALIGN(16) c_v[16];
__device__ __constant__ uint32_t _ALIGN(16) c_m[16];
__device__ __constant__ uint32_t _ALIGN(16) c_x[60];

#define GSn(a,b,c,d,x,y) { \
	v[a]+= x + v[b]; \
	v[d] = ROL16(v[d] ^ v[a]); \
	v[c]+= v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
	v[a]+= y + v[b]; \
	v[d] = ROR8(v[d] ^ v[a]); \
	v[c]+= v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
}

#define GSn4(a,b,c,d,x,y,a1,b1,c1,d1,x1,y1,a2,b2,c2,d2,x2,y2,a3,b3,c3,d3,x3,y3) { \
	v[ a]+= x + v[ b];			v[a1]+= x1 + v[b1];			v[a2]+= x2 + v[b2];		 	v[a3]+= x3 + v[b3]; \
	v[ d] = ROL16(v[ d] ^ v[ a]);		v[d1] = ROL16(v[d1] ^ v[a1]);		v[d2] = ROL16(v[d2] ^ v[a2]);		v[d3] = ROL16(v[d3] ^ v[a3]); \
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];				v[c3]+= v[d3]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);	v[b1] = ROTR32(v[b1] ^ v[c1], 12);	v[b2] = ROTR32(v[b2] ^ v[c2], 12);	v[b3] = ROTR32(v[b3] ^ v[c3], 12); \
	v[ a]+= y + v[ b];			v[a1]+= y1 + v[b1];			v[a2]+= y2 + v[b2];			v[a3]+= y3 + v[b3]; \
	v[ d] = ROR8(v[ d] ^ v[ a]);		v[d1] = ROR8(v[d1] ^ v[a1]);		v[d2] = ROR8(v[d2] ^ v[a2]);		v[d3] = ROR8(v[d3] ^ v[a3]); \
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];				v[c3]+= v[d3]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);	v[b1] = ROTR32(v[b1] ^ v[c1], 7);	v[b2] = ROTR32(v[b2] ^ v[c2], 7);	v[b3] = ROTR32(v[b3] ^ v[c3], 7); \
}

#define GSn3(a,b,c,d,x,y,a1,b1,c1,d1,x1,y1,a2,b2,c2,d2,x2,y2) { \
	v[ a]+= x + v[ b];			v[a1]+= x1 + v[b1];			v[a2]+= x2 + v[b2];\
	v[ d] = ROL16(v[ d] ^ v[ a]);		v[d1] = ROL16(v[d1] ^ v[a1]);		v[d2] = ROL16(v[d2] ^ v[a2]);\
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];\
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);	v[b1] = ROTR32(v[b1] ^ v[c1], 12);	v[b2] = ROTR32(v[b2] ^ v[c2], 12);\
	v[ a]+= y + v[ b];			v[a1]+= y1 + v[b1];			v[a2]+= y2 + v[b2];\
	v[ d] = ROR8(v[ d] ^ v[ a]);		v[d1] = ROR8(v[d1] ^ v[a1]);		v[d2] = ROR8(v[d2] ^ v[a2]);\
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];\
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);	v[b1] = ROTR32(v[b1] ^ v[c1], 7);	v[b2] = ROTR32(v[b2] ^ v[c2], 7);\
}

#define hostGS(a,b,c,d,x) { \
	const uint32_t idx1 = c_sigma[r][x]; \
	const uint32_t idx2 = c_sigma[r][x+1]; \
	v[a] += (m[idx1] ^ z[idx2]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[idx2] ^ z[idx1]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
	}

#define hostGSn(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ z[y]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ z[x]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
}

__host__ __forceinline__
static void blake256_14round_compress1st(uint32_t *h, const uint32_t *block, const uint32_t T0){
	uint32_t m[16];
	uint32_t v[16];

	const uint32_t  c_sigma[16][16] = {
		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
		{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
		{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
	};
	const uint32_t  z[16] = {
			0x243F6A88, 0x85A308D3,	0x13198A2E, 0x03707344,	0xA4093822, 0x299F31D0,	0x082EFA98, 0xEC4E6C89,
			0x452821E6, 0x38D01377,	0xBE5466CF, 0x34E90C6C,	0xC0AC29B7, 0xC97C50DD,	0x3F84D5B5, 0xB5470917
	};

	for (int i = 0; i < 16; i++) {
		m[i] = block[i];
	}

	for (int i = 0; i < 8; i++)
		v[i] = h[i];

	v[8] = z[0];
	v[9] = z[1];
	v[10] = z[2];
	v[11] = z[3];

	v[12] = z[4] ^ T0;
	v[13] = z[5] ^ T0;
	v[14] = z[6];
	v[15] = z[7];

	for (int r = 0; r < 14; r++) {
		/* column step */
		hostGS(0, 4, 0x8, 0xC, 0x0);
		hostGS(1, 5, 0x9, 0xD, 0x2);
		hostGS(2, 6, 0xA, 0xE, 0x4);
		hostGS(3, 7, 0xB, 0xF, 0x6);
		/* diagonal step */
		hostGS(0, 5, 0xA, 0xF, 0x8);
		hostGS(1, 6, 0xB, 0xC, 0xA);
		hostGS(2, 7, 0x8, 0xD, 0xC);
		hostGS(3, 4, 0x9, 0xE, 0xE);
	}

	for (int i = 0; i < 16; i++) {
		int j = i & 7;
		h[j] ^= v[i];
	}
}

__global__
void blake256_14round_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint2 * Hash){
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	
	const uint32_t z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	uint32_t v[16];
	uint32_t xors[16];

	if(thread<threads){

		volatile uint32_t nonce = startNonce+thread;
		
		#pragma unroll
		for(int i=0;i<16;i++){
			v[i] = c_v[i];
		}
		
		int i=0;		
		//partial: 0{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 }
		xors[ 0] = c_x[i++];	xors[ 1] = c_x[i++];	xors[ 2] = c_x[i++];	xors[ 3] = z[ 7];
		xors[ 4] = c_x[i++];	xors[ 5] = z[ 2]^nonce;	xors[ 6] = z[ 4];	xors[ 7] = z[ 6];

		xors[ 8] = z[ 9];	xors[ 9] = z[11];	xors[10] = z[13];	xors[11] = z[15];
		xors[12] = z[ 8];	xors[13] = z[10];	xors[14] = c_x[i++];	xors[15] = c_x[i++];
		
		v[ 1]+= xors[ 5];			v[13] = ROR8(v[13] ^ v[1]);
		
		v[ 9]+= v[13];				v[ 5] = ROTR32(v[5] ^ v[9], 7);
		
		v[ 0]+= v[5];				v[15] = ROL16(v[15] ^ v[0]);
		v[10]+= v[15];				v[ 5] = ROTR32(v[5] ^ v[10], 12);
		v[ 0]+= xors[12] + v[5];		v[15] = ROR8(v[15] ^ v[0]);
		v[10]+= v[15];				v[ 5] = ROTR32(v[5] ^ v[10], 7);

//		GSn3(1, 6,11,12,xors[ 9],xors[13],	2, 7, 8,13,xors[10],xors[14],	3, 4, 9,14,xors[11],xors[15]);
		v[ 1]+= xors[ 9] + v[ 6];
		v[12] = ROL16(v[12] ^ v[ 1]);		v[13] = ROL16(v[13] ^ v[ 2]);
		v[11]+= v[12];				v[ 8]+= v[13];				v[ 9]+= v[14];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 12);	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 12);
		v[ 1]+= xors[13] + v[ 6];		v[ 2]+= xors[14] + v[ 7];		v[ 3]+= xors[15] + v[ 4];
		v[12] = ROR8(v[12] ^ v[ 1]);		v[13] = ROR8(v[13] ^ v[ 2]);		v[14] = ROR8(v[14] ^ v[ 3]);
		v[11]+= v[12];				v[ 8]+= v[13];				v[ 9]+= v[14];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 7);	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 7);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 7);

		// 1{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
		xors[ 0] = z[10];	xors[ 1] = c_x[i++];	xors[ 2] = z[15];	xors[ 3] = c_x[i++];
		xors[ 4] = z[14];	xors[ 5] = z[ 4];	xors[ 6] = c_x[i++];	xors[ 7] = z[13];

		xors[ 8] = c_x[i++];	xors[ 9] = c_x[i++];	xors[10] = z[ 7];	xors[11] = z[ 3];
		xors[12] = z[ 1];	xors[13] = c_x[i++];	xors[14] = z[11];	xors[15] = z[ 5]^nonce;
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 2{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 }
		xors[ 0] = z[ 8];	xors[ 1] = z[ 0];	xors[ 2] = z[ 2];	xors[ 3] = c_x[i++];
		xors[ 4] = z[11];	xors[ 5] = c_x[i++];	xors[ 6] = c_x[i++];	xors[ 7] = c_x[i++];

		xors[ 8] = z[14];	xors[ 9] = nonce^z[ 6];	xors[10] = z[ 1];	xors[11] = z[ 4];
		xors[12] = z[10];	xors[13] = z[ 3];	xors[14] = c_x[i++];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 3{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 }
		xors[ 0] = z[ 9];	xors[ 1] = nonce^z[ 1];	xors[ 2] = c_x[i++];	xors[ 3] = z[14];
		xors[ 4] = z[ 7];	xors[ 5] = c_x[i++];	xors[ 6] = z[13];	xors[ 7] = z[11];
		
		xors[ 8] = c_x[i++];	xors[ 9] = z[10];	xors[10] = c_x[i++];	xors[11] = c_x[i++];
		xors[12] = z[ 2];	xors[13] = z[ 5];	xors[14] = c_x[i++];	xors[15] = z[15];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 4{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 }
		xors[ 0] = z[ 0];	xors[ 1] = z[ 7];	xors[ 2] = c_x[i++];	xors[ 3] = z[15];
		xors[ 4] = c_x[i++];	xors[ 5] = z[ 5];	xors[ 6] = c_x[i++];	xors[ 7] = c_x[i++];

		xors[ 8] = z[ 1];	xors[ 9] = z[12];	xors[10] = z[ 8];	xors[11] = nonce^z[13];
		xors[12] = c_x[i++];	xors[13] = z[11];	xors[14] = z[ 6];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 5{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 }
		xors[ 0] = c_x[i++];	xors[ 1] = z[10];	xors[ 2] = c_x[i++];	xors[ 3] = z[ 3];
		xors[ 4] = z[ 2];	xors[ 5] = z[ 6];	xors[ 6] = z[ 0];	xors[ 7] = z[ 8]^nonce;

		xors[ 8] = c_x[i++];	xors[ 9] = z[ 5];	xors[10] = c_x[i++];	xors[11] = c_x[i++];
		xors[12] = c_x[i++];	xors[13] = z[ 7];	xors[14] = z[15];	xors[15] = z[ 1];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 6{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 }
		xors[ 0] = z[ 5];	xors[ 1] = c_x[i++];	xors[ 2] = z[13];	xors[ 3] = c_x[i++];
		xors[ 4] = z[12];	xors[ 5] = c_x[i++];	xors[ 6] = c_x[i++];	xors[ 7] = z[ 4];

		xors[ 8] = c_x[i++];	xors[ 9] = z[ 3];	xors[10] = z[ 2];	xors[11] = z[11];
		xors[12] = z[ 0];	xors[13] = z[ 6]^nonce;	xors[14] = c_x[i++];	xors[15] = z[ 8];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 7{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 }
		xors[ 0] = c_x[i++];	xors[ 1] = z[14];	xors[ 2] = z[ 1];	xors[ 3] = nonce^z[ 9];
		xors[ 4] = z[13];	xors[ 5] = z[ 7];	xors[ 6] = c_x[i++];	xors[ 7] = z[ 3];

		xors[ 8] = z[ 0];	xors[ 9] = c_x[i++];	xors[10] = z[ 6];	xors[11] = c_x[i++];
		xors[12] = c_x[i++];	xors[13] = c_x[i++];	xors[14] = z[ 8];	xors[15] = z[ 2];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 8{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 }
		xors[ 0] = z[15];	xors[ 1] = z[ 9];	xors[ 2] = z[ 3];	xors[ 3] = c_x[i++];
		xors[ 4] = c_x[i++];	xors[ 5] = z[14];	xors[ 6] = z[11]^nonce;	xors[ 7] = z[ 0];

		xors[ 8] = z[ 2];	xors[ 9] = c_x[i++];	xors[10] = c_x[i++];	xors[11] = z[ 5];
		xors[12] = c_x[i++];	xors[13] = z[13];	xors[14] = c_x[i++];	xors[15] = z[10];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 9{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 }
		xors[ 0] = z[ 2];	xors[ 1] = z[ 4];	xors[ 2] = z[ 6];	xors[ 3] = c_x[i++];
		xors[ 4] = c_x[i++];	xors[ 5] = c_x[i++];	xors[ 6] = z[ 7];	xors[ 7] = z[ 1];

		xors[ 8] = c_x[i++];	xors[ 9] = z[14];	xors[10] = nonce^z[12];	xors[11] = c_x[i++];
		xors[12] = z[15];	xors[13] = z[ 9];	xors[14] = z[ 3];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		i=0;
		// 0{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 }
		xors[ 0] = c_x[i++];	xors[ 1] = c_x[i++];	xors[ 2] = c_x[i++];	xors[ 3] = z[ 7];
		xors[ 4] = c_x[i++];	xors[ 5] = z[ 2]^nonce;	xors[ 6] = z[ 4];	xors[ 7] = z[ 6];

		xors[ 8] = z[ 9];	xors[ 9] = z[11];	xors[10] = z[13];	xors[11] = z[15];
		xors[12] = z[ 8];	xors[13] = z[10];	xors[14] = c_x[i++];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 1{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
		xors[ 0] = z[10];	xors[ 1] = c_x[i++];	xors[ 2] = z[15];	xors[ 3] = c_x[i++];
		xors[ 4] = z[14];	xors[ 5] = z[ 4];	xors[ 6] = c_x[i++];	xors[ 7] = z[13];

		xors[ 8] = c_x[i++];	xors[ 9] = c_x[i++];	xors[10] = z[ 7];	xors[11] = z[ 3];
		xors[12] = z[ 1];	xors[13] = c_x[i++];	xors[14] = z[11];	xors[15] = z[ 5]^nonce;
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 2{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 }
		xors[ 0] = z[ 8];	xors[ 1] = z[ 0];	xors[ 2] = z[ 2];	xors[ 3] = c_x[i++];
		xors[ 4] = z[11];	xors[ 5] = c_x[i++];	xors[ 6] = c_x[i++];	xors[ 7] = c_x[i++];

		xors[ 8] = z[14];	xors[ 9] = nonce^z[ 6];	xors[10] = z[ 1];	xors[11] = z[ 4];
		xors[12] = z[10];	xors[13] = z[ 3];	xors[14] = c_x[i++];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
		
		// 3{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 }
		xors[ 0] = z[ 9];	xors[ 1] = nonce^z[ 1];	xors[ 2] = c_x[i++];	xors[ 3] = z[14];
		xors[ 4] = z[ 7];	xors[ 5] = c_x[i++];	xors[ 6] = z[13];	xors[ 7] = z[11];
		xors[ 8] = c_x[i++];	xors[ 9] = z[10];	xors[10] = c_x[i++];	xors[11] = c_x[i++];
		xors[12] = z[ 2];	xors[13] = z[ 5];	xors[14] = c_x[i++];	xors[15] = z[15];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
		
			
		Hash[0*threads + thread] = make_uint2(cuda_swab32(xor3x(c_h[ 0],v[ 0],v[ 8])), cuda_swab32(xor3x(c_h[ 1],v[ 1],v[ 9])));
		Hash[1*threads + thread] = make_uint2(cuda_swab32(xor3x(c_h[ 2],v[ 2],v[10])), cuda_swab32(xor3x(c_h[ 3],v[ 3],v[11])));
		Hash[2*threads + thread] = make_uint2(cuda_swab32(xor3x(c_h[ 4],v[ 4],v[12])), cuda_swab32(xor3x(c_h[ 5],v[ 5],v[13])));
		Hash[3*threads + thread] = make_uint2(cuda_swab32(xor3x(c_h[ 6],v[ 6],v[14])), cuda_swab32(xor3x(c_h[ 7],v[ 7],v[15])));
	}
}

__host__
void blake256_14round_cpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint2* d_Hash){
	const dim3 grid((threads -1)/(TPB));
	const dim3 block(TPB);
	blake256_14round_gpu_hash_80 <<<grid, block>>> (threads, startNonce, d_Hash);
}

__host__
void blake256_14round_cpu_setBlock_80(const uint32_t *pdata){
	uint32_t _ALIGN(64) h[8];
	uint32_t _ALIGN(64) v[16];
	uint32_t _ALIGN(64) data[20];
	uint32_t _ALIGN(64) x[60];
	
	memcpy(data, pdata, 80);
	memcpy(h, c_IV256, sizeof(c_IV256));
	blake256_14round_compress1st(h, pdata, 512);

	cudaMemcpyToSymbol(c_h, h, 8*sizeof(uint32_t), 0);
	
	const uint32_t m[16] 	= { 	pdata[16],	pdata[17],	pdata[18],	0,
					0x80000000,	0,		0,		0,
					0,		0,		0,		0,
					0,		1,		0,		640
				};

	cudaMemcpyToSymbol(c_m, m, 16*sizeof(uint32_t), 0);
	
	const uint32_t  z[16] = {
		0x243F6A88, 0x85A308D3,	0x13198A2E, 0x03707344,	0xA4093822, 0x299F31D0,	0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377,	0xBE5466CF, 0x34E90C6C,	0xC0AC29B7, 0xC97C50DD,	0x3F84D5B5, 0xB5470917
	};
	
	v[ 0] = h[ 0];			v[ 1] = h[ 1];			v[ 2] = h[ 2];			v[ 3] = h[ 3];
	v[ 4] = h[ 4];			v[ 5] = h[ 5];			v[ 6] = h[ 6];			v[ 7] = h[ 7];
	v[ 8] = z[ 0];			v[ 9] = z[ 1];			v[10] = z[ 2];			v[11] = z[ 3];
	v[12] = z[ 4] ^ 640;		v[13] = z[ 5] ^ 640;		v[14] = z[ 6];			v[15] = z[ 7];

	hostGSn(0, 4, 8,12, 0, 1);
	hostGSn(2, 6,10,14, 4, 5);
	hostGSn(3, 7,11,15, 6, 7);
	
	v[ 1]+= (m[ 2] ^ z[ 3]) + v[ 5];
	v[13] = ROTR32(v[13] ^ v[ 1],16);
	v[ 9] += v[13];
	v[ 5] = ROTR32(v[ 5] ^ v[ 9],12);
	
	v[ 1]+= v[ 5];
	v[ 0]+= z[ 9];
	
	v[ 2]+= z[13] + v[ 7];
	v[ 3]+= z[15] + v[ 4];
	v[14] = ROTL32(v[14] ^ v[ 3],16);
	
	cudaMemcpyToSymbol(c_v, v, 16*sizeof(uint32_t), 0);
	
	int i=0;
	x[i++] = m[ 0]^z[ 1];	x[i++] = m[ 2]^z[ 3];	x[i++] = m[ 4]^z[ 5];	x[i++] = z[ 0]^m[ 1]; 	x[i++] = z[12]^m[13];	x[i++] = z[14]^m[15];
//1
	x[i++] = m[ 4]^z[ 8];	x[i++] = m[13]^z[ 6];	x[i++] = z[ 9]^m[15];	x[i++] = m[ 1]^z[12];	x[i++] = m[ 0]^z[ 2];	x[i++] = z[ 0]^m[ 2];
//2
	x[i++] = m[15]^z[13];	x[i++] = z[12]^m[ 0];	x[i++] = z[ 5]^m[ 2];	x[i++] = z[15]^m[13];	x[i++] = z[ 7]^m[ 1];	x[i++] = z[ 9]^m[ 4];
//3
	x[i++] = z[12]^m[13];	x[i++] = z[ 3]^m[ 1];	x[i++] = m[ 2]^z[ 6];	x[i++] = m[ 4]^z[ 0];	x[i++] = m[15]^z[ 8];	x[i++] = z[ 4]^m[ 0];
//4
	x[i++] = m[ 2]^z[ 4];	x[i++] = z[ 9]^m[ 0];	x[i++] = z[ 2]^m[ 4];	x[i++] = z[10]^m[15];	x[i++] = z[14]^m[ 1];	x[i++] = z[ 3]^m[13];
//5
	x[i++] = m[ 2]^z[12];	x[i++] = m[ 0]^z[11];	x[i++] = m[ 4]^z[13];	x[i++] = z[14]^m[15];	x[i++] = m[ 1]^z[ 9];	x[i++] = z[ 4]^m[13];
//6
	x[i++] = m[ 1]^z[15];	x[i++] = m[ 4]^z[10];	x[i++] = z[ 1]^m[15];	x[i++] = z[14]^m[13];	x[i++] = m[ 0]^z[ 7];	x[i++] = z[ 9]^m[ 2];
//7
	x[i++] = m[13]^z[11];	x[i++] = z[12]^m[ 1];	x[i++] = m[15]^z[ 4];	x[i++] = m[ 2]^z[10];	x[i++] = z[ 5]^m[ 0];	x[i++] = z[15]^m[ 4];
//8
	x[i++] = m[ 0]^z[ 8];	x[i++] = z[ 6]^m[15];	x[i++] = m[13]^z[ 7];	x[i++] = m[ 1]^z[ 4];	x[i++] = z[12]^m[ 2];	x[i++] = z[ 1]^m[ 4];
//9
	x[i++] = m[ 1]^z[ 5];	x[i++] = z[10]^m[ 2];	x[i++] = z[ 8]^m[ 4];	x[i++] = m[15]^z[11];	x[i++] = m[13]^z[ 0];	x[i++] = z[13]^m[ 0];
	
	cudaMemcpyToSymbol(c_x, x, i*sizeof(uint32_t), 0);
}
