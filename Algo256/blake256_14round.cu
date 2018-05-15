/**
 * 14-round Blake-256 Cuda Kernel (Tested on SM 5.2) for SaffronCoin
 * Provos Alexis - April 2016
 *
 * Based on blake256 ccminer implementation of
 * Tanguy Pruvot / SP - Jan 2016
 *
 * Previous implementation under cuda7.5:
 * ccminer 1.8.2 on ASUS Strix 970:       1415.00MH/s - 1265MHz / intensity 31
 * ccminer 1.8.2 on GB windforce 750ti OC: 574.30MH/s - 1320MHz / intensity 30
 *
 * Further improved under CUDA 7.5
 * ASUS Strix 970:        1556.6MH/s - 1252MHz / intensity 31
 * GB windforce 750ti OC:  616MH/s - 1320MHz / intensity 30
 */

#include <stdint.h>
#include <memory.h>

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
}

/* threads per block and nonces per thread */
#define TPB 768
#define NPT 192
#define maxResults 16
/* max count of found nonces in one call */
#define NBN 2

/* hash by cpu with blake 256 */
extern "C" void blake256_14roundHash(void *output, const void *input)
{
	uchar hash[64];
	sph_blake256_context ctx;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);

	memcpy(output, hash, 32);
}

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

__constant__ uint32_t _ALIGN(32) c_v[16];
__constant__ uint32_t _ALIGN(8) c_h[ 2];
__constant__ uint32_t c_m[ 3];
__constant__ uint32_t _ALIGN(32) c_x[90];

/* 8 adapters max */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

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

#define hostGS(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ z[y]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ z[x]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
}

__global__ __launch_bounds__(TPB,1)
void blake256_14round_gpu_hash_16(const uint32_t threads,const uint32_t startNonce, uint32_t *resNonce){

	      uint64_t m3		= startNonce + blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t step		= gridDim.x * blockDim.x;
	const uint64_t maxNonce		= startNonce + threads;
	
	const uint32_t z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	uint32_t v[16];
	uint32_t m[16];

	#pragma unroll
	for(int i=0;i<3;i++){
		m[i] = c_m[i];
	}
	m[13] = 1;
	m[15] = 640;

	const uint32_t m130 = z[12]^m[13];
	const uint32_t m131 = m[13]^z[ 6];
	const uint32_t m132 = z[15]^m[13];
	const uint32_t m133 = z[ 3]^m[13];
	const uint32_t m134 = z[ 4]^m[13];
	const uint32_t m135 = z[14]^m[13];
	const uint32_t m136 = m[13]^z[11];
	const uint32_t m137 = m[13]^z[ 7];
	const uint32_t m138 = m[13]^z[ 0];

	volatile uint32_t m150 = z[14]^m[15];
	volatile uint32_t m151 = z[ 9]^m[15];
	volatile uint32_t m152 = m[15]^z[13];
	volatile uint32_t m153 = m[15]^z[ 8];
	const uint32_t m154 = z[10]^m[15];
	const uint32_t m155 = z[ 1]^m[15];
	const uint32_t m156 = m[15]^z[ 4];
	const uint32_t m157 = z[ 6]^m[15];
	const uint32_t m158 = m[15]^z[11];
	
	const uint32_t h7	= c_h[ 0];

	for( ; m3<maxNonce ; m3+=step){

		m[ 3] = m3;
		
		#pragma unroll 16
		for(int i=0;i<16;i++){
			v[i] = c_v[i];
		}
		
		uint32_t xors[16],i=0;

		//partial: 0{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 }
					xors[ 5] = z[ 2]^m[ 3];	

					xors[ 9] = c_x[i++];	xors[10] = c_x[i++];	xors[11] = z[15];
		xors[12]=c_x[i++];	xors[13] = c_x[i++];	xors[14] = m130;	xors[15] = m150;
		
		v[ 1]+= xors[ 5];			v[13] = ROR8(v[13] ^ v[1]);
		
		v[ 9]+= v[13];				v[ 5] = ROTR32(v[5] ^ v[9], 7);
		
		v[ 0]+= v[5];				v[15] = ROL16(v[15] ^ v[0]);
		v[10]+= v[15];				v[ 5] = ROTR32(v[5] ^ v[10], 12);
		v[ 0]+= xors[12] + v[5];		v[15] = ROR8(v[15] ^ v[0]);
		v[10]+= v[15];				v[ 5] = ROTR32(v[5] ^ v[10], 7);
//i=3
		GSn3(1, 6,11,12,xors[ 9],xors[13],	2, 7, 8,13,xors[10],xors[14],	3, 4, 9,14,xors[11],xors[15]);

		// 1{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
		xors[ 0] = z[10];	xors[ 1] = c_x[i++];	xors[ 2] = c_x[i++];	xors[ 3] = m131;
		xors[ 8] = m[ 1]^z[12];	xors[ 9] = m[ 0]^z[ 2];	xors[10] = c_x[i++];	xors[11] = c_x[i++];
		
		xors[ 4] = c_x[i++];	xors[ 5] = c_x[i++];	xors[ 6] = m151;	xors[ 7] = c_x[i++];
		xors[12] = c_x[i++];	xors[13] = z[ 0]^m[ 2];	xors[14] = c_x[i++];	xors[15] = z[ 5]^m[ 3];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=12
		// 2{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 }
		xors[ 0] = c_x[i++];	xors[ 1] = c_x[i++];	xors[ 2] = c_x[i++];	xors[ 3] = m152;
		xors[ 8] = c_x[i++];	xors[ 9] = m[ 3]^z[ 6];	xors[10] = c_x[i++];	xors[11] = c_x[i++];
		
		xors[ 4] = c_x[i++];	xors[ 5] = z[12]^m[ 0];	xors[ 6] = z[ 5]^m[ 2];	xors[ 7] = m132;
		xors[12] = z[10];	xors[13] = c_x[i++];	xors[14] = z[ 7]^m[ 1];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=21
		// 3{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 }
		xors[ 0] = c_x[i++];	xors[ 1] = m[ 3]^z[ 1];	xors[ 2] = m130;	xors[ 3] = c_x[i++];
		xors[ 8] = m[ 2]^z[ 6];	xors[ 9] = c_x[i++];	xors[10] = c_x[i++];	xors[11] = m153;
		
		xors[ 4] = c_x[i++];	xors[ 5] = z[ 3]^m[ 1];	xors[ 6] = c_x[i++];	xors[ 7] = z[11];
		xors[12] = c_x[i++];	xors[13] = c_x[i++];	xors[14] = z[ 4]^m[ 0];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=30
		// 4{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 }
		xors[ 0] = c_x[i++];	xors[ 1] = c_x[i++];	xors[ 2] = m[ 2]^z[ 4];	xors[ 3] = c_x[i++];
		xors[ 8] = z[ 1];	xors[ 9] = c_x[i++];	xors[10] = c_x[i++];	xors[11] = m[ 3]^z[13];
		
		xors[ 4] = z[ 9]^m[ 0];	xors[ 5] = c_x[i++];	xors[ 6] = c_x[i++];	xors[ 7] = m154;
		xors[12] = z[14]^m[ 1];	xors[13] = c_x[i++];	xors[14] = c_x[i++];	xors[15] = m133;
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=39
		// 5{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 }
		xors[ 0] = m[ 2]^z[12];	xors[ 1] = c_x[i++];	xors[ 2] = m[ 0]^z[11];	xors[ 3] = c_x[i++];
		xors[ 8] = c_x[i++];	xors[ 9] = c_x[i++];	xors[10] = m150;	xors[11] = m[ 1]^z[ 9];
		
		xors[ 4] = c_x[i++];	xors[ 5] = c_x[i++];	xors[ 6] = c_x[i++];	xors[ 7] = z[ 8]^m[ 3];
		xors[12] = m134;	xors[13] = c_x[i++];	xors[14] = z[15];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=48
		// 6{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 }
		xors[ 0] = c_x[i++];	xors[ 1] = m[ 1]^z[15];	xors[ 2] = z[13];	xors[ 3] = c_x[i++];
		xors[ 8] = m[ 0]^z[ 7];	xors[ 9] = c_x[i++];	xors[10] = c_x[i++];	xors[11] = c_x[i++];
		
		xors[ 4] = c_x[i++];	xors[ 5] = m155;	xors[ 6] = m135;	xors[ 7] = c_x[i++];
		xors[12] = c_x[i++];	xors[13] = z[ 6]^m[ 3];	xors[14] = z[ 9]^m[ 2];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=57
		// 7{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 }
		xors[ 0] = m136;	xors[ 1] = c_x[i++];	xors[ 2] = c_x[i++];	xors[ 3] = m[ 3]^z[ 9];
		xors[ 8] = c_x[i++];	xors[ 9] = m156;	xors[10] = c_x[i++];	xors[11] = m[ 2]^z[10];
		
		xors[ 4] = c_x[i++];	xors[ 5] = z[ 7];	xors[ 6] = z[12]^m[ 1];	xors[ 7] = c_x[i++];
		xors[12] = z[ 5]^m[ 0];	xors[13] = c_x[i++];	xors[14] = c_x[i++];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=66
		// 8{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 }
		xors[ 0] = c_x[i++];	xors[ 1] = z[ 9];	xors[ 2] = c_x[i++];	xors[ 3] = m[ 0]^z[ 8];
		xors[ 8] = c_x[i++];	xors[ 9] = m137;	xors[10] = m[ 1]^z[ 4];	xors[11] = c_x[i++];
		
		xors[ 4] = m157;	xors[ 5] = c_x[i++];	xors[ 6] = z[11]^m[ 3];	xors[ 7] = c_x[i++];
		xors[12] = z[12]^m[ 2];	xors[13] = c_x[i++];	xors[14] = c_x[i++];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=75
		// 9{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 }
		xors[ 0] = c_x[i++];	xors[ 1] = c_x[i++];	xors[ 2] = c_x[i++];	xors[ 3] = m[ 1]^z[ 5];
		xors[ 8] = m158;	xors[ 9] = c_x[i++];	xors[10] = m[ 3]^z[12];	xors[11] = m138;
		
		xors[ 4] = z[10]^m[ 2];	xors[ 5] = c_x[i++];	xors[ 6] = c_x[i++];	xors[ 7] = c_x[i++];
		xors[12] = c_x[i++];	xors[13] = z[ 9];	xors[14] = c_x[i++];	xors[15] = z[13]^m[ 0];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=85	
		// 0{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 }
		xors[ 0] = m[ 0]^z[ 1];	xors[ 1] = m[ 2]^z[ 3];	xors[ 2] = c_x[i++];	xors[ 3] = c_x[i++];
		xors[ 8] = c_x[i++];	xors[ 9] = c_x[ 0];	xors[10] = c_x[ 1];	xors[11] = z[15];
		
		xors[ 4] = z[ 0]^m[ 1];	xors[ 5] = z[ 2]^m[ 3];	xors[ 6] = c_x[i++];	xors[ 7] = c_x[i++];
		xors[12] = c_x[ 2];	xors[13] = c_x[ 3];	xors[14] = m130;	xors[15] = m150;
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
//i=90
		i=4;
		// 1{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
		xors[ 0] = z[10];	xors[ 1] = c_x[i++];	xors[ 2] = c_x[i++];	xors[ 3] = m131;
		xors[ 8] = m[ 1]^z[12];	xors[ 9] = m[ 0]^z[ 2];	xors[10] = c_x[i++];	xors[11] = c_x[i++];
		
		xors[ 4] = c_x[i++];	xors[ 5] = c_x[i++];	xors[ 6] = m151;	xors[ 7] = c_x[i++];
		xors[12] = c_x[i++];	xors[13] = z[ 0]^m[ 2];	xors[14] = c_x[i++];	xors[15] = z[ 5]^m[ 3];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);

		// 2{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 }
		xors[ 0] = c_x[i++];	xors[ 1] = c_x[i++];	xors[ 2] = c_x[i++];	xors[ 3] = m152;
		xors[ 8] = c_x[i++];	xors[ 9] = m[ 3]^z[ 6];	xors[10] = c_x[i++];	xors[11] = c_x[i++];
		
		xors[ 4] = c_x[i++];	xors[ 5] = z[12]^m[ 0];	xors[ 6] = z[ 5]^m[ 2];	xors[ 7] = m132;
		xors[12] = z[10];	xors[13] = c_x[i++];	xors[14] = z[ 7]^m[ 1];	xors[15] = c_x[i++];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15,	xors[ 8],xors[12],	1, 6,11,12,	xors[ 9],xors[13],	2, 7, 8,13,	xors[10],xors[14],	3, 4, 9,14,	xors[11],xors[15]);
		
		// 3{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 }
		xors[ 0] = c_x[i++];	xors[ 1] = m[ 3]^z[ 1];	xors[ 2] = m130;	xors[ 3] = c_x[i++];
		xors[ 8] = m[ 2]^z[ 6];	i++;			xors[10] = c_x[i++];
		
		xors[ 4] = c_x[i++];	xors[ 5] = z[ 3]^m[ 1];	xors[ 6] = c_x[i++];	xors[ 7] = z[11];
		xors[12] = c_x[i++];				xors[14] = z[ 4]^m[ 0];
				
		GSn4(0, 4, 8,12,	xors[ 0],xors[ 4],	1, 5, 9,13,	xors[ 1],xors[ 5],	2, 6,10,14,	xors[ 2],xors[ 6],	3, 7,11,15,	xors[ 3],xors[ 7]);
		
		v[ 0]+= xors[ 8] + v[ 5];
		v[ 2]+= xors[10] + v[ 7];
		v[15] = ROL16(v[15] ^ v[ 0]);
		v[13] = ROL16(v[13] ^ v[ 2]);
		v[10]+= v[15];
		v[ 8]+= v[13];
		v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
		v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);
		v[ 0]+= xors[12] + v[ 5];
		v[ 2]+= xors[14] + v[ 7];
		v[15] = ROTR32(v[15] ^ v[ 0],1);
		v[13] = ROR8(v[13] ^ v[ 2]);
		v[ 8]+= v[13];
		if(xor3x(v[ 7],h7,v[ 8])==v[15]){
			uint32_t pos = atomicInc(&resNonce[0],0xffffffff)+1;
			if(pos<maxResults)
				resNonce[pos]=m[ 3];
			return;
		}
	}
}

__host__
void blake256_14round_cpu_setBlock_16(const uint32_t *pend,const uint32_t *input)
{
	const uint32_t z[16] = {
		0x243F6A88UL, 0x85A308D3UL, 0x13198A2EUL, 0x03707344UL,0xA4093822UL, 0x299F31D0UL, 0x082EFA98UL, 0xEC4E6C89UL,
		0x452821E6UL, 0x38D01377UL, 0xBE5466CFUL, 0x34E90C6CUL,0xC0AC29B7UL, 0xC97C50DDUL, 0x3F84D5B5UL, 0xB5470917UL
	};
	
	sph_u32 _ALIGN(64) v[16];
	sph_u32 _ALIGN(64) h[ 2];

	sph_blake256_context ctx;
	sph_blake256_set_rounds(14);
	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 64);

	v[ 0] = ctx.H[ 0];	v[ 1] = ctx.H[ 1];	v[ 2] = ctx.H[ 2];	v[ 3] = ctx.H[ 3];
	v[ 4] = ctx.H[ 4];	v[ 5] = ctx.H[ 5];	v[ 6] = ctx.H[ 6];	v[ 7] = ctx.H[ 7];
	v[ 8] = z[ 0];		v[ 9] = z[ 1];		v[10] = z[ 2];		v[11] = z[ 3];
	v[12] = z[ 4] ^ 640;	v[13] = z[ 5] ^ 640;	v[14] = z[ 6];		v[15] = z[ 7];
	
	const uint32_t m[16] 	= { 	pend[ 0],	pend[ 1], pend[ 2],	0,
					0x80000000,	0,		0,		0,
					0,		0,		0,		0,
					0,		1,		0,		640
				};

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_m,m, 3*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	
	h[ 0] = v[ 7];

	hostGS(	0, 4, 8,12, 0, 1);
	hostGS(	2, 6,10,14, 4, 5);
	hostGS(	3, 7,11,15, 6, 7);
	
	v[ 1]+= (m[ 2] ^ z[ 3]) + v[ 5];
	v[13] = ROTR32(v[13] ^ v[ 1],16);
	v[ 9] += v[13];
	v[ 5] = ROTR32(v[ 5] ^ v[ 9],12);
	
	v[ 1]+= v[ 5];
	v[ 0]+= z[ 9];

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_v, v,16*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

	h[ 0] = SPH_ROTL32(h[ 0], 7); //align the rotation with v[7] v[15];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_h,h, 1*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	
	uint32_t x[90];
	int i=0;
	
	x[i++] = m[10]^z[11];	x[i++] = m[12]^z[13];	x[i++] = m[ 9]^z[ 8];	x[i++] = z[10]^m[11];	x[i++] = m[ 4]^z[ 8];	x[i++] = m[ 9]^z[15];	x[i++] = m[11]^z[ 7];	x[i++] = m[ 5]^z[ 3];
	x[i++] = z[14]^m[10];	x[i++] = z[ 4]^m[ 8];	x[i++] = z[13]^m[ 6];	x[i++] = z[ 1]^m[12];	x[i++] = z[11]^m[ 7];	x[i++] = m[11]^z[ 8];	x[i++] = m[12]^z[ 0];	x[i++] = m[ 5]^z[ 2];
	x[i++] = m[10]^z[14];	x[i++] = m[ 7]^z[ 1];	x[i++] = m[ 9]^z[ 4];	x[i++] = z[11]^m[ 8];	x[i++] = z[ 3]^m[ 6];	x[i++] = z[ 9]^m[ 4];	x[i++] = m[ 7]^z[ 9];	x[i++] = m[11]^z[14];
	x[i++] = m[ 5]^z[10];	x[i++] = m[ 4]^z[ 0];	x[i++] = z[ 7]^m[ 9];	x[i++] = z[13]^m[12];	x[i++] = z[ 2]^m[ 6];	x[i++] = z[ 5]^m[10];	x[i++] = z[15]^m[ 8];	x[i++] = m[ 9]^z[ 0];
	x[i++] = m[ 5]^z[ 7];	x[i++] = m[10]^z[15];	x[i++] = m[11]^z[12];	x[i++] = m[ 6]^z[ 8];	x[i++] = z[ 5]^m[ 7];	x[i++] = z[ 2]^m[ 4];	x[i++] = z[11]^m[12];	x[i++] = z[ 6]^m[ 8];
	x[i++] = m[ 6]^z[10];	x[i++] = m[ 8]^z[ 3];	x[i++] = m[ 4]^z[13];	x[i++] = m[ 7]^z[ 5];	x[i++] = z[ 2]^m[12];	x[i++] = z[ 6]^m[10];	x[i++] = z[ 0]^m[11];	x[i++] = z[ 7]^m[ 5];
	x[i++] = z[ 1]^m[ 9];	x[i++] = m[12]^z[ 5];	x[i++] = m[ 4]^z[10];	x[i++] = m[ 6]^z[ 3];	x[i++] = m[ 9]^z[ 2];	x[i++] = m[ 8]^z[11];	x[i++] = z[12]^m[ 5];	x[i++] = z[ 4]^m[10];
	x[i++] = z[ 0]^m[ 7];	x[i++] = z[ 8]^m[11];	x[i++] = m[ 7]^z[14];	x[i++] = m[12]^z[ 1];	x[i++] = m[ 5]^z[ 0];	x[i++] = m[ 8]^z[ 6];	x[i++] = z[13]^m[11];	x[i++] = z[ 3]^m[ 9];
	x[i++] = z[15]^m[ 4];	x[i++] = z[ 8]^m[ 6];	x[i++] = z[ 2]^m[10];	x[i++] = m[ 6]^z[15];	x[i++] = m[11]^z[ 3];	x[i++] = m[12]^z[ 2];	x[i++] = m[10]^z[ 5];	x[i++] = z[14]^m[ 9];
	x[i++] = z[ 0]^m[ 8];	x[i++] = z[13]^m[ 7];	x[i++] = z[ 1]^m[ 4];	x[i++] = z[10]^m[ 5];	x[i++] = m[10]^z[ 2];	x[i++] = m[ 8]^z[ 4];	x[i++] = m[ 7]^z[ 6];	x[i++] = m[ 9]^z[14];
	x[i++] = z[ 8]^m[ 4];	x[i++] = z[ 7]^m[ 6];	x[i++] = z[ 1]^m[ 5];	x[i++] = z[15]^m[11];	x[i++] = z[ 3]^m[12];	x[i++] = m[ 4]^z[ 5];	x[i++] = m[ 6]^z[ 7];	x[i++] = m[ 8]^z[ 9];
	x[i++] = z[ 4]^m[ 5];	x[i++] = z[ 6]^m[ 7];
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x, x, i*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

/* ############################################################################################################################### */

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_blake256_14round(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500) ? 31 : 30;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	const dim3 grid((throughput + (NPT*TPB)-1)/(NPT*TPB));
	const dim3 block(TPB);
	
	int rc = 0;

	if (opt_benchmark) {
		ptarget[6] = swab32(0xff);
	}

	if (!init[thr_id])
	{
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
	
	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256_14round_cpu_setBlock_16(&pdata[16], endiandata);
	cudaMemset(d_resNonce[thr_id], 0x00, maxResults*sizeof(uint32_t));
	do {
		// GPU HASH
		blake256_14round_gpu_hash_16<<<grid, block>>>(throughput, pdata[19], d_resNonce[thr_id]);
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
				blake256_14roundHash(vhashcpu, endiandata);
				if (vhashcpu[ 6] <= ptarget[ 6] && fulltest(vhashcpu, ptarget)){
					work_set_target_ratio(work, vhashcpu);
					*hashes_done = pdata[19] - first_nonce + throughput;
					pdata[19] = h_resNonce[thr_id][i];
					rc = 1;
					//search for 2nd nonce
					for(uint32_t j=i+1;j<h_resNonce[thr_id][0]+1;j++){
						be32enc(&endiandata[19], h_resNonce[thr_id][j]);
						blake256_14roundHash(vhashcpu, endiandata);
						if (vhashcpu[ 6] <= ptarget[ 6] && fulltest(vhashcpu, ptarget)) {
							pdata[21] = h_resNonce[thr_id][j];
//							if(!opt_quiet)
//								gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %u/%08X - %u/%08X",i,pdata[19],j,pdata[21]);
							if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio[0]) {
								work_set_target_ratio(work, vhashcpu);
								xchg(pdata[21], pdata[19]);
							}
							rc = 2;
							break;
						}
					}
					return rc;
				}
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && (uint64_t)max_nonce > (uint64_t)throughput + pdata[19]);

	*hashes_done = pdata[19] - first_nonce;

	return rc;
}

// cleanup
extern "C" void free_blake256_14round(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}

