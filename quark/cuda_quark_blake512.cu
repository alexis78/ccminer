/*
	Based upon Tanguy Pruvot's and SP's work
			
	Provos Alexis - 2016
*/
#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h"

#define TPB80 512

#define TPB52_64 192
#define TPB50_64 192

__constant__ uint2 _ALIGN(16) c_m[16]; // padded message (80 bytes + padding)

__constant__ uint2 _ALIGN(16) c_v[16]; //state

__constant__ uint2 _ALIGN(16) c_x[128]; //precomputed xors

// ---------------------------- BEGIN CUDA quark_blake512 functions ------------------------------------

__constant__ _ALIGN(16) uint2 z[16] =
{
	{0x85a308d3,0x243f6a88},{0x03707344,0x13198a2e},{0x299f31d0,0xa4093822},{0xec4e6c89,0x082efa98},
	{0x38d01377,0x452821e6},{0x34e90c6c,0xbe5466cf},{0xc97c50dd,0xc0ac29b7},{0xb5470917,0x3f84d5b5},
	{0x8979fb1b,0x9216d5d9},{0x98dfb5ac,0xd1310ba6},{0xd01adfb7,0x2ffd72db},{0x6a267e96,0xb8e1afed},
	{0xf12c7f99,0xba7c9045},{0xb3916cf7,0x24a19947},{0x858efc16,0x0801f2e2},{0x71574e69,0x636920d8}
};

__constant__ const uint2 h[8] = {
		{ 0xf3bcc908UL, 0x6a09e667UL },
		{ 0x84caa73bUL, 0xbb67ae85UL },
		{ 0xfe94f82bUL, 0x3c6ef372UL },
		{ 0x5f1d36f1UL, 0xa54ff53aUL },
		{ 0xade682d1UL, 0x510e527fUL },
		{ 0x2b3e6c1fUL, 0x9b05688cUL },
		{ 0xfb41bd6bUL, 0x1f83d9abUL },
		{ 0x137e2179UL, 0x5be0cd19UL }
	};

#define G4(x, a,b,c,d,a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3) { \
	v[a] += (m[c_sigma[i][x]] ^ z[c_sigma[i][x+1]]) + v[b]; \
	v[a1] += (m[c_sigma[i][x+2]] ^ z[c_sigma[i][x+3]]) + v[b1]; \
	v[a2] += (m[c_sigma[i][x+4]] ^ z[c_sigma[i][x+5]]) + v[b2]; \
	v[a3] += (m[c_sigma[i][x+6]] ^ z[c_sigma[i][x+7]]) + v[b3]; \
	v[d] = xorswap32(v[d] , v[a]); \
	v[d1] = xorswap32(v[d1] , v[a1]); \
	v[d2] = xorswap32(v[d2] , v[a2]); \
	v[d3] = xorswap32(v[d3] , v[a3]); \
	v[c] += v[d]; \
	v[c1] += v[d1]; \
	v[c2] += v[d2]; \
	v[c3] += v[d3]; \
	v[b] = ROR2( v[b] ^ v[c], 25); \
	v[b1] = ROR2( v[b1] ^ v[c1], 25); \
	v[b2] = ROR2( v[b2] ^ v[c2], 25); \
	v[b3] = ROR2( v[b3] ^ v[c3], 25); \
	v[a] += (m[c_sigma[i][x+1]] ^ z[c_sigma[i][x]]) + v[b]; \
	v[a1] += (m[c_sigma[i][x+3]] ^ z[c_sigma[i][x+2]]) + v[b1]; \
	v[a2] += (m[c_sigma[i][x+5]] ^ z[c_sigma[i][x+4]]) + v[b2]; \
	v[a3] += (m[c_sigma[i][x+7]] ^ z[c_sigma[i][x+6]]) + v[b3]; \
	v[d] = ROR16( v[d] ^ v[a]); \
	v[d1] = ROR16( v[d1] ^ v[a1]); \
	v[d2] = ROR16( v[d2] ^ v[a2]); \
	v[d3] = ROR16( v[d3] ^ v[a3]); \
	v[c] += v[d]; \
	v[c1] += v[d1]; \
	v[c2] += v[d2]; \
	v[c3] += v[d3]; \
	v[b] = ROR2( v[b] ^ v[c], 11); \
	v[b1] = ROR2( v[b1] ^ v[c1], 11); \
	v[b2] = ROR2( v[b2] ^ v[c2], 11); \
	v[b3] = ROR2( v[b3] ^ v[c3], 11); \
}

#define GS4(a,b,c,d,e,f,a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3){\
	v[ a]+= (m[ e] ^ z[ f]) + v[ b];	v[a1]+= (m[e1] ^ z[f1]) + v[b1];	v[a2]+= (m[e2] ^ z[f2]) + v[b2];	v[a3]+= (m[e3] ^ z[f3]) + v[b3];\
	v[ d] = SWAPDWORDS2(v[ d] ^ v[ a]);	v[d1] = SWAPDWORDS2(v[d1] ^ v[a1]);	v[d2] = SWAPDWORDS2(v[d2] ^ v[a2]);	v[d3] = SWAPDWORDS2(v[d3] ^ v[a3]);\
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];				v[c3]+= v[d3];\
	v[ b] = ROR2(v[b] ^ v[c], 25);		v[b1] = ROR2(v[b1] ^ v[c1], 25);	v[b2] = ROR2(v[b2] ^ v[c2], 25);	v[b3] = ROR2(v[b3] ^ v[c3], 25); \
	v[ a]+= (m[ f] ^ z[ e]) + v[ b];	v[a1]+= (m[f1] ^ z[e1]) + v[b1];	v[a2]+= (m[f2] ^ z[e2]) + v[b2];	v[a3]+= (m[f3] ^ z[e3]) + v[b3];\
	v[ d] = ROR16(v[d] ^ v[a]);		v[d1] = ROR16(v[d1] ^ v[a1]);		v[d2] = ROR16(v[d2] ^ v[a2]);		v[d3] = ROR16(v[d3] ^ v[a3]);\
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];				v[c3]+= v[d3];\
	v[ b] = ROR2(v[b] ^ v[c], 11);		v[b1] = ROR2(v[b1] ^ v[c1], 11);	v[b2] = ROR2(v[b2] ^ v[c2], 11);	v[b3] = ROR2(v[b3] ^ v[c3], 11);\
}

#define GSn4(a,b,c,d,e,f,a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3){\
	v[ a] = v[ a] + e + v[ b];		v[a1] = v[a1] + e1 + v[b1];		v[a2] = v[a2] + e2 + v[b2];		v[a3] = v[a3] + e3 + v[b3];\
	v[ d] = SWAPDWORDS2(v[ d] ^ v[ a]);	v[d1] = SWAPDWORDS2(v[d1] ^ v[a1]);	v[d2] = SWAPDWORDS2(v[d2] ^ v[a2]);	v[d3] = SWAPDWORDS2(v[d3] ^ v[a3]);\
	v[ c] = v[ c] + v[ d];			v[c1] = v[c1] + v[d1];			v[c2] = v[c2] + v[d2];			v[c3] = v[c3] + v[d3];\
	v[ b] = ROR2(v[b] ^ v[c],25);		v[b1] = ROR2(v[b1] ^ v[c1],25);		v[b2] = ROR2(v[b2] ^ v[c2],25);		v[b3] = ROR2(v[b3] ^ v[c3],25); \
	v[ a] = v[ a] + f + v[ b];		v[a1] = v[a1] + f1 + v[b1];		v[a2] = v[a2] + f2 + v[b2];		v[a3] = v[a3] + f3 + v[b3];\
	v[ d] = ROR16(v[d] ^ v[a]);		v[d1] = ROR16(v[d1] ^ v[a1]);		v[d2] = ROR16(v[d2] ^ v[a2]);		v[d3] = ROR16(v[d3] ^ v[a3]);\
	v[ c] = v[ c] + v[ d];			v[c1] = v[c1] + v[d1];			v[c2] = v[c2] + v[d2];			v[c3] = v[c3] + v[d3];\
	v[ b] = ROR2(v[b] ^ v[c],11);		v[b1] = ROR2(v[b1] ^ v[c1],11);		v[b2] = ROR2(v[b2] ^ v[c2],11);		v[b3] = ROR2(v[b3] ^ v[c3],11);\
}

#define GShost(a,b,c,d,e,f) { \
	v[a] += (m[e] ^ z[f]) + v[b]; \
	v[d] = ROTR64(v[d] ^ v[a],32); \
	v[c] += v[d]; \
	v[b] = ROTR64( v[b] ^ v[c], 25); \
	v[a] += (m[f] ^ z[e]) + v[b]; \
	v[d] = ROTR64( v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR64( v[b] ^ v[c], 11); \
}

__global__
__launch_bounds__(192, 1)
void quark_blake512_gpu_hash_64(uint32_t threads, const uint32_t *const __restrict__ g_nonceVector, uint2* g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){
		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];

		uint2 msg[16];

		uint2x4 *phash = (uint2x4*)&g_hash[hashPosition<<3];
		uint2x4 *outpt = (uint2x4*)msg;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		uint2 m[16];
		m[ 0] = cuda_swab64_U2(msg[0]);
		m[ 1] = cuda_swab64_U2(msg[1]);
		m[ 2] = cuda_swab64_U2(msg[2]);
		m[ 3] = cuda_swab64_U2(msg[3]);
		m[ 4] = cuda_swab64_U2(msg[4]);
		m[ 5] = cuda_swab64_U2(msg[5]);
		m[ 6] = cuda_swab64_U2(msg[6]);
		m[ 7] = cuda_swab64_U2(msg[7]);
		m[ 8] = make_uint2(0,0x80000000);
		m[ 9] = make_uint2(0,0);
		m[10] = make_uint2(0,0);
		m[11] = make_uint2(0,0);
		m[12] = make_uint2(0,0);
		m[13] = make_uint2(1,0);
		m[14] = make_uint2(0,0);
		m[15] = make_uint2(0x200,0);

		uint2 v[16] = {
			h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
			z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]
		};
		v[12].x ^= 512U;
		v[13].x ^= 512U;

		GS4(0, 4, 8,12, 0, 1,		1, 5, 9,13, 2, 3,		2, 6,10,14, 4, 5,		3, 7,11,15, 6, 7);
		GS4(0, 5,10,15, 8, 9,		1, 6,11,12,10,11,		2, 7, 8,13,12,13,		3, 4, 9,14,14,15);

		GS4(0, 4, 8, 12, 14, 10,	1, 5, 9, 13, 4, 8,		2, 6, 10, 14, 9, 15,		3, 7, 11, 15, 13, 6);
		GS4(0, 5, 10, 15, 1, 12,	1, 6, 11, 12, 0, 2,		2, 7, 8, 13, 11, 7,		3, 4, 9, 14, 5, 3);

		GS4(0, 4, 8, 12, 11, 8,		1, 5, 9, 13, 12, 0,		2, 6, 10, 14, 5, 2,		3, 7, 11, 15, 15, 13);
		GS4(0, 5, 10, 15, 10, 14,	1, 6, 11, 12, 3, 6,		2, 7, 8, 13, 7, 1,		3, 4, 9, 14, 9, 4);

		GS4(0, 4, 8, 12, 7, 9,		1, 5, 9, 13, 3, 1,		2, 6, 10, 14, 13, 12,		3, 7, 11, 15, 11, 14);
		GS4(0, 5, 10, 15, 2, 6,		1, 6, 11, 12, 5, 10,		2, 7, 8, 13, 4, 0,		3, 4, 9, 14, 15, 8);

		GS4(0, 4, 8, 12, 9, 0,		1, 5, 9, 13, 5, 7,		2, 6, 10, 14, 2, 4,		3, 7, 11, 15, 10, 15);
		GS4(0, 5, 10, 15, 14, 1,	1, 6, 11, 12, 11, 12,		2, 7, 8, 13, 6, 8,		3, 4, 9, 14, 3, 13);

		GS4(0, 4, 8, 12, 2, 12,		1, 5, 9, 13, 6, 10,		2, 6, 10, 14, 0, 11,		3, 7, 11, 15, 8, 3);
		GS4(0, 5, 10, 15, 4, 13,	1, 6, 11, 12, 7, 5,		2, 7, 8, 13, 15, 14,		3, 4, 9, 14, 1, 9);

		GS4(0, 4, 8, 12, 12, 5,		1, 5, 9, 13, 1, 15,		2, 6, 10, 14, 14, 13,		3, 7, 11, 15, 4, 10);
		GS4(0, 5, 10, 15, 0, 7,		1, 6, 11, 12, 6, 3,		2, 7, 8, 13, 9, 2,		3, 4, 9, 14, 8, 11);

		GS4(0, 4, 8, 12, 13, 11,	1, 5, 9, 13, 7, 14,		2, 6, 10, 14, 12, 1,		3, 7, 11, 15, 3, 9);
		GS4(0, 5, 10, 15, 5, 0,		1, 6, 11, 12, 15, 4,		2, 7, 8, 13, 8, 6,		3, 4, 9, 14, 2, 10);

		GS4(0, 4, 8, 12, 6, 15,		1, 5, 9, 13, 14, 9,		2, 6, 10, 14, 11, 3,		3, 7, 11, 15, 0, 8);
		GS4(0, 5, 10, 15, 12, 2,	1, 6, 11, 12, 13, 7,		2, 7, 8, 13, 1, 4,		3, 4, 9, 14, 10, 5);

		GS4(0, 4, 8, 12, 10, 2,		1, 5, 9, 13, 8, 4,		2, 6, 10, 14, 7, 6,		3, 7, 11, 15, 1, 5);
		GS4(0, 5, 10, 15,15,11,		1, 6, 11, 12, 9, 14,		2, 7, 8, 13, 3, 12,		3, 4, 9, 14, 13, 0);

//		#if __CUDA_ARCH__ == 500

		GS4(0, 4, 8,12, 0, 1,		1, 5, 9,13, 2, 3,		2, 6,10,14, 4, 5,		3, 7,11,15, 6, 7);
		GS4(0, 5,10,15, 8, 9,		1, 6,11,12,10,11,		2, 7, 8,13,12,13,		3, 4, 9,14,14,15);

		GS4(0, 4, 8, 12, 14, 10,	1, 5, 9, 13, 4, 8,		2, 6, 10, 14, 9, 15,		3, 7, 11, 15, 13, 6);
		GS4(0, 5, 10, 15, 1, 12,	1, 6, 11, 12, 0, 2,		2, 7, 8, 13, 11, 7,		3, 4, 9, 14, 5, 3);

		GS4(0, 4, 8, 12, 11, 8,		1, 5, 9, 13, 12, 0,		2, 6, 10, 14, 5, 2,		3, 7, 11, 15, 15, 13);
		GS4(0, 5, 10, 15, 10, 14,	1, 6, 11, 12, 3, 6,		2, 7, 8, 13, 7, 1,		3, 4, 9, 14, 9, 4);

		GS4(0, 4, 8, 12, 7, 9,		1, 5, 9, 13, 3, 1,		2, 6, 10, 14, 13, 12,		3, 7, 11, 15, 11, 14);
		GS4(0, 5, 10, 15, 2, 6,		1, 6, 11, 12, 5, 10,		2, 7, 8, 13, 4, 0,		3, 4, 9, 14, 15, 8);

		GS4(0, 4, 8, 12, 9, 0,		1, 5, 9, 13, 5, 7,		2, 6, 10, 14, 2, 4,		3, 7, 11, 15, 10, 15);
		GS4(0, 5, 10, 15, 14, 1,	1, 6, 11, 12, 11, 12,		2, 7, 8, 13, 6, 8,		3, 4, 9, 14, 3, 13);

		GS4(0, 4, 8, 12, 2, 12,		1, 5, 9, 13, 6, 10,		2, 6, 10, 14, 0, 11,		3, 7, 11, 15, 8, 3);
		GS4(0, 5, 10, 15, 4, 13,	1, 6, 11, 12, 7, 5,		2, 7, 8, 13, 15, 14,		3, 4, 9, 14, 1, 9);

//		#else*/
/*
		for (int i = 0; i < 6; i++)
		{
			G4(0,	0, 4, 8,12,	1, 5, 9,13,	2, 6,10,14,	3, 7,11,15);
			G4(8,	0, 5,10,15,	1, 6,11,12,	2, 7, 8,13,	3, 4, 9,14);
		}
*/
//		#endif
		v[0] = cuda_swab64_U2(xor3x(v[0],h[0],v[ 8]));
		v[1] = cuda_swab64_U2(xor3x(v[1],h[1],v[ 9]));
		v[2] = cuda_swab64_U2(xor3x(v[2],h[2],v[10]));
		v[3] = cuda_swab64_U2(xor3x(v[3],h[3],v[11]));
		v[4] = cuda_swab64_U2(xor3x(v[4],h[4],v[12]));
		v[5] = cuda_swab64_U2(xor3x(v[5],h[5],v[13]));
		v[6] = cuda_swab64_U2(xor3x(v[6],h[6],v[14]));
		v[7] = cuda_swab64_U2(xor3x(v[7],h[7],v[15]));

/*		uint2* outHash = &g_hash[hashPosition<<3];
		#pragma unroll 8
		for(uint32_t i=0;i<8;i++){
			outHash[i] = v[i];
		}*/
		phash[0] = *(uint2x4*)&v[ 0];
		phash[1] = *(uint2x4*)&v[ 4];
	}
}

__global__ __launch_bounds__(TPB80,2)
void quark_blake512_gpu_hash_80(const uint32_t threads,const uint32_t startNounce, uint2x4 *const __restrict__ g_hash){
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint2 v[16];
	uint2 m[10];
	uint2 xors[16];

	const uint2 h[8] = {
		{0xf3bcc908,0x6a09e667}, {0x84caa73b,0xbb67ae85},{0xfe94f82b,0x3c6ef372}, {0x5f1d36f1,0xa54ff53a},
		{0xade682d1,0x510e527f}, {0x2b3e6c1f,0x9b05688c},{0xfb41bd6b,0x1f83d9ab}, {0x137e2179,0x5be0cd19}
	};
	const uint2 z[16] = {
		{0x85a308d3,0x243f6a88},{0x03707344,0x13198a2e},{0x299f31d0,0xa4093822},{0xec4e6c89,0x082efa98},
		{0x38d01377,0x452821e6},{0x34e90c6c,0xbe5466cf},{0xc97c50dd,0xc0ac29b7},{0xb5470917,0x3f84d5b5},
		{0x8979fb1b,0x9216d5d9},{0x98dfb5ac,0xd1310ba6},{0xd01adfb7,0x2ffd72db},{0x6a267e96,0xb8e1afed},
		{0xf12c7f99,0xba7c9045},{0xb3916cf7,0x24a19947},{0x858efc16,0x0801f2e2},{0x71574e69,0x636920d8}
	};
	const uint32_t m150 = 0x280 ^ z[ 9].x;//make_uint2(0x280,0) ^ z[ 9];//2
	const uint32_t m151 = 0x280 ^ z[13].x;//2
	const uint32_t m152 = 0x280 ^ z[ 8].x;//2
	const uint32_t m153 = 0x280 ^ z[10].x;//2
	const uint32_t m154 = 0x280 ^ z[14].x;//3
	const uint32_t m155 = 0x280 ^ z[ 1].x;//1
	const uint32_t m156 = 0x280 ^ z[ 4].x;//1
	const uint32_t m157 = 0x280 ^ z[ 6].x;//1
	const uint32_t m158 = 0x280 ^ z[11].x;//1

	const uint32_t m130 = 0x01 ^ z[ 6].x;//2
	const uint32_t m131 = 0x01 ^ z[15].x;//2
	const uint32_t m132 = 0x01 ^ z[12].x;//3
	const uint32_t m133 = 0x01 ^ z[ 3].x;//2
	const uint32_t m134 = 0x01 ^ z[ 4].x;//2
	const uint32_t m135 = 0x01 ^ z[14].x;//1
	const uint32_t m136 = 0x01 ^ z[11].x;//1
	const uint32_t m137 = 0x01 ^ z[ 7].x;//1
	const uint32_t m138 = 0x01 ^ z[ 0].x;//1

	const uint32_t m100 = 0x80000000 ^ z[14].y;//4
	const uint32_t m101 = 0x80000000 ^ z[ 5].y;//3
	const uint32_t m102 = 0x80000000 ^ z[15].y;//2
	const uint32_t m103 = 0x80000000 ^ z[ 6].y;//2
	const uint32_t m104 = 0x80000000 ^ z[ 4].y;//1
	const uint32_t m105 = 0x80000000 ^ z[ 2].y;//2
	const uint32_t m106 = 0x80000000 ^ z[11].y;//2

	if (thread < threads){

		int i=0;
		
		#pragma unroll 10
		for (int i=0; i < 10; ++i)
			m[i] = c_m[i];


		m[ 9].x = startNounce + thread;

		#pragma unroll 16
		for(int i=0; i < 16; i++)
			v[i] = c_v[i];

//		GSn( 0, 5,10,15, 8, 9);
		v[ 0]+= (m[ 9] ^ z[ 8]);
		v[15] = ROR16(v[15] ^ v[ 0]);
		v[10]+= v[15];
		v[ 5] = ROR2(v[ 5] ^ v[10], 11);

		xors[ 0] = z[10];			xors[ 1] = c_x[i++];			xors[ 2] = m[ 9]^z[15];			xors[ 3] = make_uint2(m130,z[ 6].y);
		xors[ 4] = make_uint2(z[14].x,m100);	xors[ 5] = c_x[i++];			xors[ 6] = make_uint2(m150,z[9].y);	xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = z[ 7];			xors[11] = c_x[i++];
		xors[12] = z[ 1];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//2:{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 }
		xors[ 0] = z[ 8];			xors[ 1] = z[ 0];			xors[ 2] = c_x[i++];			xors[ 3] = make_uint2(m151,z[13].y);
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = make_uint2(m131,z[15].y);
		
		xors[ 8] = make_uint2(z[14].x,m100);	xors[ 9] = c_x[i++];			xors[10] = c_x[i++];			xors[11] = m[ 9]^z[ 4];
		xors[12] = z[10];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//3:{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 }
		xors[ 0] = c_x[i++];			xors[ 1] = c_x[i++];			xors[ 2] = make_uint2(m132,z[12].y);	xors[ 3] = z[14];
		xors[ 4] = m[ 9]^z[ 7];			xors[ 5] = c_x[i++];			xors[ 6] = z[13];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = c_x[i++];			xors[11] = make_uint2(m152,z[ 8].y);
		xors[12] = c_x[i++];			xors[13] = make_uint2(z[ 5].x,m101);	xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		
		//4:{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 }
		xors[ 0] = m[ 9]^z[ 0];			xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = make_uint2(z[15].x,m102);
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = make_uint2(m153,z[10].y);
		
		xors[ 8] = z[ 1];			xors[ 9] = z[12];			xors[10] = c_x[i++];			xors[11] = c_x[i++];
		xors[12] = c_x[i++];			xors[13] = z[11];			xors[14] = c_x[i++];			xors[15] = make_uint2(m133,z[ 3].y);
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		
		//5:{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
		xors[ 0] = c_x[i++];			xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = c_x[i++];
		xors[ 4] = z[ 2];			xors[ 5] = make_uint2(z[ 6].x,m103);	xors[ 6] = z[ 0];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = make_uint2(m154,z[14].y);	xors[11] = c_x[i++];
		xors[12] = make_uint2(m134,z[ 4].y);	xors[13] = c_x[i++];			xors[14] = z[15];			xors[15] = m[ 9]^z[ 1];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		
		//6:{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 }
		xors[ 0] = z[ 5];			xors[ 1] = c_x[i++];			xors[ 2] = z[13];			xors[ 3] = c_x[i++];
		xors[ 4] = c_x[i++];			xors[ 5] = make_uint2(m155,z[ 1].y);	xors[ 6] = make_uint2(m135,z[14].y);	xors[ 7] = make_uint2(z[ 4].x,m104);
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = m[ 9]^z[ 2];			xors[11] = c_x[i++];
		xors[12] = c_x[i++];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = z[ 8];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//7:{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 }
		xors[ 0] = make_uint2(m136,z[11].y);	xors[ 1] = c_x[i++];			xors[ 2] = z[ 1];			xors[ 3] = c_x[i++];
		xors[ 4] = z[13];			xors[ 5] = z[ 7];			xors[ 6] = c_x[i++];			xors[ 7] = m[ 9]^z[ 3];
		
		xors[ 8] = c_x[i++];			xors[ 9] = make_uint2(m156,z[ 4].y);	xors[10] = c_x[i++];			xors[11] = c_x[i++];
		xors[12] = c_x[i++];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = make_uint2(z[ 2].x,m105);
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		
		//8:{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 }
		xors[ 0] = c_x[i++];			xors[ 1] = z[ 9];			xors[ 2] = z[ 3];			xors[ 3] = c_x[i++];
		xors[ 4] = make_uint2(m157,z[ 6].y);	xors[ 5] = m[ 9] ^ z[14];		xors[ 6] = c_x[i++];			xors[ 7] = c_x[i++];

		xors[ 8] = z[ 2];			xors[ 9] = make_uint2(m137,z[ 7].y);	xors[10] = c_x[i++];			xors[11] = make_uint2(z[ 5].x,m101);
		xors[12] = c_x[i++];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];

		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//9:{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 }
		xors[ 0] = make_uint2(z[ 2].x,m105);	xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = c_x[i++];
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = c_x[i++];
		
		xors[ 8] = make_uint2(m158,z[11].y);	xors[ 9] = m[ 9]^z[14];			xors[10] = c_x[i++];			xors[11] = make_uint2(m138,z[ 0].y);
		xors[12] = z[15];			xors[13] = z[ 9];			xors[14] = z[ 3];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		//10:{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }
		xors[ 0] = c_x[i++];		xors[ 1] = c_x[i++];				xors[ 2] = c_x[i++];			xors[ 3] = c_x[i++];
		xors[ 4] = c_x[i++];		xors[ 5] = c_x[i++];				xors[ 6] = c_x[i++];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];		xors[ 9] = make_uint2(z[11].x,m106);		xors[10] = z[13];			xors[11] = z[15];
		xors[12] = m[ 9]^z[ 8];		xors[13] = z[10];				xors[14] = make_uint2(m132,z[12].y);	xors[15] = make_uint2(m154,z[14].y);
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
//------------------
		i=0;
		//11:{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
		xors[ 0] = z[10];			xors[ 1] = c_x[i++];			xors[ 2] = m[ 9]^z[15];			xors[ 3] = make_uint2(m130,z[ 6].y);
		xors[ 4] = make_uint2(z[14].x,m100);	xors[ 5] = c_x[i++];			xors[ 6] = make_uint2(m150,z[9].y);	xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = z[ 7];			xors[11] = c_x[i++];
		xors[12] = z[ 1];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//12:{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 }
		xors[ 0] = z[ 8];			xors[ 1] = z[ 0];			xors[ 2] = c_x[i++];			xors[ 3] = make_uint2(m151,z[13].y);
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = make_uint2(m131,z[15].y);
		
		xors[ 8] = make_uint2(z[14].x,m100);	xors[ 9] = c_x[i++];			xors[10] = c_x[i++];			xors[11] = m[ 9]^z[ 4];
		xors[12] = z[10];			xors[13] = c_x[i++];			xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//13:{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 }
		xors[ 0] = c_x[i++];			xors[ 1] = c_x[i++];			xors[ 2] = make_uint2(m132,z[12].y);	xors[ 3] = z[14];
		xors[ 4] = m[ 9]^z[ 7];			xors[ 5] = c_x[i++];			xors[ 6] = z[13];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = c_x[i++];			xors[11] = make_uint2(m152,z[ 8].y);
		xors[12] = c_x[i++];			xors[13] = make_uint2(z[ 5].x,m101);	xors[14] = c_x[i++];			xors[15] = c_x[i++];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		//14:{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 }
		xors[ 0] = m[ 9]^z[ 0];			xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = make_uint2(z[15].x,m102);
		xors[ 4] = c_x[i++];			xors[ 5] = c_x[i++];			xors[ 6] = c_x[i++];			xors[ 7] = make_uint2(m153,z[10].y);
		
		xors[ 8] = z[ 1];			xors[ 9] = z[12];			xors[10] = c_x[i++];			xors[11] = c_x[i++];
		xors[12] = c_x[i++];			xors[13] = z[11];			xors[14] = c_x[i++];			xors[15] = make_uint2(m133,z[ 3].y);
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);
		//15:{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
		xors[ 0] = c_x[i++];			xors[ 1] = c_x[i++];			xors[ 2] = c_x[i++];			xors[ 3] = c_x[i++];
		xors[ 4] = z[ 2];			xors[ 5] = make_uint2(z[ 6].x,m103);	xors[ 6] = z[ 0];			xors[ 7] = c_x[i++];
		
		xors[ 8] = c_x[i++];			xors[ 9] = c_x[i++];			xors[10] = make_uint2(m154,z[14].y);	xors[11] = c_x[i++];
		xors[12] = make_uint2(m134,z[ 4].y);	xors[13] = c_x[i++];			xors[14] = z[15];			xors[15] = m[ 9]^z[ 1];
		
		GSn4(0, 4, 8,12, xors[ 0],xors[ 4], 1, 5, 9,13, xors[ 1],xors[ 5], 2, 6,10,14, xors[ 2],xors[ 6], 3, 7,11,15, xors[ 3],xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8],xors[12], 1, 6,11,12, xors[ 9],xors[13], 2, 7, 8,13, xors[10],xors[14], 3, 4, 9,14, xors[11],xors[15]);

		v[0] = cuda_swab64_U2(xor3x(v[0],h[0],v[ 8]));
		v[1] = cuda_swab64_U2(xor3x(v[1],h[1],v[ 9]));
		v[2] = cuda_swab64_U2(xor3x(v[2],h[2],v[10]));
		v[3] = cuda_swab64_U2(xor3x(v[3],h[3],v[11]));
		v[4] = cuda_swab64_U2(xor3x(v[4],h[4],v[12]));
		v[5] = cuda_swab64_U2(xor3x(v[5],h[5],v[13]));
		v[6] = cuda_swab64_U2(xor3x(v[6],h[6],v[14]));
		v[7] = cuda_swab64_U2(xor3x(v[7],h[7],v[15]));
		
		uint2x4* outpt = &g_hash[thread<<1];
		outpt[0] = *(uint2x4*)&v[0];
		outpt[1] = *(uint2x4*)&v[4];
	}
}

__host__
void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_outputHash){
	uint32_t tpb = TPB52_64;
	int dev_id = device_map[thr_id];
	
	if (device_sm[dev_id] <= 500) tpb = TPB50_64;
	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);
	quark_blake512_gpu_hash_64<<<grid, block>>>(threads, d_nonceVector, (uint2*)d_outputHash);
}

__host__
void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash){
	dim3 grid((threads + TPB80-1)/TPB80);
	dim3 block(TPB80);

	quark_blake512_gpu_hash_80<<<grid, block>>>(threads, startNounce, (uint2x4*)d_outputHash);
}

// ---------------------------- END CUDA quark_blake512 functions ------------------------------------

// ----------------------------- Host midstate for 80-bytes input ------------------------------------
__host__
void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *endiandata){
	uint64_t m[16],v[16],xors[128];
	memcpy(m, endiandata, 80);
	m[10] = 0x8000000000000000ull;
	m[11] = 0;
	m[12] = 0;
	m[13] = 0x01;
	m[14] = 0;
	m[15] = 0x280;

	for(int i=0;i<10;i++){
		m[ i] = cuda_swab64(m[ i]);
	}
	
	uint64_t h[8] = {
		0x6a09e667f3bcc908ULL,	0xbb67ae8584caa73bULL,	0x3c6ef372fe94f82bULL,	0xa54ff53a5f1d36f1ULL,
		0x510e527fade682d1ULL,	0x9b05688c2b3e6c1fULL,	0x1f83d9abfb41bd6bULL,	0x5be0cd19137e2179ULL
	};

	const uint64_t z[16] = {
		0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,	0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
		0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,	0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
		0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL,	0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
		0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL,	0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
	};

	for(int i=0;i<8;i++){
		v[ i] = h[ i];
	}
	v[ 8] = z[0];
	v[ 9] = z[1];
	v[10] = z[2];
	v[11] = z[3];
	v[12] = z[4] ^ 640;
	v[13] = z[5] ^ 640;
	v[14] = z[6];
	v[15] = z[7];
	
	/* column step */
	GShost( 0, 4, 8,12, 0, 1);
	GShost( 1, 5, 9,13, 2, 3);
	GShost( 2, 6,10,14, 4, 5);
	GShost( 3, 7,11,15, 6, 7);

	GShost( 1, 6,11,12,10,11);
	GShost( 2, 7, 8,13,12,13);
	GShost( 3, 4, 9,14,14,15);
/*
	v[a] += (m[e] ^ z[f]) + v[b]; \
	v[d] = ROTR64(v[d] ^ v[a],32); \
	v[c] += v[d]; \
	v[b] = ROTR64( v[b] ^ v[c], 25); \
	v[a] += (m[f] ^ z[e]) + v[b]; \
	v[d] = ROTR64( v[d] ^ v[a], 16); \
	v[c] += v[d];
	v[b] = ROTR64( v[b] ^ v[c], 11);
*/

	v[ 0]+= (m[ 8] ^ z[ 9]) + v[ 5];
	v[15] = ROTR64(v[15]^v[ 0],32);
	v[10]+= v[15];
	v[ 5] = ROTR64(v[ 5] ^ v[10], 25);

	v[ 0]+= v[ 5];
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_m, m, sizeof(m), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_v, v, sizeof(m), 0, cudaMemcpyHostToDevice));

	int i=0;
	
	xors[i++] = m[ 4]^z[ 8];
	xors[i++] = m[ 8]^z[ 4];	
	xors[i++] = m[ 6]^z[13];
	xors[i++] = m[ 1]^z[12];	
	xors[i++] = m[ 0]^z[ 2];
	xors[i++] = m[ 5]^z[ 3];
	xors[i++] = m[ 2]^z[ 0];	
	xors[i++] = m[ 7]^z[11];	
	xors[i++] = m[ 3]^z[ 5];
//2
	xors[i++] = m[ 5]^z[ 2];
	xors[i++] = m[ 8]^z[11];	
	xors[i++] = m[ 0]^z[12];
	xors[i++] = m[ 2]^z[ 5];
	xors[i++] = m[ 3]^z[ 6];	
	xors[i++] = m[ 7]^z[ 1];	
	xors[i++] = m[ 6]^z[ 3];	
	xors[i++] = m[ 1]^z[ 7];	
	xors[i++] = m[ 4]^z[ 9];
//3
	xors[i++] = m[ 7]^z[ 9];
	xors[i++] = m[ 3]^z[ 1];
	xors[i++] = m[ 1]^z[ 3];
	xors[i++] = m[14]^z[11];
	xors[i++] = m[ 2]^z[ 6];
	xors[i++] = m[ 5]^z[10];
	xors[i++] = m[ 4]^z[ 0];
	xors[i++] = m[ 6]^z[ 2];
	xors[i++] = m[ 0]^z[ 4];
	xors[i++] = m[ 8]^z[15];
//4
	xors[i++] = m[ 5]^z[ 7];
	xors[i++] = m[ 2]^z[ 4];
	xors[i++] = m[ 0]^z[ 9];
	xors[i++] = m[ 7]^z[ 5];
	xors[i++] = m[ 4]^z[ 2];
	xors[i++] = m[ 6]^z[ 8];
	xors[i++] = m[ 3]^z[13];
	xors[i++] = m[ 1]^z[14];
	xors[i++] = m[ 8]^z[ 6];
//5
	xors[i++] = m[ 2]^z[12];
	xors[i++] = m[ 6]^z[10];
	xors[i++] = m[ 0]^z[11];
	xors[i++] = m[ 8]^z[ 3];
	xors[i++] = m[ 3]^z[ 8];
	xors[i++] = m[ 4]^z[13];
	xors[i++] = m[ 7]^z[ 5];	
	xors[i++] = m[ 1]^z[ 9];
	xors[i++] = m[ 5]^z[ 7];
//6
	xors[i++] = m[ 1]^z[15];
	xors[i++] = m[ 4]^z[10];
	xors[i++] = m[ 5]^z[12];
	xors[i++] = m[ 0]^z[ 7];
	xors[i++] = m[ 6]^z[ 3];
	xors[i++] = m[ 8]^z[11];
	xors[i++] = m[ 7]^z[ 0];
	xors[i++] = m[ 3]^z[ 6];
	xors[i++] = m[ 2]^z[ 9];
//7
	xors[i++] = m[ 7]^z[14];
	xors[i++] = m[ 3]^z[ 9];
	xors[i++] = m[ 1]^z[12];
	xors[i++] = m[ 5]^z[ 0];
	xors[i++] = m[ 8]^z[ 6];
	xors[i++] = m[ 2]^z[10];
	xors[i++] = m[ 0]^z[ 5];
	xors[i++] = m[ 4]^z[15];
	xors[i++] = m[ 6]^z[ 8];
//8
	xors[i++] = m[ 6]^z[15];
	xors[i++] = m[ 0]^z[ 8];
	xors[i++] = m[ 3]^z[11];
	xors[i++] = m[ 8]^z[ 0];
	xors[i++] = m[ 1]^z[ 4];
	xors[i++] = m[ 2]^z[12];
	xors[i++] = m[ 7]^z[13];
	xors[i++] = m[ 4]^z[ 1];
	xors[i++] = m[ 5]^z[10];
//9
	xors[i++] = m[ 8]^z[ 4];
	xors[i++] = m[ 7]^z[ 6];
	xors[i++] = m[ 1]^z[ 5];
	xors[i++] = m[ 2]^z[10];
	xors[i++] = m[ 4]^z[ 8];
	xors[i++] = m[ 6]^z[ 7];
	xors[i++] = m[ 5]^z[ 1];
	xors[i++] = m[ 3]^z[12];
	xors[i++] = m[ 0]^z[13];
//10
	xors[i++] = m[ 0]^z[ 1];
	xors[i++] = m[ 2]^z[ 3];
	xors[i++] = m[ 4]^z[ 5];
	xors[i++] = m[ 6]^z[ 7];
	xors[i++] = m[ 1]^z[ 0];
	xors[i++] = m[ 3]^z[ 2];
	xors[i++] = m[ 5]^z[ 4];
	xors[i++] = m[ 7]^z[ 6];
	xors[i++] = m[ 8]^z[ 9];

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x,xors, i*sizeof(uint2), 0, cudaMemcpyHostToDevice));
}
