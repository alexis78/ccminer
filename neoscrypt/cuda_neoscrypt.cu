// originally from djm34 (https://github.com/djm34/ccminer-sp-neoscrypt/)
// forked from KlausT repository

// 1. Removed compute3.0 and lower compute capability
// 2. Replaced 64bit shifts with byte permutations
// 3. Removed pointer fetching from constant memory
// 4. Better loop unrolling factors for cp5.0/5.2
// 6. More precomputations
// 7. Increased default intensity (?)
// 8. Restored Second nonce buffer
// 9. Use video SIMD instruction for cumulative sum of bufidx
// Provos Alexis - 2016

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h" 

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

#ifdef _MSC_VER
#define THREAD __declspec(thread)
#else
#define THREAD __thread
#endif

static cudaStream_t stream[MAX_GPUS][2];

__constant__ uint32_t key_init[16];
__constant__ uint32_t _ALIGN(16) c_data[64];
__constant__ uint32_t _ALIGN(8) buf_shifts[16];

/// constants ///

static const __constant__  uint8 BLAKE2S_IV_Vec =
{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};


static const  uint8 BLAKE2S_IV_Vechost =
{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint32_t BLAKE2S_SIGMA_host[10][16] =
{
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},	{7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},	{2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
	{6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
};

#define SALSA(a,b,c,d){ \
	b^=ROTL32(a+d,  7);\
	c^=ROTL32(a+b,  9);\
	d^=ROTL32(b+c, 13);\
	a^=ROTL32(c+d, 18);\
}

#define SALSA_CORE(state) { \
	SALSA(state.s0,state.s4,state.s8,state.sc);\
	SALSA(state.s5,state.s9,state.sd,state.s1);\
	SALSA(state.sa,state.se,state.s2,state.s6);\
	SALSA(state.sf,state.s3,state.s7,state.sb);\
	SALSA(state.s0,state.s1,state.s2,state.s3);\
	SALSA(state.s5,state.s6,state.s7,state.s4);\
	SALSA(state.sa,state.sb,state.s8,state.s9);\
	SALSA(state.sf,state.sc,state.sd,state.se);\
} 

#define CHACHA_STEP(a,b,c,d) { \
	a += b; d = __byte_perm(d^a,0,0x1032); \
	c += d; b = ROTL32(b^c, 12); \
	a += b; d = __byte_perm(d^a,0,0x2103); \
	c += d; b = ROTL32(b^c, 7); \
}

#define CHACHA_CORE_PARALLEL(state){\
	CHACHA_STEP(state.lo.s0, state.lo.s4, state.hi.s0, state.hi.s4); \
	CHACHA_STEP(state.lo.s1, state.lo.s5, state.hi.s1, state.hi.s5); \
	CHACHA_STEP(state.lo.s2, state.lo.s6, state.hi.s2, state.hi.s6); \
	CHACHA_STEP(state.lo.s3, state.lo.s7, state.hi.s3, state.hi.s7); \
	CHACHA_STEP(state.lo.s0, state.lo.s5, state.hi.s2, state.hi.s7); \
	CHACHA_STEP(state.lo.s1, state.lo.s6, state.hi.s3, state.hi.s4); \
	CHACHA_STEP(state.lo.s2, state.lo.s7, state.hi.s0, state.hi.s5); \
	CHACHA_STEP(state.lo.s3, state.lo.s4, state.hi.s1, state.hi.s6); \
}

#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA[idx0][idx1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = ROTR32(b^c, 12); \
	idx = BLAKE2S_SIGMA[idx0][idx1+1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = ROTR32(b^c, 7); \
} 

#define BLAKE(a, b, c, d, key1,key2) { \
	a += b + key1; \
	d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = ROTR32(b^c, 12); \
	a += b + key2; \
	d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = ROTR32(b^c, 7); \
} 

#define BLAKE_G_PRE(idx0, idx1, a, b, c, d, key) { \
	a += b + key[idx0]; \
	d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = ROTR32(b^c, 12); \
	a += b + key[idx1]; \
	d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = ROTR32(b^c, 7); \
} 

#define BLAKE_G_PRE0(idx0, idx1, a, b, c, d, key) { \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = ROTR32(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = ROTR32(b^c, 7); \
} 

#define BLAKE_G_PRE1(idx0, idx1, a, b, c, d, key) { \
	a += b + key[idx0]; \
	d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = ROTR32(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = ROTR32(b^c, 7); \
} 

#define BLAKE_G_PRE2(idx0, idx1, a, b, c, d, key) { \
	a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = ROTR32(b^c, 12); \
	a += b + key[idx1]; \
	d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = ROTR32(b^c, 7); \
} 

#define BLAKE_Ghost(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA_host[idx0][idx1]; \
	a += b + key[idx]; \
	d = ROTR32(d^a, 16); \
	c += d; b = ROTR32(b^c, 12); \
	idx = BLAKE2S_SIGMA_host[idx0][idx1+1]; \
	a += b + key[idx]; \
	d = ROTR32(d^a, 8); \
	c += d; b = ROTR32(b^c, 7); \
} 

__device__ __forceinline__
static void Blake2S_v2(uint32_t *out, const uint32_t* __restrict__  inout, const  uint32_t * __restrict__ TheKey){

	uint16 V;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	V.lo = BLAKE2S_IV_Vec;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= 64;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE0(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE0(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE1(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE1(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE2(9, 0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(5, 7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(10, 15, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(14, 1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(11, 12, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(3, 13, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE1(2, 12, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 10, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE1(0, 11, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(8, 3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(4, 13, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(7, 5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(15, 14, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(1, 9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE2(12, 5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(1, 15, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 13, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(4, 10, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(6, 3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(9, 2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(8, 11, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE0(13, 11, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(7, 14, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(12, 1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(3, 9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(5, 0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(15, 4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 10, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE1(6, 15, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE0(14, 9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(11, 3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(0, 8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(12, 2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(13, 7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(1, 4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(10, 5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE2(10, 2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(7, 6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(1, 5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(15, 11, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(9, 14, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(3, 12, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(13, 0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	V.lo = V.lo ^ tmpblock ^ V.hi;

	V.hi = BLAKE2S_IV_Vec;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//#pragma unroll

	//		13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
	//		6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
	//		10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[9], inout[0]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[5], inout[7]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[2], inout[4]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[10], inout[15]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[14], inout[1]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[11], inout[12]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[6], inout[8]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[3], inout[13]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[2], inout[12]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[6], inout[10]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[0], inout[11]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[8], inout[3]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[4], inout[13]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[7], inout[5]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[15], inout[14]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[1], inout[9]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[12], inout[5]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[1], inout[15]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[14], inout[13]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[4], inout[10]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[0], inout[7]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[6], inout[3]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[9], inout[2]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[8], inout[11]);

	//		13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
	//		6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[13], inout[11]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[7], inout[14]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[12], inout[1]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[3], inout[9]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[5], inout[0]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[15], inout[4]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[8], inout[6]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[2], inout[10]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[6], inout[15]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[14], inout[9]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[11], inout[3]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[0], inout[8]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[12], inout[2]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[13], inout[7]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[1], inout[4]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[10], inout[5]);
	//		10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[10], inout[2]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[8], inout[4]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[7], inout[6]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[1], inout[5]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[15], inout[11]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[9], inout[14]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[3], inout[12]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[13], inout[0]);

	((uint8*)out)[0] = V.lo ^ tmpblock ^ V.hi;
}

__device__ __forceinline__
static uint16 salsa_small_scalar_rnd(const uint16 &X)
{
	uint16 state = X;

	for (uint32_t i = 0; i < 10; i++) {
		SALSA_CORE(state);
	}

	return(X + state);
}

__device__ __forceinline__
static uint16 chacha_small_parallel_rnd(const uint16 &X){

	uint16 state = X;

	for (uint32_t i = 0; i < 10; i++) {
		CHACHA_CORE_PARALLEL(state);
	}
	return (X + state);
}

static void Blake2Shost(uint32_t * inout, const uint32_t * inkey){

	uint16 V;
	uint32_t idx;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	V.lo = BLAKE2S_IV_Vechost;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= 64;

	for(int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inkey);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inkey);
	}

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	for(int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	V.lo ^= V.hi ^ tmpblock;

	((uint8*)inout)[0] = V.lo;
}

#define TPB     128

#define TPBchacha152 64
#define TPBchacha150 128

#define TPBchacha252 512
#define TPBchacha250 128

#define TPBsalsa152 512
#define TPBsalsa150 512

#define TPBsalsa252 512
#define TPBsalsa250 128

__global__ __launch_bounds__(TPB, 2)
void neoscrypt_gpu_hash_start(int stratum, uint32_t threads, uint32_t startNonce,uint2x4* Input){

	__shared__ uint32_t s_data[64 * TPB];

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if(thread<threads){
		const uint32_t Nonce = startNonce + thread;
		const uint32_t nonce = (stratum) ? cuda_swab32(Nonce) : Nonce; //freaking morons !!!
		uint32_t data18 = c_data[18];
		uint32_t data20 = c_data[0];
		uint32_t input[16];
		uint32_t key[16] = { 0 };
		uint32_t qbuf, rbuf, bitbuf;

		uint32_t* B = (uint32_t*)&s_data[threadIdx.x<<6];
		#pragma unroll 8
		for(uint32_t i = 0; i<64 ;i+=8){
			*(uint2x4*)&B[i] = *(uint2x4*)&c_data[i];
		}

		B[19] = nonce;
		B[39] = nonce;
		B[59] = nonce;

		//uint32_t values[11];//qbuf,bitbuf,shiftedvalues
		qbuf    = buf_shifts[ 0];
		bitbuf  = buf_shifts[ 1];

		uint32_t temp[9];
		temp[0] = B[(0 + qbuf) & 0x3f] ^ buf_shifts[ 2];
		temp[1] = B[(1 + qbuf) & 0x3f] ^ buf_shifts[ 3];
		temp[2] = B[(2 + qbuf) & 0x3f] ^ buf_shifts[ 4];
		temp[3] = B[(3 + qbuf) & 0x3f] ^ buf_shifts[ 5];
		temp[4] = B[(4 + qbuf) & 0x3f] ^ buf_shifts[ 6];
		temp[5] = B[(5 + qbuf) & 0x3f] ^ buf_shifts[ 7];
		temp[6] = B[(6 + qbuf) & 0x3f] ^ buf_shifts[ 8];
		temp[7] = B[(7 + qbuf) & 0x3f] ^ buf_shifts[ 9];
		temp[8] = B[(8 + qbuf) & 0x3f] ^ buf_shifts[10];

		uint32_t a = c_data[qbuf & 0x3f], b;

		#pragma unroll 8
		for (uint32_t k = 0; k<16; k += 2){
			b = c_data[(qbuf + k + 1) & 0x3f];
			input[ k] = __byte_perm(a,b,bitbuf);
			a = c_data[(qbuf + k + 2) & 0x3f];
			input[k+1] = __byte_perm(b,a,bitbuf);
			key[(k>>1)] = __byte_perm(temp[(k>>1)],temp[(k>>1)+1],bitbuf);			
		}

		if(qbuf < 60){
			const uint32_t noncepos = 19 - qbuf % 20;
			if (noncepos <= 16){
				if (noncepos)
					input[noncepos - 1] = __byte_perm(data18,nonce,bitbuf);
				if (noncepos != 16U)
					input[noncepos] = __byte_perm(nonce,data20,bitbuf);
			}
		}

		Blake2S_v2(input, input, key);

		#pragma unroll 9
		for (uint32_t k = 0; k < 9; k++)
			B[(k + qbuf) & 0x3f] = temp[k];

		#pragma unroll 31
		for (uint32_t i = 1; i < 31; i++)
		{
			uint8_t bufidx = 0;
			#pragma unroll 4
			for (uint32_t x = 0; x < 8; x+=2){
				bufidx+= __vsadu4(input[x],0) + __vsadu4(input[x+1],0);
			}
			qbuf = bufidx >> 2;
			rbuf = bufidx & 3;
			bitbuf = rbuf << 3;

			uint32_t temp[9];

			uint32_t shift = 32 - bitbuf;
			const uint32_t byte_perm_shift = 0x76543210ULL >> (shift>>1);
			const uint32_t byte_perm_bitbf = 0x76543210ULL >> (bitbuf>>1);
			
			shift = (input[7]>>shift);
			temp[8] = B[(8 + qbuf) & 0x3f] ^ shift;
			shift = __byte_perm(input[ 6],input[ 7],byte_perm_shift);
			temp[7] = B[(7 + qbuf) & 0x3f] ^ shift;
			shift = __byte_perm(input[ 5],input[ 6],byte_perm_shift);
			temp[6] = B[(6 + qbuf) & 0x3f] ^ shift;
			shift = __byte_perm(input[ 4],input[ 5],byte_perm_shift);
			temp[5] = B[(5 + qbuf) & 0x3f] ^ shift;
			shift = __byte_perm(input[ 3],input[ 4],byte_perm_shift);
			temp[4] = B[(4 + qbuf) & 0x3f] ^ shift;
			shift = __byte_perm(input[ 2],input[ 3],byte_perm_shift);
			temp[3] = B[(3 + qbuf) & 0x3f] ^ shift;
			shift = __byte_perm(input[ 1],input[ 2],byte_perm_shift);
			temp[2] = B[(2 + qbuf) & 0x3f] ^ shift;
			shift = __byte_perm(input[ 0],input[ 1],byte_perm_shift);
			temp[1] = B[(1 + qbuf) & 0x3f] ^ shift;
			temp[0] = B[(0 + qbuf) & 0x3f] ^ (input[ 0]<<bitbuf);
			
			uint32_t a = c_data[qbuf & 0x3f];

			#pragma unroll 8
			for (int k = 0; k<16; k += 2)
			{
				const uint32_t b = c_data[(qbuf + k + 1) & 0x3f];
				input[k] = __byte_perm(a,b,byte_perm_bitbf);
				a = c_data[(qbuf + k + 2) & 0x3f];
				input[k+1] = __byte_perm(b,a,byte_perm_bitbf);
				key[(k>>1)] = __byte_perm(temp[(k>>1)],temp[(k>>1)+1],byte_perm_bitbf);
			}

			if(qbuf < 60){
				const uint32_t noncepos = 19 - qbuf % 20U;
				if (noncepos <= 16){
					if (noncepos)
						input[noncepos - 1] = __byte_perm(data18,nonce,byte_perm_bitbf);
					if (noncepos != 16U)
						input[noncepos] = __byte_perm(nonce,data20,byte_perm_bitbf);
				}
			}

			Blake2S_v2(input, input, key);

			#pragma unroll 9
			for (int k = 0; k < 9; k++)
				B[(k + qbuf) & 0x3f] = temp[k];
		}

		uint8_t bufidx = 0;
		#pragma unroll 4
		for (uint32_t x = 0; x < 8; x+=2){
			bufidx+= __vsadu4(input[x],0) + __vsadu4(input[x+1],0);
		}
		qbuf = bufidx >> 2;
		rbuf = bufidx & 3;

		uint32_t byte_perm_bitbf = 0x76543210ULL >> (rbuf<<2);

		uint2x4 output[8];
		#pragma unroll
		for (uint32_t i = 0; i<64; i++) {
			const uint32_t a = (qbuf + i) & 0x3f, b = (qbuf + i + 1) & 0x3f;
			((uint32_t*)output)[i] = __byte_perm(B[a],B[b],byte_perm_bitbf);
		}

		output[0] ^= ((uint2x4*)input)[0];
		((uint32_t*)output)[19] ^= nonce;
		((uint32_t*)output)[39] ^= nonce;
		((uint32_t*)output)[59] ^= nonce;

		#pragma unroll 8
		for(uint32_t i=0;i<8;i++){
			Input[i*threads+thread] = output[i] ^ ((uint2x4*)c_data)[i];
		}
	}
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPBchacha152, 6)
#else
__launch_bounds__(TPBchacha150, 6)
#endif
void neoscrypt_gpu_hash_chacha1_stream1(const uint32_t threads,const uint2x4* Input, uint2x4 *const __restrict__ W, uint2x4 *const __restrict__ Tr){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if(thread<threads){

		uint2x4 X[8];
		uint16* XV = (uint16*)X;
		
		#pragma unroll 8
		for(int i = 0; i<8; i++)
			X[i] = __ldg4(&Input[i*threads+thread]);

		#pragma nounroll
		for(uint32_t i = 0; i < 128; ++i){
			#if __CUDA_ARCH__ > 500
				#pragma unroll 8
				for(uint32_t j = 0; j<8; j++)
					W[(thread<<10) + (i<<3) + j] = X[j];
			#else
				#pragma unroll 8
				for(uint32_t j = 0; j<8; j++)
					W[(i*8+j)*threads+thread] = X[j];
			#endif

			const uint16 temp = XV[2];
			XV[0] = chacha_small_parallel_rnd(XV[0] ^ XV[3]);
			XV[2] = chacha_small_parallel_rnd(XV[1] ^ XV[0]);
			XV[1] = chacha_small_parallel_rnd(XV[2] ^ temp);
			XV[3] = chacha_small_parallel_rnd(XV[3] ^ XV[1]);

		}

		#pragma unroll 8
		for(uint32_t i = 0; i<8; i++)
			Tr[i*threads+thread] = X[i];
	}
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPBchacha252, 1)
#else
__launch_bounds__(TPBchacha250, 1)
#endif
void neoscrypt_gpu_hash_chacha2_stream1(const uint32_t threads, const uint2x4 *const __restrict__ W, uint2x4 *const __restrict__ Tr){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint2x4 X[8];
	uint16* XV = (uint16*)X;

	if(thread<threads){
	
		#pragma unroll
		for(int i = 0; i<8; i++)
			X[i] = __ldg4(&Tr[i*threads+thread]);

		#pragma unroll 128
		for(int t = 0; t < 128; t++){

			const uint32_t idx = (X[6].x.x & 0x7F) << 3;

			#if __CUDA_ARCH__ > 500
				#pragma unroll 8
				for(uint32_t j = 0; j<8; j++)
					X[j] ^= __ldg4(&W[(thread<<10) + idx + j]);
			#else
				#pragma nounroll
				for(uint32_t j = 0; j<8; j++)
					X[j] ^= __ldg4(&W[(idx+j)*threads + thread]);
			#endif

			const uint16 temp = XV[2];
			XV[0] = chacha_small_parallel_rnd(XV[0] ^ XV[3]);
			XV[2] = chacha_small_parallel_rnd(XV[1] ^ XV[0]);
			XV[1] = chacha_small_parallel_rnd(XV[2] ^ temp);
			XV[3] = chacha_small_parallel_rnd(XV[3] ^ XV[1]);
		}

		#pragma unroll 8
		for(uint32_t i = 0; i<8; i++)
			Tr[i*threads+thread] = X[i];
	}
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPBsalsa152, 1)
#else
__launch_bounds__(TPBsalsa150, 1)
#endif
void neoscrypt_gpu_hash_salsa1_stream1(const uint32_t threads,const uint2x4* Input, uint2x4 *const __restrict__ W2, uint2x4 *const __restrict__ Tr2){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint2x4 Z[8];
	uint16* XV = (uint16*)Z;
	
	if(thread<threads){

		#pragma unroll 8
		for(uint32_t i = 0; i<8; i++)
			Z[i] = Input[i*threads+thread];

		#pragma nounroll
		for(uint32_t i = 0; i < 128; ++i){
		
			#if __CUDA_ARCH__ > 500
				#pragma unroll 8
				for(uint32_t j = 0; j<8; j++)
					W2[(thread<<10) + (i<<3) + j] = Z[j];
			#else
				#pragma unroll 8
				for(uint32_t j = 0; j<8; j++)
					W2[((i<<3)+j)*threads+thread] = Z[j];
			#endif
			const uint16 temp = XV[ 2];

			XV[0] = salsa_small_scalar_rnd(XV[0] ^ XV[3]);
			XV[2] = salsa_small_scalar_rnd(XV[1] ^ XV[0]);
			XV[1] = salsa_small_scalar_rnd(XV[2] ^ temp);
			XV[3] = salsa_small_scalar_rnd(XV[3] ^ XV[1]);
		}

		#pragma unroll 8
		for(uint32_t i = 0; i<8; i++)
			Tr2[i*threads+thread] = Z[i];
	}
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPBsalsa252, 1)
#else
__launch_bounds__(TPBsalsa250, 1)
#endif
void neoscrypt_gpu_hash_salsa2_stream1(const uint32_t threads, const uint2x4 *const __restrict__ W2, uint2x4 *const __restrict__ Tr2){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint2x4 X[8];
	uint16* XV = (uint16*)X;
	if(thread<threads){

		#pragma unroll 8
		for(uint32_t i = 0; i<8; i++)
			X[i] = __ldg4(&Tr2[i*threads+thread]);

		#pragma unroll 128
		for(uint32_t t = 0; t < 128; t++)
		{
			const uint32_t idx = (X[6].x.x & 0x7F) << 3;
			#if __CUDA_ARCH__ > 500
				#pragma unroll 8
				for(uint32_t j = 0; j<8; j++)
					X[j] ^= __ldg4(&W2[(thread<<10) + idx + j]);
			#else
				uint2x4 tmp[8];
				#pragma nounroll
				for(uint32_t j = 0; j<8; j++)
					tmp[j] = __ldg4(&W2[(idx+j)*threads + thread]);				
				#pragma nounroll
				for(uint32_t j = 0; j<8; j++)
					X[j] ^= tmp[j];
			#endif
			const uint16 temp = XV[ 2];

			XV[0] = salsa_small_scalar_rnd(XV[0] ^ XV[3]);
			XV[2] = salsa_small_scalar_rnd(XV[1] ^ XV[0]);
			XV[1] = salsa_small_scalar_rnd(XV[2] ^ temp);
			XV[3] = salsa_small_scalar_rnd(XV[3] ^ XV[1]);
		}

		#pragma unroll 8
		for(uint32_t i = 0; i<8; i++)
			Tr2[i*threads+thread] = X[i];
	}
}

__global__ __launch_bounds__(TPB, 3)
void neoscrypt_gpu_hash_ending(int stratum, uint32_t threads, uint32_t startNonce,const uint2x4 *const __restrict__ Tr,const uint2x4 *const __restrict__ Tr2, uint32_t *resNonces,const uint32_t target){

	__shared__ uint32_t s_data[64 * TPB];

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if(thread<threads){
		const uint32_t cdata7 = c_data[7];
		const uint32_t data18 = c_data[18];
		const uint32_t data20 = c_data[0];
		const uint32_t Nonce = startNonce + thread;
		const uint32_t nonce = (stratum) ? cuda_swab32(Nonce) : Nonce;

		uint32_t* B0 = (uint32_t*)&s_data[threadIdx.x * 64];

		uint32_t input[16];
		#pragma unroll 8
		for (int i = 0; i<8; i++){
			*(uint2x4*)&B0[i<<3] = __ldg4(&Tr2[i*threads+thread]) ^ __ldg4(&Tr[i*threads+thread]);
		}

		*(uint2x4*)&input[ 0] = *(uint2x4*)&c_data[ 0];
		*(uint2x4*)&input[ 8] = *(uint2x4*)&c_data[ 8];

		uint32_t key[16];
		*(uint2x4*)&key[0] = *(uint2x4*)&B0[0];
		*(uint4*)&key[ 8] = make_uint4(0, 0, 0, 0);
		*(uint4*)&key[12] = make_uint4(0, 0, 0, 0);

		uint32_t qbuf, bitbuf;
		uint32_t temp[9];

		#pragma unroll 
		for (int i = 0; i < 31; i++){

			Blake2S_v2(input, input, key);

			uint8_t bufidx = 0;
			#pragma unroll 4
			for (uint32_t x = 0; x < 8; x+=2){
				bufidx+= __vsadu4(input[x],0) + __vsadu4(input[x+1],0);
			}
			qbuf = bufidx >> 2;
			bitbuf = (bufidx & 3) << 3;

			uint32_t shift = 32U - bitbuf;
			const uint32_t byte_perm_shift = 0x76543210ULL >> (shift>>1);
			const uint32_t byte_perm_bitbf = 0x76543210ULL >> (bitbuf>>1);
			
			shift = (input[7]>>shift);
			B0[(8 + qbuf) & 0x3f] = (temp[8] = B0[(8 + qbuf) & 0x3f] ^ shift);
			shift = __byte_perm(input[ 6],input[ 7],byte_perm_shift);
			B0[(7 + qbuf) & 0x3f] = (temp[7] = B0[(7 + qbuf) & 0x3f] ^ shift);
			shift = __byte_perm(input[ 5],input[ 6],byte_perm_shift);
			B0[(6 + qbuf) & 0x3f] = (temp[6] = B0[(6 + qbuf) & 0x3f] ^ shift);
			shift = __byte_perm(input[ 4],input[ 5],byte_perm_shift);
			B0[(5 + qbuf) & 0x3f] = (temp[5] = B0[(5 + qbuf) & 0x3f] ^ shift);
			shift = __byte_perm(input[ 3],input[ 4],byte_perm_shift);
			B0[(4 + qbuf) & 0x3f] = (temp[4] = B0[(4 + qbuf) & 0x3f] ^ shift);
			shift = __byte_perm(input[ 2],input[ 3],byte_perm_shift);
			B0[(3 + qbuf) & 0x3f] = (temp[3] = B0[(3 + qbuf) & 0x3f] ^ shift);
			shift = __byte_perm(input[ 1],input[ 2],byte_perm_shift);
			B0[(2 + qbuf) & 0x3f] = (temp[2] = B0[(2 + qbuf) & 0x3f] ^ shift);
			shift = __byte_perm(input[ 0],input[ 1],byte_perm_shift);
			B0[(1 + qbuf) & 0x3f] = (temp[1] = B0[(1 + qbuf) & 0x3f] ^ shift);
			B0[(0 + qbuf) & 0x3f] = (temp[0] = B0[(0 + qbuf) & 0x3f] ^ (input[ 0]<<bitbuf));
			

			uint32_t a = c_data[qbuf & 0x3f];

			#pragma unroll 8
			for (int k = 0; k<16; k += 2)
			{
				const uint32_t b = c_data[(qbuf + k + 1) & 0x3f];
				input[k] = __byte_perm(a,b,byte_perm_bitbf);
				a = c_data[(qbuf + k + 2) & 0x3f];
				input[k+1] = __byte_perm(b,a,byte_perm_bitbf);
				key[(k>>1)] = __byte_perm(temp[(k>>1)],temp[(k>>1)+1],byte_perm_bitbf);
			}

			if(qbuf<60){
				const uint32_t noncepos = 19 - qbuf % 20;
				if (noncepos <= 16){
					if (noncepos != 0)
						input[noncepos - 1] = __byte_perm(data18,nonce,byte_perm_bitbf);
					if (noncepos != 16)
						input[noncepos] = __byte_perm(nonce,data20,byte_perm_bitbf);
				}
			}
		}

		Blake2S_v2(input, input, key);

		uint8_t bufidx = 0;
		#pragma unroll 4
		for (uint32_t x = 0; x < 8; x+=2){
			bufidx+= __vsadu4(input[x],0) + __vsadu4(input[x+1],0);
		}
		qbuf = bufidx >> 2;
		const uint32_t byte_perm_bitbf = 0x76543210ULL >> ((bufidx & 3)<<2);
		const uint32_t output = input[ 7] ^ cdata7 ^ __byte_perm(B0[(qbuf + 7) & 0x3f],B0[(qbuf + 8) & 0x3f],byte_perm_bitbf);

		if (output <= target){
	//		resNonces[0] = nonce;
			uint32_t tmp = atomicExch(resNonces, Nonce);
			if(tmp != UINT32_MAX)
				resNonces[1] = tmp;
		}
	}
}

uint2x4* W[MAX_GPUS];
uint2x4* W2[MAX_GPUS]; // 2 streams
uint2x4* Trans1[MAX_GPUS];
uint2x4* Trans2[MAX_GPUS]; // 2 streams
uint2x4 *Input[MAX_GPUS]; // 2 streams

void neoscrypt_cpu_init(int thr_id, uint32_t threads)
{
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[thr_id][0]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[thr_id][1]));

	CUDA_SAFE_CALL(cudaMalloc(&W[thr_id], 32 * 128 * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&W2[thr_id], 32 * 128 * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&Trans1[thr_id], 32 * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&Trans2[thr_id], 32 * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&Input[thr_id], 32 * sizeof(uint64_t) * threads));
}

__host__
void neoscrypt_free(int thr_id){

	cudaFree(W[thr_id]);
	cudaFree(W2[thr_id]);
	cudaFree(Trans1[thr_id]);
	cudaFree(Trans2[thr_id]);
	cudaFree(Input[thr_id]);

	cudaStreamDestroy(stream[thr_id][0]);
	cudaStreamDestroy(stream[thr_id][1]);
}

__host__ void neoscrypt_cpu_hash_k4(bool stratum, int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_resNonce,const uint32_t target7){
	
	const uint32_t threadsperblock2 = TPB;
	dim3 grid((threads + threadsperblock2 - 1) / threadsperblock2);
	dim3 block(threadsperblock2);

	const int threadsperblock3 = device_sm[device_map[thr_id]]>500 ? TPBchacha152 : TPBchacha150;
	dim3 grid3((threads + threadsperblock3 - 1) / threadsperblock3);
	dim3 block3(threadsperblock3);
	
	const int threadsperblock4 = device_sm[device_map[thr_id]]>500 ? TPBsalsa152 : TPBsalsa150;
	dim3 grid4((threads + threadsperblock4 - 1) / threadsperblock4);
	dim3 block4(threadsperblock4);
	
	const int threadsperblock5 = device_sm[device_map[thr_id]]>500 ? TPBchacha252 : TPBchacha250;
	dim3 grid5((threads + threadsperblock5 - 1) / threadsperblock5);
	dim3 block5(threadsperblock5);
	
	const int threadsperblock6 = device_sm[device_map[thr_id]]>500 ? TPBsalsa252 : TPBsalsa250;
	dim3 grid6((threads + threadsperblock6 - 1) / threadsperblock6);
	dim3 block6(threadsperblock6);

	neoscrypt_gpu_hash_start <<< grid, block >>>(stratum, threads, startNounce,Input[thr_id]); //fastkdf

	cudaThreadSynchronize();

	neoscrypt_gpu_hash_salsa1_stream1 <<< grid4, block4, 0, stream[thr_id][0] >>>(threads,Input[thr_id], W2[thr_id], Trans2[thr_id]); //salsa
	neoscrypt_gpu_hash_chacha1_stream1 <<< grid3, block3, 0, stream[thr_id][1] >>>(threads,Input[thr_id], W[thr_id], Trans1[thr_id]); //chacha

	neoscrypt_gpu_hash_salsa2_stream1 <<< grid6, block6, 0, stream[thr_id][0] >>>(threads, W2[thr_id], Trans2[thr_id]);//salsa
	neoscrypt_gpu_hash_chacha2_stream1 <<< grid5, block5, 0, stream[thr_id][1] >>>(threads, W[thr_id], Trans1[thr_id]); //chacha

	cudaStreamSynchronize(stream[thr_id][0]);
	cudaStreamSynchronize(stream[thr_id][1]);

	neoscrypt_gpu_hash_ending << <grid, block >> >(stratum, threads, startNounce, Trans1[thr_id], Trans2[thr_id], d_resNonce,target7); //fastkdf+end
}

__host__ void neoscrypt_setBlockTarget(uint32_t* pdata)
{
	uint32_t PaddedMessage[64];
	uint32_t input[16], key[16] = { 0 };

	for (int i = 0; i < 19; i++)
	{
		PaddedMessage[i] = pdata[i];
		PaddedMessage[i + 20] = pdata[i];
		PaddedMessage[i + 40] = pdata[i];
	}
	for (int i = 0; i<4; i++)
		PaddedMessage[i + 60] = pdata[i];

	PaddedMessage[19] = 0;
	PaddedMessage[39] = 0;
	PaddedMessage[59] = 0;

	((uint16*)input)[0] = ((uint16*)pdata)[0];
	((uint8*)key)[0] = ((uint8*)pdata)[0];

	Blake2Shost(input, key);

//	cudaMemcpyToSymbol(input_init, input, 64, 0, cudaMemcpyHostToDevice);
	uint8_t bufidx = 0;
	for (int x = 0; x < 8; x++){
		uint32_t bufhelper = (input[x] & 0x00ff00ff) + ((input[x] & 0xff00ff00) >> 8);
		bufhelper = bufhelper + (bufhelper >> 16);
		bufidx += bufhelper;
	}
	uint32_t qbuf, rbuf, bitbuf;
	qbuf = bufidx >> 2;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;
	uint32_t shifted[ 9];

	uint32_t shift = 32 - bitbuf;

	#define LONGLONG(LO,HI) ((uint64_t)LO | (((uint64_t)HI) << 32))

	shifted[ 0] = input[ 0] << bitbuf;
	shifted[ 1] = LONGLONG(input[ 0],input[ 1]) >> shift;
	shifted[ 2] = LONGLONG(input[ 1],input[ 2]) >> shift;
	shifted[ 3] = LONGLONG(input[ 2],input[ 3]) >> shift;
	shifted[ 4] = LONGLONG(input[ 3],input[ 4]) >> shift;
	shifted[ 5] = LONGLONG(input[ 4],input[ 5]) >> shift;
	shifted[ 6] = LONGLONG(input[ 5],input[ 6]) >> shift;
	shifted[ 7] = LONGLONG(input[ 6],input[ 7]) >> shift;
	shifted[ 8] = LONGLONG(input[ 7],        0) >> shift;
		
	uint32_t values[11];//qbuf,bitbuf,shiftedvalues
	values[ 0] = qbuf;
	values[ 1] = 0x76543210ULL >> (bitbuf >> 1);

	memcpy(&values[2],shifted,9*sizeof(uint32_t));
	
	cudaMemcpyToSymbol(buf_shifts,values,11*sizeof(uint32_t),0,cudaMemcpyHostToDevice);
	
		
	cudaMemcpyToSymbol(key_init, key, 64, 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(c_data, PaddedMessage, 64 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaGetLastError());
}
