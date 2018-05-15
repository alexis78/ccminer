#include <stdio.h>
#include <memory.h>
#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h" 

__device__ ulonglong8to16 *state2;
uint32_t *d_YNonce[MAX_GPUS];
__constant__  uint32_t pTarget[8];
__constant__  uint32_t  c_data[32];
__constant__  uint16 shapad;

static uint32_t *d_hash[MAX_GPUS];
static uint8*    d_hash2[MAX_GPUS];
static uint32*   d_hash3[MAX_GPUS];
static uint32*   d_hash4[MAX_GPUS];

#define xor3b(a,b,c) (a^b^c)
#define andor32(x, y, z)    ((x & (y | z)) | (y & z))
#define xandx(a, b, c)     (((b^c) & a) ^ c)

#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define Ch(a, b, c)     (((b^c) & a) ^ c)

static __device__ __forceinline__ void madd4long2(ulonglong2 &a, ulonglong2 b){

	asm ("{\n\t"
		".reg .u32 a0,a1,a2,a3,b0,b1,b2,b3;\n\t"
		"mov.b64 {a0,a1}, %0;\n\t"
		"mov.b64 {a2,a3}, %1;\n\t"
		"mov.b64 {b0,b1}, %2;\n\t"
		"mov.b64 {b2,b3}, %3;\n\t"
		"mad.lo.cc.u32        b0,a0,a1,b0;  \n\t"
		"madc.hi.u32          b1,a0,a1,b1;  \n\t"
		"mad.lo.cc.u32        b2,a2,a3,b2;  \n\t"
		"madc.hi.u32          b3,a2,a3,b3;  \n\t"
		"mov.b64 %0, {b0,b1};\n\t"
		"mov.b64 %1, {b2,b3};\n\t"
		"}\n\t"
		: "+l"(a.x), "+l"(a.y) : "l"(b.x), "l"(b.y));

}

static __device__ __forceinline__ ulonglong2 __ldg2(const ulonglong2 *ptr){

	ulonglong2 ret;
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR(ptr));
	return ret;
}

static __device__ __forceinline__ uint32 __ldg32b(const uint32 *ptr)
{
	uint32 ret;
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"     : "=r"(ret.lo.s0), "=r"(ret.lo.s1), "=r"(ret.lo.s2), "=r"(ret.lo.s3) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];"  : "=r"(ret.lo.s4), "=r"(ret.lo.s5), "=r"(ret.lo.s6), "=r"(ret.lo.s7) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+32];"  : "=r"(ret.lo.s8), "=r"(ret.lo.s9), "=r"(ret.lo.sa), "=r"(ret.lo.sb) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+48];"  : "=r"(ret.lo.sc), "=r"(ret.lo.sd), "=r"(ret.lo.se), "=r"(ret.lo.sf) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+64];"  : "=r"(ret.hi.s0), "=r"(ret.hi.s1), "=r"(ret.hi.s2), "=r"(ret.hi.s3) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+80];"  : "=r"(ret.hi.s4), "=r"(ret.hi.s5), "=r"(ret.hi.s6), "=r"(ret.hi.s7) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+96];"  : "=r"(ret.hi.s8), "=r"(ret.hi.s9), "=r"(ret.hi.sa), "=r"(ret.hi.sb) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+112];" : "=r"(ret.hi.sc), "=r"(ret.hi.sd), "=r"(ret.hi.se), "=r"(ret.hi.sf) : __LDG_PTR(ptr));
	return ret;
}

static __device__ __forceinline__ uint16 __ldg16b(const uint16 *ptr)
{
	uint16 ret; 
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"     : "=r"(ret.s0), "=r"(ret.s1), "=r"(ret.s2), "=r"(ret.s3) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];"  : "=r"(ret.s4), "=r"(ret.s5), "=r"(ret.s6), "=r"(ret.s7) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+32];"  : "=r"(ret.s8), "=r"(ret.s9), "=r"(ret.sa), "=r"(ret.sb) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+48];"  : "=r"(ret.sc), "=r"(ret.sd), "=r"(ret.se), "=r"(ret.sf) : __LDG_PTR(ptr));
	return ret;
}

///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// sha256 Transform function /////////////////////////


static __constant__ const uint16 pad1 = 
{
	0x36363636, 0x36363636, 0x36363636, 0x36363636,	0x36363636, 0x36363636, 0x36363636, 0x36363636,	0x36363636, 0x36363636, 0x36363636, 0x36363636,	0x36363636, 0x36363636, 0x36363636, 0x36363636
};
static __constant__ const uint16 pad2 = 
{
	0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c,	0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c,	0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c,	0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c
};
static __constant__ const uint16 pad5 =
{
	0x00000001, 0x80000000, 0x00000000, 0x00000000,	0x00000000, 0x00000000, 0x00000000, 0x00000000,	0x00000000, 0x00000000, 0x00000000, 0x00000000,	0x00000000, 0x00000000, 0x00000000, 0x00002220
};
static __constant__ const uint16 padsha80 =
{
	0x00000000, 0x00000000, 0x00000000, 0x00000000,	0x80000000, 0x00000000, 0x00000000, 0x00000000,	0x00000000, 0x00000000, 0x00000000, 0x00000000,	0x00000000, 0x00000000, 0x00000000, 0x00000280
};

static __constant__ const uint8 pad4 =
{
	0x80000000, 0x00000000, 0x00000000, 0x00000000,	0x00000000, 0x00000000, 0x00000000, 0x00000300
};

static __constant__ const uint8 H256 = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C,	0x1F83D9AB, 0x5BE0CD19
};

__constant__ static uint32_t _ALIGN(16) Ksha[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,	0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,	0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,	0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,	0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,	0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,	0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,	0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,	0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

__device__ __forceinline__
static uint32_t bsg2_0(const uint32_t x){
	return xor3b(ROTR32(x, 2), ROTR32(x, 13), ROTR32(x, 22));
}
__device__ __forceinline__
static uint32_t bsg2_1(const uint32_t x){
	return xor3b(ROTR32(x, 6), ROTR32(x, 11), ROTR32(x, 25));
}
__device__ __forceinline__
static uint32_t ssg2_0(const uint32_t x){
	return xor3b(ROTR32(x, 7), ROTR32(x, 18), shr_u32(x, 3));
}
__device__ __forceinline__
static uint32_t ssg2_1(const uint32_t x){
	return xor3b(ROTR32(x, 17), ROTR32(x, 19), shr_u32(x, 10));
}

__device__ __forceinline__
static void sha2_step1(const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d, const uint32_t e, const uint32_t f, const uint32_t g, uint32_t &h, const uint32_t in, const uint32_t Kshared){
	const uint32_t t1 = h + bsg2_1(e) + Ch(e, f, g) + Kshared + in;
	h = t1 + bsg2_0(a) + Maj(a, b, c);
	d+= t1;
}

__device__ __forceinline__
static void sha2_step2(const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d, const uint32_t e,const uint32_t f, const uint32_t g, uint32_t &h, uint32_t* in, const uint32_t pc, const uint32_t Kshared){
	uint32_t t1, t2;

	int pcidx1 = (pc - 2) & 0xF;
	int pcidx2 = (pc - 7) & 0xF;
	int pcidx3 = (pc - 15) & 0xF;
	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];

	uint32_t ssg21 = ssg2_1(inx1);
	uint32_t ssg20 = ssg2_0(inx3);
	uint32_t vxandx = xandx(e, f, g);
	uint32_t bsg21 = bsg2_1(e);
	uint32_t bsg20 = bsg2_0(a);
	uint32_t andorv = andor32(a, b, c);

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}



#define SALSA(a,b,c,d) { \
	b^=ROTL32(a+d,  7); \
	c^=ROTL32(a+b,  9); \
	d^=ROTL32(b+c, 13); \
	a^=ROTL32(d+c, 18); \
}

#define SALSA_CORE(state) { \
\
SALSA(state.s0,state.s4,state.s8,state.sc); \
SALSA(state.s5,state.s9,state.sd,state.s1); \
SALSA(state.sa,state.se,state.s2,state.s6); \
SALSA(state.sf,state.s3,state.s7,state.sb); \
SALSA(state.s0,state.s1,state.s2,state.s3); \
SALSA(state.s5,state.s6,state.s7,state.s4); \
SALSA(state.sa,state.sb,state.s8,state.s9); \
SALSA(state.sf,state.sc,state.sd,state.se); \
} 

#define uSALSA_CORE(state) { \
\
SALSA(state.s0,state.s4,state.s8,state.sc); \
SALSA(state.s1,state.s5,state.s9,state.sd); \
SALSA(state.s2,state.s6,state.sa,state.se); \
SALSA(state.s3,state.s7,state.sb,state.sf); \
SALSA(state.s0,state.sd,state.sa,state.s7); \
SALSA(state.s1,state.se,state.sb,state.s4); \
SALSA(state.s2,state.sf,state.s8,state.s5); \
SALSA(state.s3,state.sc,state.s9,state.s6); \
} 


#define shuffle(stat,state) { \
stat.s0 = state.s0; \
stat.s1 = state.s5; \
stat.s2 = state.sa; \
stat.s3 = state.sf; \
stat.s4 = state.s4; \
stat.s5 = state.s9; \
stat.s6 = state.se; \
stat.s7 = state.s3; \
stat.s8 = state.s8; \
stat.s9 = state.sd; \
stat.sa = state.s2; \
stat.sb = state.s7; \
stat.sc = state.sc; \
stat.sd = state.s1; \
stat.se = state.s6; \
stat.sf = state.sb; \
}
#define unshuffle(state,X) { \
    state.s0 = X.s0; \
    state.s1 = X.sd; \
    state.s2 = X.sa; \
    state.s3 = X.s7; \
    state.s4 = X.s4; \
    state.s5 = X.s1; \
    state.s6 = X.se; \
    state.s7 = X.sb; \
    state.s8 = X.s8; \
    state.s9 = X.s5; \
    state.sa = X.s2; \
    state.sb = X.sf; \
    state.sc = X.sc; \
    state.sd = X.s9; \
    state.se = X.s6; \
    state.sf = X.s3; \
}

__device__ __forceinline__
static void sha256_Transform(uint32_t* in, uint32_t *state){ // also known as sha2_round_body
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step1(a, b, c, d, e, f, g, h, in[ 0], Ksha[0]);
	sha2_step1(h, a, b, c, d, e, f, g, in[ 1], Ksha[1]);
	sha2_step1(g, h, a, b, c, d, e, f, in[ 2], Ksha[2]);
	sha2_step1(f, g, h, a, b, c, d, e, in[ 3], Ksha[3]);
	sha2_step1(e, f, g, h, a, b, c, d, in[ 4], Ksha[4]);
	sha2_step1(d, e, f, g, h, a, b, c, in[ 5], Ksha[5]);
	sha2_step1(c, d, e, f, g, h, a, b, in[ 6], Ksha[6]);
	sha2_step1(b, c, d, e, f, g, h, a, in[ 7], Ksha[7]);
	sha2_step1(a, b, c, d, e, f, g, h, in[ 8], Ksha[8]);
	sha2_step1(h, a, b, c, d, e, f, g, in[ 9], Ksha[9]);
	sha2_step1(g, h, a, b, c, d, e, f, in[10], Ksha[10]);
	sha2_step1(f, g, h, a, b, c, d, e, in[11], Ksha[11]);
	sha2_step1(e, f, g, h, a, b, c, d, in[12], Ksha[12]);
	sha2_step1(d, e, f, g, h, a, b, c, in[13], Ksha[13]);
	sha2_step1(c, d, e, f, g, h, a, b, in[14], Ksha[14]);
	sha2_step1(b, c, d, e, f, g, h, a, in[15], Ksha[15]);

#pragma unroll 3
	for (uint32_t i = 0; i<3; i++) {

		sha2_step2(a, b, c, d, e, f, g, h, in, 0, Ksha[16 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, in, 1, Ksha[17 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, in, 2, Ksha[18 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, in, 3, Ksha[19 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, in, 4, Ksha[20 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, in, 5, Ksha[21 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, in, 6, Ksha[22 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, in, 7, Ksha[23 + 16 * i]);
		sha2_step2(a, b, c, d, e, f, g, h, in, 8, Ksha[24 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, in, 9, Ksha[25 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, in, 10, Ksha[26 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, in, 11, Ksha[27 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, in, 12, Ksha[28 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, in, 13, Ksha[29 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, in, 14, Ksha[30 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, in, 15, Ksha[31 + 16 * i]);

	}
	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

__device__ __forceinline__
static uint8 sha256_Transform2(uint16 in[1], const uint8 &r){ // also known as sha2_round_body

	uint8 tmp = r;
#define a  tmp.s0
#define b  tmp.s1
#define c  tmp.s2
#define d  tmp.s3
#define e  tmp.s4
#define f  tmp.s5
#define g  tmp.s6
#define h  tmp.s7

	sha2_step1(a, b, c, d, e, f, g, h, in[0].s0, Ksha[0]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s1, Ksha[1]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].s2, Ksha[2]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].s3, Ksha[3]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].s4, Ksha[4]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].s5, Ksha[5]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].s6, Ksha[6]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].s7, Ksha[7]);
	sha2_step1(a, b, c, d, e, f, g, h, in[0].s8, Ksha[8]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s9, Ksha[9]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].sa, Ksha[10]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].sb, Ksha[11]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].sc, Ksha[12]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].sd, Ksha[13]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].se, Ksha[14]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].sf, Ksha[15]);

	#pragma unroll 3
	for (uint32_t i = 0; i<3; i++) {

		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 0, Ksha[16 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 1, Ksha[17 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 2, Ksha[18 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 3, Ksha[19 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 4, Ksha[20 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 5, Ksha[21 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 6, Ksha[22 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 7, Ksha[23 + 16 * i]);
		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 8, Ksha[24 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 9, Ksha[25 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 10, Ksha[26 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 11, Ksha[27 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 12, Ksha[28 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 13, Ksha[29 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 14, Ksha[30 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 15, Ksha[31 + 16 * i]);

	}
#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
	return (r + tmp);
}


__device__ __forceinline__
static uint8 sha256_Transform3(uint32_t nonce,uint32_t next, const uint8 &r){ // also known as sha2_round_body

	uint8 tmp = r;
	uint16 in[1]={shapad};
	in[0].s3=nonce;
	in[0].s4=next;
#define a  tmp.s0
#define b  tmp.s1
#define c  tmp.s2
#define d  tmp.s3
#define e  tmp.s4
#define f  tmp.s5
#define g  tmp.s6
#define h  tmp.s7

	sha2_step1(a, b, c, d, e, f, g, h, in[0].s0, Ksha[0]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s1, Ksha[1]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].s2, Ksha[2]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].s3, Ksha[3]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].s4, Ksha[4]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].s5, Ksha[5]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].s6, Ksha[6]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].s7, Ksha[7]);
	sha2_step1(a, b, c, d, e, f, g, h, in[0].s8, Ksha[8]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s9, Ksha[9]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].sa, Ksha[10]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].sb, Ksha[11]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].sc, Ksha[12]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].sd, Ksha[13]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].se, Ksha[14]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].sf, Ksha[15]);

#pragma unroll 3
	for (uint32_t i = 0; i<3; i++) {

		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 0, Ksha[16 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 1, Ksha[17 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 2, Ksha[18 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 3, Ksha[19 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 4, Ksha[20 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 5, Ksha[21 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 6, Ksha[22 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 7, Ksha[23 + 16 * i]);
		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 8, Ksha[24 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 9, Ksha[25 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 10, Ksha[26 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 11, Ksha[27 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 12, Ksha[28 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 13, Ksha[29 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 14, Ksha[30 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 15, Ksha[31 + 16 * i]);

	}
#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
	return (r + tmp);
}

//////////////////////////////// end sha transform mechanism ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__
static uint16 salsa20_8(const uint16 &X){
	uint16 state=X;
	
	#pragma unroll 4
	for (uint32_t i = 0; i < 4; ++i)
		uSALSA_CORE(state);
	
	return(X + state);
}

__device__ __forceinline__
static void block_pwxform_long(int thread, ulonglong2to8 *const __restrict__ Bout,uint32 *const __restrict__ prevstate){

		ulonglong2 vec = Bout->l0;

		#pragma unroll 6
		for (uint32_t i = 0; i < 6; i++)
		{
			ulonglong2 p0, p1;
			uint2 x = vectorize((vec.x >> 4) & 0x000000FF000000FF);
			p0 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread))[x.x]);
			madd4long2(vec, p0);
			p1 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread + 32))[x.y]);


			vec ^= p1;
		}
		Bout->l0 = vec;
	        vec = Bout->l1;

		#pragma unroll 6
		for (uint32_t i = 0; i < 6; i++){
			ulonglong2 p0, p1;
			uint2 x = vectorize((vec.x >> 4) & 0x000000FF000000FF);
			p0 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread))[x.x]);
			madd4long2(vec, p0);
			p1 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread + 32))[x.y]);


			vec ^= p1;
		}
		Bout->l1 = vec;

		vec = Bout->l2;

		#pragma unroll 6
		for (uint32_t i = 0; i < 6; i++){
			ulonglong2 p0, p1;
			uint2 x = vectorize((vec.x >> 4) & 0x000000FF000000FF);
			p0 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread))[x.x]);
			madd4long2(vec, p0);
			p1 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread + 32))[x.y]);


			vec ^= p1;
		}
		Bout->l2 = vec;
		vec = Bout->l3;

		#pragma unroll 6
		for (uint32_t i = 0; i < 6; i++){
			ulonglong2 p0, p1;
			uint2 x = vectorize((vec.x >> 4) & 0x000000FF000000FF);
			p0 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread))[x.x]);
			madd4long2(vec, p0);
			p1 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread + 32))[x.y]);


			vec ^= p1;
		}
		Bout->l3 = vec;


}

__device__ __forceinline__
static void blockmix_salsa8_small2(uint32 &Bin)
{
	uint16 X = Bin.hi;
	X ^= Bin.lo;
	X = salsa20_8(X);
	Bin.lo = X;
	X ^= Bin.hi;
	X = salsa20_8(X);
	Bin.hi = X;
}

__device__ __forceinline__
static void blockmix_pwxform3(int thread, ulonglong2to8 *const __restrict__ Bin,uint32 *const __restrict__ prevstate){
	Bin[0] ^= Bin[15];
	block_pwxform_long(thread, &Bin[0],prevstate);

	for (uint32_t i = 1; i < 16; i++){

		Bin[i] ^= Bin[i - 1];
		block_pwxform_long(thread, &Bin[i],prevstate);
	}
	((uint16*)Bin)[15] = salsa20_8(((uint16*)Bin)[15]);
//	Bin[15] = salsa20_8_long(Bin[15]);
}


__global__ __launch_bounds__(256, 1)
void yescrypt_gpu_hash_k0(const uint32_t threads,const uint32_t startNonce, uint8* sha256test, uint32* B){


	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){
	
		const uint32_t nonce = startNonce + thread;
		uint16 in[1];
		uint8 state1, state2;
		uint8 passwd;// = sha256_80(nonce);

		uint32_t in1[16] = { 0 };
//		uint32_t buf[ 8];

		((uint16*)in1)[0] = ((uint16*)c_data)[0];

		passwd = H256;

		sha256_Transform(in1, ((uint32_t*)&passwd));
		((uint16*)in1)[0] = padsha80;
		in1[0] = c_data[16];
		in1[1] = c_data[17];
		in1[2] = c_data[18];
		in1[3] = nonce;
	
		sha256_Transform(in1, ((uint32_t*)&passwd));
		
		in[0].lo = pad1.lo ^ passwd;
		in[0].hi = pad1.hi;
		state1 = sha256_Transform2(in, H256);
		in[0].lo = pad2.lo ^ passwd;
		in[0].hi = pad2.hi;
		state2 = sha256_Transform2(in, H256);
		in[0] = ((uint16*)c_data)[0];
		///HMAC_SHA256_update(salt)
		state1 = sha256_Transform2(in, state1);
		#pragma unroll
		for (uint32_t i = 0; i<8; i++)
		{
			uint32 result;

			in[0].lo = sha256_Transform3(nonce,4*i+1, state1);
			in[0].hi = pad4;
			result.lo.lo = swapvec(sha256_Transform2(in, state2));
			if (i == 0) (sha256test + thread)[0] = result.lo.lo;

			in[0].lo = sha256_Transform3(nonce,4*i+2, state1);
			in[0].hi = pad4;
			result.lo.hi = swapvec(sha256_Transform2(in, state2));


			in[0].lo = sha256_Transform3(nonce,4*i+3, state1);
			in[0].hi = pad4;
			result.hi.lo = swapvec(sha256_Transform2(in, state2));

			in[0].lo = sha256_Transform3(nonce,4*i+4, state1);
			in[0].hi = pad4;
			result.hi.hi = swapvec(sha256_Transform2(in, state2));


			shuffle((B + 8 * thread)[i].lo, result.lo);
			shuffle((B + 8 * thread)[i].hi, result.hi);
		}

 
	}
}

__global__ __launch_bounds__(32, 1)
void yescrypt_gpu_hash_k1(const uint32_t threads, uint32_t startNonce,uint32* prevstate,uint32* B){


	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

//		smix1_first(thread);
		uint32 X;

#define Bdev(x) (B+8*thread)[x]
#define state(x) (prevstate+64*thread)[x]
		X = Bdev(0);
		state(0) = X; 
		blockmix_salsa8_small2(X);
		state(1) = X;
		blockmix_salsa8_small2(X);



		uint32_t n = 1;

		#pragma unroll
		for (uint32_t i = 2; i < 64; i ++)
		{

			state(i) = X;

			if ((i&(i - 1)) == 0) n = n << 1;

			uint32_t j = X.hi.s0 & (n - 1);

			j += i - n;
			X ^= __ldg32b(&state(j));

			blockmix_salsa8_small2(X);
		}

		Bdev(0) = X;
#undef Bdev
#undef state

	}
}

__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(128, 1)
#else
__launch_bounds__(16, 1)
#endif
void yescrypt_gpu_hash_k2c(int threads,uint32* prevstate,uint32* B){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		//		smix1_second(thread);
		ulonglong8to16 X[8]; //,Z;
		const uint32_t length = 8;
		const uint32_t shift = 8 * 2048 * thread;
 
#define Bdev(x) (B+8*thread)[x]

		#pragma unroll 8
		for (uint32_t i = 0; i<8; i++) {
			((uint32*)X)[i] = __ldg32b(&Bdev(i));
		}
		
		#pragma unroll 8
		for (uint32_t i = 0; i<length; i++)
			(state2+shift)[i] = X[i];

		blockmix_pwxform3(thread, (ulonglong2to8*)X, prevstate);

		#pragma unroll 8
		for (uint32_t i = 0; i<length; i++)
			(state2 + shift+length)[i] = X[i];

		blockmix_pwxform3(thread, (ulonglong2to8*)X, prevstate);

		uint32_t n = 1;

		for (uint32_t i = 2; i < 2048; i++){

			#pragma unroll 8
			for (uint32_t k = 0; k<length; k++)
				(state2 + shift + length * i)[k] = X[k];


			if ((i&(i - 1)) == 0) n = n << 1;

			const uint32_t j = (((uint32*)X)[7].hi.s0 & (n - 1)) + i - n;

			#pragma unroll 64
			for (uint32_t k = 0; k < 64; k++)
				((ulonglong2*)X)[k] ^= __ldg2(&((ulonglong2*)(state2 + shift + length * j))[k]);
						

			blockmix_pwxform3(thread, (ulonglong2to8*)X, prevstate);

		}
		#pragma unroll 8
		for (uint32_t i = 0; i<8; i++) {
			(B + 8 * thread)[i] = ((uint32*)X)[i];
		}
		/////////////////////////////////////////////////
	}
}

__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(128, 1)
#else
__launch_bounds__(16, 1)
#endif
void yescrypt_gpu_hash_k2c1(int threads,uint32* prevstate,uint32* B){


	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		ulonglong8to16 X[8]; //,Z;
		const uint32_t length = 8;
		const uint32_t shift = 8 * 2048 * thread;

#define Bdev(x) (B+8*thread)[x]
#define BigStore(s,i) (state2 + shift + s)[i]


		#pragma unroll 8
		for (uint32_t i = 0; i<8; i++) {
			((uint32*)X)[i] = __ldg32b(&Bdev(i));
		}
		
		for (uint32_t z = 0; z < 682; z++){
		
			uint32_t j = ((uint32*)X)[7].hi.s0 & 2047;

			#pragma unroll 64
			for (uint32_t k = 0; k < 64; k++)
				((ulonglong2*)X)[k] ^= __ldg2(&((ulonglong2*)(state2 + shift + length * j))[k]);

			#pragma unroll 8			
			for (uint32_t k = 0; k<length; k++)
				BigStore(length * j, k) = X[k];

			blockmix_pwxform3(thread, (ulonglong2to8*)X, prevstate);
		}
		for (uint32_t z = 682; z < 684; z++){
		
			uint32_t j = ((uint32*)X)[7].hi.s0 & 2047;

			#pragma unroll 64
			for (uint32_t k = 0; k < 64; k++)
				((ulonglong2*)X)[k] ^= __ldg2(&((ulonglong2*)(state2 + shift + length * j))[k]);

			blockmix_pwxform3(thread, (ulonglong2to8*)X, prevstate);

		}
		#pragma unroll 8
		for (uint32_t i = 0; i<8; i++) {
//			((uint32*)X)[i] = Bdev(i);
			unshuffle(Bdev(i).lo, ((uint32*)X)[i].lo);
			unshuffle(Bdev(i).hi, ((uint32*)X)[i].hi);
		}

	}
}

__global__ __launch_bounds__(16, 1)
void yescrypt_gpu_hash_k5(int threads, uint32_t startNonce, uint32_t *resNonces, uint8* sha256test, uint32* B){

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	const uint32_t nonce = startNonce + thread;

	uint16 in[1];
	uint8 state1, state2;

	uint8 swpass = (sha256test + thread)[0];
#define Bdev(x) (B+8*thread)[x]
	swpass = swapvec(swpass);
	in[0].lo = pad1.lo ^ swpass;
	in[0].hi = pad1.hi;

	state1 = sha256_Transform2(in, H256);

	in[0].lo = pad2.lo ^ swpass;
	in[0].hi = pad2.hi;
	state2 = sha256_Transform2(in, H256);

	for (uint32_t i = 0; i<8; i++) {
		in[0] = __ldg16b(&Bdev(i).lo);
		in[0] = swapvec(in[0]);
		state1 = sha256_Transform2(in, state1);
		in[0] = __ldg16b(&Bdev(i).hi);
		in[0] = swapvec(in[0]);
		state1 = sha256_Transform2(in, state1);
	}
	in[0] = pad5;
	state1 = sha256_Transform2(in, state1);
	in[0].lo = state1;
	in[0].hi = pad4;
	uint8 res = sha256_Transform2(in, state2);

	//hmac and final sha
	state1 = state2 = H256;
	in[0].lo = pad1.lo ^ res;
	in[0].hi = pad1.hi;
	state1 = sha256_Transform2(in, state1);
	in[0].lo = pad2.lo ^ res;
	in[0].hi = pad2.hi;
	state2 = sha256_Transform2(in, state2);
	in[0] = ((uint16*)c_data)[0];
	state1 = sha256_Transform2(in, state1);
	in[0] = padsha80;
	in[0].s0 = c_data[16];
	in[0].s1 = c_data[17];
	in[0].s2 = c_data[18];
	in[0].s3 = nonce;
	in[0].sf = 0x480;
	state1 = sha256_Transform2(in, state1);
	in[0].lo = state1;
	in[0].hi = pad4;
	state1 = sha256_Transform2(in, state2);
//	state2 = H256;
	in[0].lo = state1;
	in[0].hi = pad4;
	in[0].sf = 0x100;
	res = sha256_Transform2(in, H256);
//	return(swapvec(res));

//	uint8 res = pbkdf_sha256_second2(thread, nonce);
	if (cuda_swab32(res.s7) <= pTarget[7]) {
		uint32_t tmp = atomicExch(&resNonces[0], nonce);
		if(tmp != UINT32_MAX)
			resNonces[1] = tmp;
	}
}

__host__
void yescrypt_cpu_init(int thr_id, int threads){

	//PREPEI NA MPEI KAI FREE!
	CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id],  2048 * 128 * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&d_hash2[thr_id], 8 * sizeof(uint32_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&d_hash3[thr_id], 32*64 * sizeof(uint32_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&d_hash4[thr_id], 32*8 * sizeof(uint32_t) * threads));

	cudaMemcpyToSymbol(state2, &d_hash[thr_id], sizeof(d_hash[thr_id]), 0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(sha256test, &hash2, sizeof(hash2), 0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(prevstate, &d_hash3[thr_id], sizeof(d_hash3[thr_id]), 0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(B, &d_hash4[thr_id], sizeof(d_hash4[thr_id]), 0, cudaMemcpyHostToDevice);
} 

__host__
void yescrypt_free(int thr_id){
	cudaFree(d_hash[thr_id]);
	cudaFree(d_hash2[thr_id]);
	cudaFree(d_hash3[thr_id]);
	cudaFree(d_hash4[thr_id]);
}

__host__
void yescrypt_cpu_hash_k4(int thr_id, int threads, uint32_t startNounce, uint32_t* resNonce){
 
	int dev_id = device_map[thr_id];
	
	const uint32_t threadsperblock = 16;
	const uint32_t threadsperblock2 = 64;
	const uint32_t threadsperblock3 = 64;
	const uint32_t threadsperblock4 = 32;
	
	const uint32_t tpbk2c = device_sm[dev_id]<=500 ? 16 : 128;
	dim3 gridk2c((threads + tpbk2c - 1) / tpbk2c);
	dim3 blockk2c(tpbk2c);	
	 
	dim3 grid4((threads + threadsperblock4 - 1) / threadsperblock4);
	dim3 block4(threadsperblock4);

	dim3 grid3((threads + threadsperblock3 - 1) / threadsperblock3);
	dim3 block3(threadsperblock3);
	 
	dim3 grid2((threads + threadsperblock2 - 1) / threadsperblock2);
	dim3 block2(threadsperblock2);
	
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	yescrypt_gpu_hash_k0 << <grid3, block3>> >(threads, startNounce,d_hash2[thr_id],d_hash4[thr_id]);
	yescrypt_gpu_hash_k1 << <grid4, block4 >> >(threads, startNounce,d_hash3[thr_id],d_hash4[thr_id]);
	yescrypt_gpu_hash_k2c << <gridk2c, blockk2c >> >(threads, d_hash3[thr_id],d_hash4[thr_id]);
	yescrypt_gpu_hash_k2c1 << <gridk2c, blockk2c >> >(threads, d_hash3[thr_id],d_hash4[thr_id]);
	yescrypt_gpu_hash_k5 << <grid, block >> >(threads, startNounce, resNonce,d_hash2[thr_id],d_hash4[thr_id]);
}

__host__ void yescrypt_setBlockTarget(uint32_t* pdata, const void *target){

		unsigned char PaddedMessage[128]; //bring balance to the force
		memcpy(PaddedMessage,     pdata, 80);
//		memcpy(PaddedMessage+80, 0, 48);
		uint32_t pad3[16] =
		{
			0x00000000, 0x00000000, 0x00000000, 0x00000000,
			0x00000000, 0x80000000, 0x00000000, 0x00000000,
			0x00000000, 0x00000000, 0x00000000, 0x00000000,
			0x00000000, 0x00000000, 0x00000000, 0x000004a0
		};
		pad3[0] = pdata[16];
		pad3[1] = pdata[17];
		pad3[2] = pdata[18];
		 
		cudaMemcpyToSymbol(shapad, pad3, 16 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, target, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_data, PaddedMessage, 32 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}
