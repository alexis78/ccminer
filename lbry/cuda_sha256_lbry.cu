/*
 * sha256 + ripemd CUDA implementation for compute5.0 GPUS
 * tpruvot and Provos Alexis - JUL 2016
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <cuda_vectors.h>

#include <miner.h>

__constant__ static uint32_t _ALIGN(16) c_midstate112[8];
__constant__ static uint32_t _ALIGN(16) c_midbuffer112[8];
__constant__ static uint32_t _ALIGN(16) c_dataEnd112[12];

__constant__  const uint32_t c_H256[8] = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};
__constant__ static uint32_t c_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

#ifdef __INTELLISENSE__
#define atomicExch(p,y) y
#endif

// ------------------------------------------------------------------------------------------------

static const uint32_t cpu_H256[8] = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint32_t cpu_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

__host__
static void sha256_step1_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h, uint32_t in, const uint32_t Kshared)
{
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);
	uint32_t t1 = h + bsg21 + vxandx + Kshared + in;
	uint32_t t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

__host__
static void sha256_step2_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h, uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
	int pcidx1 = (pc-2)  & 0xF;
	int pcidx2 = (pc-7)  & 0xF;
	int pcidx3 = (pc-15) & 0xF;

	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];

	uint32_t ssg21 = ROTR32(inx1, 17) ^ ROTR32(inx1, 19) ^ SPH_T32((inx1) >> 10); //ssg2_1(inx1);
	uint32_t ssg20 = ROTR32(inx3, 7) ^ ROTR32(inx3, 18) ^ SPH_T32((inx3) >> 3); //ssg2_0(inx3);
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);
	uint32_t t1,t2;

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d =  d + t1;
	h = t1 + t2;
}

__host__
static void sha256_round_body_host(uint32_t* in, uint32_t* state, const uint32_t* Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha256_step1_host(a,b,c,d,e,f,g,h,in[0], Kshared[0]);
	sha256_step1_host(h,a,b,c,d,e,f,g,in[1], Kshared[1]);
	sha256_step1_host(g,h,a,b,c,d,e,f,in[2], Kshared[2]);
	sha256_step1_host(f,g,h,a,b,c,d,e,in[3], Kshared[3]);
	sha256_step1_host(e,f,g,h,a,b,c,d,in[4], Kshared[4]);
	sha256_step1_host(d,e,f,g,h,a,b,c,in[5], Kshared[5]);
	sha256_step1_host(c,d,e,f,g,h,a,b,in[6], Kshared[6]);
	sha256_step1_host(b,c,d,e,f,g,h,a,in[7], Kshared[7]);
	sha256_step1_host(a,b,c,d,e,f,g,h,in[8], Kshared[8]);
	sha256_step1_host(h,a,b,c,d,e,f,g,in[9], Kshared[9]);
	sha256_step1_host(g,h,a,b,c,d,e,f,in[10],Kshared[10]);
	sha256_step1_host(f,g,h,a,b,c,d,e,in[11],Kshared[11]);
	sha256_step1_host(e,f,g,h,a,b,c,d,in[12],Kshared[12]);
	sha256_step1_host(d,e,f,g,h,a,b,c,in[13],Kshared[13]);
	sha256_step1_host(c,d,e,f,g,h,a,b,in[14],Kshared[14]);
	sha256_step1_host(b,c,d,e,f,g,h,a,in[15],Kshared[15]);

	for (int i=0; i<3; i++)
	{
		sha256_step2_host(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
		sha256_step2_host(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
		sha256_step2_host(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
		sha256_step2_host(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
		sha256_step2_host(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
		sha256_step2_host(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
		sha256_step2_host(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
		sha256_step2_host(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
		sha256_step2_host(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
		sha256_step2_host(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
		sha256_step2_host(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha256_step2_host(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha256_step2_host(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha256_step2_host(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha256_step2_host(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha256_step2_host(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
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

#define xor3b(a,b,c) (a ^ b ^ c)

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
	return xor3b(ROTR32(x,2),ROTR32(x,13),ROTR32(x,22));
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
	return xor3b(ROTR32(x,6),ROTR32(x,11),ROTR32(x,25));
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
	return xor3b(ROTR32(x,7),ROTR32(x,18),(x>>3));
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
	return xor3b(ROTR32(x,17),ROTR32(x,19),(x>>10));
}

__device__ __forceinline__ uint32_t ssg2_11(const uint32_t x)
{
	return xor3b(ROTR32(x,17),ROTR32(x,19),shr_u32(x,10));
}

__device__ __forceinline__ uint2 vectorizeswap(const uint64_t v)
{
	uint2 result;
	asm volatile ("mov.b64 {%0,%1},%2;" : "=r"(result.y), "=r"(result.x) : "l"(v));
	return result;
}

#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define Ch(a, b, c)     (((b^c) & a) ^ c)

__device__ __forceinline__
static void sha2_step(const uint32_t a,const uint32_t b,const uint32_t c, uint32_t &d,const uint32_t e,const uint32_t f,const uint32_t g, uint32_t &h,const uint32_t in, const uint32_t Kshared)
{
	const uint32_t t1 = h + bsg2_1(e) + Ch(e, f, g) + Kshared + in;
	h = t1 + bsg2_0(a) + Maj(a, b, c);
	d+= t1;

}

__device__ __forceinline__
static void sha256_round_first(uint32_t *in,uint32_t* buf,const uint32_t *const __restrict__ state)
{
	uint32_t a = buf[0] + in[11];
	uint32_t b = buf[1];
	uint32_t c = buf[2];
	uint32_t d = buf[3];
	uint32_t e = buf[4] + in[11];
	uint32_t f = buf[5];
	uint32_t g = buf[6];
	uint32_t h = buf[7];

	// 10 first steps made on host
	//sha2_step(f,g,h,a,b,c,d,e,in[11],c_K[11]);

	sha2_step(e,f,g,h,a,b,c,d,in[12],c_K[12]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],c_K[13]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],c_K[14]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],c_K[15]);

	//in is partially precomputed on host
	in[2]+= in[11];
	in[4]+= ssg2_1(in[2]);
	in[6]+= ssg2_1(in[4]);
	in[8]+= ssg2_1(in[6]);
	in[9]+= in[ 2];

	sha2_step(a,b,c,d,e,f,g,h,in[0], c_K[16]);
	sha2_step(h,a,b,c,d,e,f,g,in[1], c_K[17]);
	sha2_step(g,h,a,b,c,d,e,f,in[2], c_K[18]);
	sha2_step(f,g,h,a,b,c,d,e,in[3], c_K[19]);
	sha2_step(e,f,g,h,a,b,c,d,in[4], c_K[20]);
	sha2_step(d,e,f,g,h,a,b,c,in[5], c_K[21]);
	sha2_step(c,d,e,f,g,h,a,b,in[6], c_K[22]);
	sha2_step(b,c,d,e,f,g,h,a,in[7], c_K[23]);
	sha2_step(a,b,c,d,e,f,g,h,in[8], c_K[24]);
	sha2_step(h,a,b,c,d,e,f,g,in[9], c_K[25]);

	#pragma unroll 6
	for (uint32_t j = 10; j < 16; j++){
		in[j] = in[j] + in[(j + 9) & 15] + ssg2_0(in[(j + 1) & 15]) + ssg2_1(in[(j + 14) & 15]);
	}

	sha2_step(g,h,a,b,c,d,e,f,in[10],c_K[26]);
	sha2_step(f,g,h,a,b,c,d,e,in[11],c_K[27]);
	sha2_step(e,f,g,h,a,b,c,d,in[12],c_K[28]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],c_K[29]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],c_K[30]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],c_K[31]);
	
	#pragma unroll 16
	for (uint32_t j = 0; j < 16; j++){
		in[j] = in[j] + in[(j + 9) & 15] + ssg2_0(in[(j + 1) & 15]) + ssg2_1(in[(j + 14) & 15]);
	}

	sha2_step(a,b,c,d,e,f,g,h,in[0], c_K[16+16]);
	sha2_step(h,a,b,c,d,e,f,g,in[1], c_K[17+16]);
	sha2_step(g,h,a,b,c,d,e,f,in[2], c_K[18+16]);
	sha2_step(f,g,h,a,b,c,d,e,in[3], c_K[19+16]);
	sha2_step(e,f,g,h,a,b,c,d,in[4], c_K[20+16]);
	sha2_step(d,e,f,g,h,a,b,c,in[5], c_K[21+16]);
	sha2_step(c,d,e,f,g,h,a,b,in[6], c_K[22+16]);
	sha2_step(b,c,d,e,f,g,h,a,in[7], c_K[23+16]);
	sha2_step(a,b,c,d,e,f,g,h,in[8], c_K[24+16]);
	sha2_step(h,a,b,c,d,e,f,g,in[9], c_K[25+16]);
	sha2_step(g,h,a,b,c,d,e,f,in[10],c_K[26+16]);
	sha2_step(f,g,h,a,b,c,d,e,in[11],c_K[27+16]);
	sha2_step(e,f,g,h,a,b,c,d,in[12],c_K[28+16]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],c_K[29+16]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],c_K[30+16]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],c_K[31+16]);

	#pragma unroll 16
	for (uint32_t j = 0; j < 16; j++){
		in[j] = in[j] + in[(j + 9) & 15] + ssg2_0(in[(j + 1) & 15]) + ssg2_1(in[(j + 14) & 15]);
	}

	sha2_step(a,b,c,d,e,f,g,h,in[0], c_K[16+16*2]);
	sha2_step(h,a,b,c,d,e,f,g,in[1], c_K[17+16*2]);
	sha2_step(g,h,a,b,c,d,e,f,in[2], c_K[18+16*2]);
	sha2_step(f,g,h,a,b,c,d,e,in[3], c_K[19+16*2]);
	sha2_step(e,f,g,h,a,b,c,d,in[4], c_K[20+16*2]);
	sha2_step(d,e,f,g,h,a,b,c,in[5], c_K[21+16*2]);
	sha2_step(c,d,e,f,g,h,a,b,in[6], c_K[22+16*2]);
	sha2_step(b,c,d,e,f,g,h,a,in[7], c_K[23+16*2]);
	sha2_step(a,b,c,d,e,f,g,h,in[8], c_K[24+16*2]);
	sha2_step(h,a,b,c,d,e,f,g,in[9], c_K[25+16*2]);
	sha2_step(g,h,a,b,c,d,e,f,in[10],c_K[26+16*2]);
	sha2_step(f,g,h,a,b,c,d,e,in[11],c_K[27+16*2]);
	sha2_step(e,f,g,h,a,b,c,d,in[12],c_K[28+16*2]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],c_K[29+16*2]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],c_K[30+16*2]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],c_K[31+16*2]);

	buf[ 0] = state[0] + a;
	buf[ 1] = state[1] + b;
	buf[ 2] = state[2] + c;
	buf[ 3] = state[3] + d;
	buf[ 4] = state[4] + e;
	buf[ 5] = state[5] + f;
	buf[ 6] = state[6] + g;
	buf[ 7] = state[7] + h;
}

__device__ __forceinline__
static void sha256_round_body(uint32_t* in, uint32_t* state)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step(a,b,c,d,e,f,g,h,in[0], c_K[0]);
	sha2_step(h,a,b,c,d,e,f,g,in[1], c_K[1]);
	sha2_step(g,h,a,b,c,d,e,f,in[2], c_K[2]);
	sha2_step(f,g,h,a,b,c,d,e,in[3], c_K[3]);
	sha2_step(e,f,g,h,a,b,c,d,in[4], c_K[4]);
	sha2_step(d,e,f,g,h,a,b,c,in[5], c_K[5]);
	sha2_step(c,d,e,f,g,h,a,b,in[6], c_K[6]);
	sha2_step(b,c,d,e,f,g,h,a,in[7], c_K[7]);
	sha2_step(a,b,c,d,e,f,g,h,in[8], c_K[8]);
	sha2_step(h,a,b,c,d,e,f,g,in[9], c_K[9]);
	sha2_step(g,h,a,b,c,d,e,f,in[10],c_K[10]);
	sha2_step(f,g,h,a,b,c,d,e,in[11],c_K[11]);
	sha2_step(e,f,g,h,a,b,c,d,in[12],c_K[12]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],c_K[13]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],c_K[14]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],c_K[15]);

	#pragma unroll 3
	for (uint32_t i=0; i<3; i++)
	{
		#pragma unroll 16
		for (uint32_t j = 0; j < 16; j++){
			in[j]+= ssg2_11(in[(j + 14) & 15]) + in[(j + 9) & 15] + ssg2_0(in[(j + 1) & 15]);
		}
		sha2_step(a, b, c, d, e, f, g, h, in[0], c_K[16 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[1], c_K[17 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[2], c_K[18 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[3], c_K[19 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[4], c_K[20 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[5], c_K[21 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[6], c_K[22 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[7], c_K[23 + 16 * i]);
		sha2_step(a, b, c, d, e, f, g, h, in[8], c_K[24 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[9], c_K[25 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[10], c_K[26 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[11], c_K[27 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[12], c_K[28 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[13], c_K[29 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[14], c_K[30 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[15], c_K[31 + 16 * i]);
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
static void sha256_round_body_final(uint32_t* in, uint32_t* state)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step(a,b,c,d,e,f,g,h,in[0], c_K[0]);
	sha2_step(h,a,b,c,d,e,f,g,in[1], c_K[1]);
	sha2_step(g,h,a,b,c,d,e,f,in[2], c_K[2]);
	sha2_step(f,g,h,a,b,c,d,e,in[3], c_K[3]);
	sha2_step(e,f,g,h,a,b,c,d,in[4], c_K[4]);
	sha2_step(d,e,f,g,h,a,b,c,in[5], c_K[5]);
	sha2_step(c,d,e,f,g,h,a,b,in[6], c_K[6]);
	sha2_step(b,c,d,e,f,g,h,a,in[7], c_K[7]);
	sha2_step(a,b,c,d,e,f,g,h,in[8], c_K[8]);
	sha2_step(h,a,b,c,d,e,f,g,in[9], c_K[9]);
	sha2_step(g,h,a,b,c,d,e,f,in[10],c_K[10]);
	sha2_step(f,g,h,a,b,c,d,e,in[11],c_K[11]);
	sha2_step(e,f,g,h,a,b,c,d,in[12],c_K[12]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],c_K[13]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],c_K[14]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],c_K[15]);

	#pragma unroll 2
	for (uint32_t i=0; i<2; i++)
	{
		#pragma unroll 16
		for (uint32_t j = 0; j < 16; j++){
			in[j] = in[j] + in[(j + 9) & 15] + ssg2_0(in[(j + 1) & 15]) + ssg2_1(in[(j + 14) & 15]);
		}
		sha2_step(a, b, c, d, e, f, g, h, in[0], c_K[16 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[1], c_K[17 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[2], c_K[18 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[3], c_K[19 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[4], c_K[20 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[5], c_K[21 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[6], c_K[22 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[7], c_K[23 + 16 * i]);
		sha2_step(a, b, c, d, e, f, g, h, in[8], c_K[24 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[9], c_K[25 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[10], c_K[26 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[11], c_K[27 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[12], c_K[28 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[13], c_K[29 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[14], c_K[30 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[15], c_K[31 + 16 * i]);
	}
	#pragma unroll 14
	for (uint32_t j = 0; j < 14; j++){
		in[j] = in[j] + in[(j + 9) & 15] + ssg2_0(in[(j + 1) & 15]) + ssg2_1(in[(j + 14) & 15]);
	}
	sha2_step(a, b, c, d, e, f, g, h, in[0], c_K[16 + 16 * 2]);
	sha2_step(h, a, b, c, d, e, f, g, in[1], c_K[17 + 16 * 2]);
	sha2_step(g, h, a, b, c, d, e, f, in[2], c_K[18 + 16 * 2]);
	sha2_step(f, g, h, a, b, c, d, e, in[3], c_K[19 + 16 * 2]);
	sha2_step(e, f, g, h, a, b, c, d, in[4], c_K[20 + 16 * 2]);
	sha2_step(d, e, f, g, h, a, b, c, in[5], c_K[21 + 16 * 2]);
	sha2_step(c, d, e, f, g, h, a, b, in[6], c_K[22 + 16 * 2]);
	sha2_step(b, c, d, e, f, g, h, a, in[7], c_K[23 + 16 * 2]);
	sha2_step(a, b, c, d, e, f, g, h, in[8], c_K[24 + 16 * 2]);
	sha2_step(h, a, b, c, d, e, f, g, in[9], c_K[25 + 16 * 2]);
	sha2_step(g, h, a, b, c, d, e, f, in[10], c_K[26 + 16 * 2]);
	sha2_step(f, g, h, a, b, c, d, e, in[11], c_K[27 + 16 * 2]);
	sha2_step(e, f, g, h, a, b, c, d, in[12], c_K[28 + 16 * 2]);
	sha2_step(d, e, f, g, h, a, b, c, in[13], c_K[29 + 16 * 2]);
	
	state[6] = cuda_swab32(state[6] + g);
	state[7] = cuda_swab32(state[7] + h);
}

__global__ __launch_bounds__(768,2) /* to force 32 regs */
void lbry_sha256d_gpu_hash_112(const uint32_t threads, const uint32_t startNonce, uint64_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint32_t buf[8], state[8];
	if (thread < threads)
	{
		uint32_t dat[16];

		#pragma unroll 11
		for (uint32_t i=0; i<11; i++) dat[i] = c_dataEnd112[i];
		dat[11] = startNonce + thread;
		dat[12] = 0x80000000;
		dat[13] = 0;
		dat[14] = 0;
		dat[15] = 0x380;

		*(uint2x4*)&state[0] = *(uint2x4*)&c_midstate112[0];
		*(uint2x4*)&buf[0]   = *(uint2x4*)&c_midbuffer112[0];

		sha256_round_first(dat, buf, state); // no shared mem here

		// second sha256

		*(uint2x4*)&dat[0] = *(uint2x4*)&buf[0];

		dat[8] = 0x80000000;

		#pragma unroll 6
		for (uint32_t i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		*(uint2x4*)&buf[0] = *(uint2x4*)&c_H256[0];

		sha256_round_body(dat, buf); //no shared mem at all

		// output
		*(uint2*)&buf[0] = vectorizeswap(((uint64_t*)buf)[0]);
		*(uint2*)&buf[2] = vectorizeswap(((uint64_t*)buf)[1]);
		*(uint2*)&buf[4] = vectorizeswap(((uint64_t*)buf)[2]);
		*(uint2*)&buf[6] = vectorizeswap(((uint64_t*)buf)[3]);

		*(uint2x4*)&outputHash[thread<<3] = *(uint2x4*)&buf[0];
	}
}

__host__
void lbry_sha256d_hash_112(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_outputHash){

	const int threadsperblock = 768;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	lbry_sha256d_gpu_hash_112 <<<grid, block>>> (threads, startNonce, (uint64_t*) d_outputHash);
}

__host__
void lbry_sha256_init(int thr_id)
{
	cudaMemcpyToSymbol(c_K, cpu_K, sizeof(cpu_K), 0, cudaMemcpyHostToDevice);
}

__host__
void lbry_sha256_setBlock_112(uint32_t *pdata){

	uint32_t in[16], buf[8], end[16];
	for (int i=0;i<16;i++) in[i] = cuda_swab32(pdata[i]);
	for (int i=0; i<8;i++) buf[i] = cpu_H256[i];
	for (int i=0;i<11;i++) end[i] = cuda_swab32(pdata[16+i]);
	sha256_round_body_host(in, buf, cpu_K);

	cudaMemcpyToSymbol(c_midstate112, buf, 32, 0, cudaMemcpyHostToDevice);
	
	uint32_t a = buf[0];
	uint32_t b = buf[1];
	uint32_t c = buf[2];
	uint32_t d = buf[3];
	uint32_t e = buf[4];
	uint32_t f = buf[5];
	uint32_t g = buf[6];
	uint32_t h = buf[7];

	sha256_step1_host(a,b,c,d,e,f,g,h,end[0], cpu_K[0]);
	sha256_step1_host(h,a,b,c,d,e,f,g,end[1], cpu_K[1]);
	sha256_step1_host(g,h,a,b,c,d,e,f,end[2], cpu_K[2]);
	sha256_step1_host(f,g,h,a,b,c,d,e,end[3], cpu_K[3]);
	sha256_step1_host(e,f,g,h,a,b,c,d,end[4], cpu_K[4]);
	sha256_step1_host(d,e,f,g,h,a,b,c,end[5], cpu_K[5]);
	sha256_step1_host(c,d,e,f,g,h,a,b,end[6], cpu_K[6]);
	sha256_step1_host(b,c,d,e,f,g,h,a,end[7], cpu_K[7]);
	sha256_step1_host(a,b,c,d,e,f,g,h,end[8], cpu_K[8]);
	sha256_step1_host(h,a,b,c,d,e,f,g,end[9], cpu_K[9]);
	sha256_step1_host(g,h,a,b,c,d,e,f,end[10],cpu_K[10]);
	sha256_step1_host(f, g, h, a, b, c, d, e, 0, cpu_K[11]);

	buf[0] = a;
	buf[1] = b;
	buf[2] = c;
	buf[3] = d;
	buf[4] = e;
	buf[5] = f;
	buf[6] = g;
	buf[7] = h;

	cudaMemcpyToSymbol(c_midbuffer112, buf, 32, 0, cudaMemcpyHostToDevice);

	end[12] = 0x80000000;
	end[13] = 0;
	end[14] = 0;
	end[15] = 0x380;
	uint32_t x2_0,x2_1;
	
	x2_0 = ROTR32(end[1], 7) ^ ROTR32(end[1], 18) ^ SPH_T32(end[1] >> 3); //ssg2_0(end[1]);
//	x2_1 = ROTR32(end[14], 17) ^ ROTR32(end[14], 19) ^ SPH_T32(end[14] >> 10) + x2_0; //ssg2_1(end[14]) + x2_0;
	end[0] = end[0] + end[9] + x2_0;
	
	x2_0 = ROTR32(end[2], 7) ^ ROTR32(end[2], 18) ^ SPH_T32(end[2] >> 3);
	x2_1 = (ROTR32(end[15], 17) ^ ROTR32(end[15], 19) ^ SPH_T32(end[15] >> 10)) + x2_0;
	end[1] = end[1] + end[10] + x2_1;
	
	x2_0 = ROTR32(end[3], 7) ^ ROTR32(end[3], 18) ^ SPH_T32(end[3] >> 3);//ssg2_0(end[3]);
	x2_1 = (ROTR32(end[0], 17) ^ ROTR32(end[0], 19) ^ SPH_T32(end[0] >> 10)) + x2_0;
	end[2]+= x2_1;
	
	x2_0 = ROTR32(end[4], 7) ^ ROTR32(end[4], 18) ^ SPH_T32(end[4] >> 3);//ssg2_0(end[4]);
	x2_1 = (ROTR32(end[1], 17) ^ ROTR32(end[1], 19) ^ SPH_T32(end[1] >> 10)) + x2_0;
	end[3] = end[3] + end[12] + x2_1;
	
	x2_0 = ROTR32(end[5], 7) ^ ROTR32(end[5], 18) ^ SPH_T32(end[5] >> 3);//ssg2_0(end[4]);
	end[4] = end[4] + end[13] + x2_0;
	
	x2_0 = ROTR32(end[6], 7) ^ ROTR32(end[6], 18) ^ SPH_T32(end[6] >> 3);//ssg2_0(end[6]);
	x2_1 = (ROTR32(end[3], 17) ^ ROTR32(end[3], 19) ^ SPH_T32(end[3] >> 10)) + x2_0;
	end[5] = end[5] + end[14] + x2_1;

	x2_0 = ROTR32(end[7], 7) ^ ROTR32(end[7], 18) ^ SPH_T32(end[7] >> 3);//ssg2_0(end[7]);
	end[6] = end[6] + end[15] + x2_0;

	x2_0 = ROTR32(end[8], 7) ^ ROTR32(end[8], 18) ^ SPH_T32(end[8] >> 3);//ssg2_0(end[8]);
	x2_1 = (ROTR32(end[5], 17) ^ ROTR32(end[5], 19) ^ SPH_T32(end[5] >> 10)) + x2_0;
	end[7] = end[7] + end[0] + x2_1;
	
	x2_0 = ROTR32(end[9], 7) ^ ROTR32(end[9], 18) ^ SPH_T32(end[9] >> 3);//ssg2_0(end[9]);
	end[8] = end[8] + end[1] + x2_0;

	x2_0 = ROTR32(end[10], 7) ^ ROTR32(end[10], 18) ^ SPH_T32(end[10] >> 3);//ssg2_0(end[10]);
	x2_1 = (ROTR32(end[7], 17) ^ ROTR32(end[7], 19) ^ SPH_T32(end[7] >> 10)) + x2_0;
	end[9] = end[9] + x2_1;
	
	cudaMemcpyToSymbol(c_dataEnd112,  end, 12*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

// RIPEMD MACROS-----------------------------------------------------------------------------

/*
 * Round constants for RIPEMD-160.
 */
static __constant__ const uint32_t c_IV[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
static __constant__ const uint32_t KL[5]   = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
static __constant__ const uint32_t KR[5]   = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};

/* Left line */
static __constant__ const uint32_t RL[5][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },   /* Round 1: id */
	{ 7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8 },   /* Round 2: rho */
	{ 3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12 },   /* Round 3: rho^2 */
	{ 1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2 },   /* Round 4: rho^3 */
	{ 4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13 }    /* Round 5: rho^4 */
};

/* Right line */
static __constant__ const uint32_t RR[5][16] = {
	{ 5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12 },   /* Round 1: pi */
	{ 6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2 },   /* Round 2: rho pi */
	{ 15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13 },   /* Round 3: rho^2 pi */
	{ 8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14 },   /* Round 4: rho^3 pi */
	{ 12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11 }    /* Round 5: rho^4 pi */
};

/* Shifts, left line */
static __constant__ const uint32_t SL[5][16] = {
	{ 11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8 }, /* Round 1 */
	{ 7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12 }, /* Round 2 */
	{ 11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5 }, /* Round 3 */
	{ 11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12 }, /* Round 4 */
	{ 9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6 }  /* Round 5 */
};

/* Shifts, right line */
static __constant__ const uint32_t SR[5][16] = {
	{ 8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6 }, /* Round 1 */
	{ 9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11 }, /* Round 2 */
	{ 9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5 }, /* Round 3 */
	{ 15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8 }, /* Round 4 */
	{ 8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11 }  /* Round 5 */
};

__device__ __forceinline__
static uint32_t ROTATE(const uint32_t x,const uint32_t r){
	if(r==8)
		return __byte_perm(x, 0, 0x2103);
	else
		return ROTL32(x,r);
}

/*
 * Round functions for RIPEMD-160.
 */
//#define F1(x, y, z)   xor3x(x, y, z)
__device__ __forceinline__
uint32_t F1(const uint32_t a,const uint32_t b,const uint32_t c){
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result) : "r"(a), "r"(b),"r"(c));
	#else
		result = a^b^c;
	#endif
	return result;
}


//#define F2(x, y, z)   ((x & (y ^ z)) ^ z)
__device__ __forceinline__
uint32_t F2(const uint32_t a,const uint32_t b,const uint32_t c){
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(result) : "r"(a), "r"(b),"r"(c)); //0xCA=((F0∧(CC⊻AA))⊻AA)
	#else
		result = ((a & (b ^ c)) ^ c);
	#endif
	return result;
}

//#define F3(x, y, z)   ((x | ~y) ^ z)
__device__ __forceinline__
uint32_t F3(const uint32_t x,const uint32_t y,const uint32_t z){
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0x59;" : "=r"(result) : "r"(x), "r"(y),"r"(z)); //0x59=((F0∨(¬CC))⊻AA)
	#else
		result = ((x | ~y) ^ z);
	#endif
	return result;
}
//#define F4(x, y, z)   (y ^ ((x ^ y) & z))
__device__ __forceinline__
uint32_t F4(const uint32_t x,const uint32_t y,const uint32_t z){
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(result) : "r"(x), "r"(y),"r"(z)); //0xE4=(CC⊻((F0⊻CC)∧AA))
	#else
		result = (y ^ ((x ^ y) & z));
	#endif
	return result;
}

//#define F5(x, y, z)   (x ^ (y | ~z))
__device__ __forceinline__
uint32_t F5(const uint32_t x,const uint32_t y,const uint32_t z){
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0x2D;" : "=r"(result) : "r"(x), "r"(y),"r"(z)); //0x2D=(F0⊻(CC∨(¬AA)))
	#else
		result = (x ^ (y | ~z));
	#endif
	return result;
}

__device__ __forceinline__
static void RIPEMD160_ROUND_BODY(const uint32_t *in, uint32_t *h){
	uint32_t T;
	uint32_t AL, BL, CL, DL, EL;    /* left line */
	uint32_t AR, BR, CR, DR, ER;    /* right line */
    
	AL = AR = h[0];
	BL = BR = h[1];
	CL = CR = h[2];
	DL = DR = h[3];
	EL = ER = h[4];

	/* Round 1 */
	#pragma unroll 16
	for (uint32_t w = 0; w < 16; w++) {
		T = ROTATE(AL + F1(BL, CL, DL) + in[RL[0][w]] + KL[0], SL[0][w]) + EL;
		AL = EL; EL = DL; DL = ROTL32(CL,10); CL = BL; BL = T;

		T = ROTATE(AR + F5(BR, CR, DR) + in[RR[0][w]] + KR[0], SR[0][w]) + ER;
		AR = ER; ER = DR; DR = ROTL32(CR,10); CR = BR; BR = T;
	}

	/* Round 2 */
	#pragma unroll 16
	for (uint32_t w = 0; w < 16; w++) {
		T = ROTATE(AL + F2(BL, CL, DL) + in[RL[1][w]] + KL[1], SL[1][w]) + EL;
		AL = EL; EL = DL; DL = ROTL32(CL,10); CL = BL; BL = T;

		T = ROTATE(AR + F4(BR, CR, DR) + in[RR[1][w]] + KR[1], SR[1][w]) + ER;
		AR = ER; ER = DR; DR = ROTL32(CR,10); CR = BR; BR = T;
	}

	/* Round 3 */
	#pragma unroll 16
	for (uint32_t w = 0; w < 16; w++) {
		T = ROTATE(AL + F3(BL, CL, DL) + in[RL[2][w]] + KL[2], SL[2][w]) + EL;
		AL = EL; EL = DL; DL = ROTL32(CL,10); CL = BL; BL = T;

		T = ROTATE(AR + F3(BR, CR, DR) + in[RR[2][w]] + KR[2], SR[2][w]) + ER;
		AR = ER; ER = DR; DR = ROTL32(CR,10); CR = BR; BR = T;
	}

	/* Round 4 */
	#pragma unroll 16
	for (uint32_t w = 0; w < 16; w++) {
		T = ROTATE(AL + F4(BL, CL, DL) + in[RL[3][w]] + KL[3], SL[3][w]) + EL;
		AL = EL; EL = DL; DL = ROTL32(CL,10); CL = BL; BL = T;

		T = ROTATE(AR + F2(BR, CR, DR) + in[RR[3][w]] + KR[3], SR[3][w]) + ER;
		AR = ER; ER = DR; DR = ROTL32(CR,10); CR = BR; BR = T;
	}

	/* Round 5 */
	#pragma unroll 16
	for (uint32_t w = 0; w < 16; w++) {
		T = ROTATE(AL + F5(BL, CL, DL) + in[RL[4][w]] + KL[4], SL[4][w]) + EL;
		AL = EL; EL = DL; DL = ROTL32(CL,10); CL = BL; BL = T;

		T = ROTATE(AR + F1(BR, CR, DR) + in[RR[4][w]] + KR[4], SR[4][w]) + ER;
		AR = ER; ER = DR; DR = ROTL32(CR,10); CR = BR; BR = T;
	}

	T    = h[1] + CL + DR;
	h[1] = h[2] + DL + ER;
	h[2] = h[3] + EL + AR;
	h[3] = h[4] + AL + BR;
	h[4] = h[0] + BL + CR;
	h[0] = T;
}
// END OF RIPEMD MACROS----------------------------------------------------------------------

__global__ __launch_bounds__(1024,2) /* to force 32 regs */
void lbry_ripemd(const uint32_t threads, uint32_t *const __restrict__ Hash512){
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint32_t dat[16];
	uint32_t h[5];	
	uint32_t buf[8]; // align for vectorize
	if(thread<threads){

		const uint32_t* input = &Hash512[thread<<4];

		*(uint2x4*)&dat[0] = __ldg4((uint2x4*)&input[0]);

		dat[8] = 0x80;

		#pragma unroll 7
		for (int i=9;i<16;i++) dat[i] = 0;

		dat[14] = 0x100; // size in bits

		#pragma unroll 5
		for (int i=0; i<5; i++)
			h[i] = c_IV[i];

		RIPEMD160_ROUND_BODY(dat, h);

		#pragma unroll 5
		for (int i=0; i<5; i++)
			buf[i] = h[i];

		// second 32 bytes block hash

		*(uint2x4*)&dat[0] = __ldg4((uint2x4*)&input[8]);

		dat[8] = 0x80;

		#pragma unroll 7
		for (int i=9;i<16;i++) dat[i] = 0;

		dat[14] = 0x100; // size in bits

		#pragma unroll 5
		for (int i=0; i<5; i++)
			h[i] = c_IV[i];

		RIPEMD160_ROUND_BODY(dat, h);

		#pragma unroll 5
		for (int i=0;i<5;i++) dat[i] = cuda_swab32(buf[i]);
		#pragma unroll 5
		for (int i=0;i<5;i++) dat[i+5] = cuda_swab32(h[i]);
		
		*(uint2x4*)&Hash512[(thread<<4)] = *(uint2x4*)&dat[ 0];
		*(uint2*)&Hash512[(thread<<4)+8] = *(uint2*)&dat[ 8];
	}
}

__global__ __launch_bounds__(768,2) /* to force 32 regs */
void lbry_sha256d_gpu_hash_final(const uint32_t threads,const uint32_t* __restrict__ Hash512, uint32_t *resNonces,const uint64_t target64)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint32_t dat[16];
	uint32_t buf[8]; // align for vectorize	
	if (thread < threads)
	{
		// first final sha256

		const uint32_t* input = &Hash512[thread<<4];

		*(uint2x4*)&dat[0] = __ldg4((uint2x4*)&input[0]);
		*(uint2*)&dat[8] = __ldg((uint2*)&input[8]);
		dat[10] = 0x80000000;
		#pragma unroll 4
		for (int i=11; i<15; i++) dat[i] = 0;
		dat[15] = 0x140;

		*(uint2x4*)&buf[0] = *(uint2x4*)&c_H256[0];

		sha256_round_body(dat, buf);

		// second sha256

		*(uint2x4*)&dat[0] = *(uint2x4*)&buf[0];
		*(uint2x4*)&buf[0] = *(uint2x4*)&c_H256[0];
		
		dat[8] = 0x80000000;

		#pragma unroll 6
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		sha256_round_body_final(dat, buf);

		// valid nonces
		if (*(uint64_t*)&buf[ 6] <= target64){
			uint32_t tmp = atomicExch(&resNonces[0], thread);
			if (tmp != UINT32_MAX)
				resNonces[1] = tmp;
		}
	}
}

__host__
void lbry_sha256d_hash_final(int thr_id, uint32_t threads, uint32_t *d_inputHash, uint32_t *d_resNonce, const uint64_t target64)
{
	int threadsperblock;
	
	threadsperblock = 1024;
	dim3 grid1((threads + threadsperblock - 1) / threadsperblock);
	dim3 block1(threadsperblock);

	threadsperblock = 768;
	dim3 grid2((threads + threadsperblock - 1) / threadsperblock);
	dim3 block2(threadsperblock);
	lbry_ripemd <<<grid1, block1>>> (threads,d_inputHash);
	lbry_sha256d_gpu_hash_final <<<grid2, block2>>> (threads, d_inputHash, d_resNonce, target64);
}
