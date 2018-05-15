// Auf Myriadcoin spezialisierte Version von Groestl inkl. Bitslice
// Based on Tanguy Pruvot's repo
// Provos Alexis - 2016

#include "cuda_helper.h"
#include "miner.h"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#define __funnelshift_r(x,y,n) (x >> n)
#define atomicExch(p,x) x
#endif

// 64 Registers Variant for Compute 3.0
#include "quark/groestl_functions_quad.h"
#include "quark/groestl_transf_quad.h"

// globaler Speicher für alle HeftyHashes aller Threads
static uint32_t *d_outputHashes[MAX_GPUS];

__constant__ uint32_t _ALIGN(8) c_input[32];
// muss expandiert werden
__constant__ const uint32_t sha256_constantTable[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

__constant__ const uint32_t sha256_constantTable2[64] = {
	0xC28A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF374, 
	0x649B69C1, 0xF0FE4786, 0x0FE1EDC6, 0x240CF254, 0x4FE9346F, 0x6CC984BE, 0x61B9411E, 0x16F988FA, 0xF2C65152, 0xA88E5A6D, 0xB019FC65, 0xB9D99EC7, 0x9A1231C3, 0xE70EEAA0, 0xFDB1232B, 0xC7353EB0, 
	0x3069BAD5, 0xCB976D5F, 0x5A0F118F, 0xDC1EEEFD, 0x0A35B689, 0xDE0B7A04, 0x58F4CA9D, 0xE15D5B16, 0x007F3E86, 0x37088980, 0xA507EA32, 0x6FAB9537, 0x17406110, 0x0D8CD6F1, 0xCDAA3B6D, 0xC0BBBE37, 
	0x83613BDA, 0xDB48A363, 0x0B02E931, 0x6FD15CA7, 0x521AFACA, 0x31338431, 0x6ED41A95, 0x6D437890, 0xC39C91F2, 0x9ECCABBD, 0xB5C9A0E6, 0x532FB63C, 0xD2C741C6, 0x07237EA3, 0xA4954B68, 0x4C191D76
};

#define Ch(a, b, c)     (((b^c) & a) ^ c)
#define Maj(x, y, z)    ((x & (y | z)) | (y & z)) //((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

#define xor3b(a,b,c) ((a ^ b) ^ c)

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

__device__ __forceinline__
static void sha2_step1(const uint32_t a,const uint32_t b,const uint32_t c, uint32_t &d,const uint32_t e,const uint32_t f,const uint32_t g, uint32_t &h,const uint32_t in, const uint32_t Kshared)
{
	const uint32_t t1 = h + bsg2_1(e) + Ch(e, f, g) + Kshared + in;
	h = t1 + bsg2_0(a) + Maj(a, b, c);
	d+= t1;

}

__device__ __forceinline__
static void sha2_step2(const uint32_t a,const uint32_t b,const uint32_t c, uint32_t &d,const uint32_t e,const uint32_t f,const uint32_t g, uint32_t &h, const uint32_t Kshared)
{
	const uint32_t t1 = h + bsg2_1(e) + Ch(e, f, g) + Kshared;
	h = t1 + bsg2_0(a) + Maj(a, b, c);
	d+= t1;

}

__device__ __forceinline__
static void sha256_round_body(uint32_t* in, uint32_t* state,const uint32_t* __restrict__ Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step1(a,b,c,d,e,f,g,h,in[0], Kshared[0]);
	sha2_step1(h,a,b,c,d,e,f,g,in[1], Kshared[1]);
	sha2_step1(g,h,a,b,c,d,e,f,in[2], Kshared[2]);
	sha2_step1(f,g,h,a,b,c,d,e,in[3], Kshared[3]);
	sha2_step1(e,f,g,h,a,b,c,d,in[4], Kshared[4]);
	sha2_step1(d,e,f,g,h,a,b,c,in[5], Kshared[5]);
	sha2_step1(c,d,e,f,g,h,a,b,in[6], Kshared[6]);
	sha2_step1(b,c,d,e,f,g,h,a,in[7], Kshared[7]);
	sha2_step1(a,b,c,d,e,f,g,h,in[8], Kshared[8]);
	sha2_step1(h,a,b,c,d,e,f,g,in[9], Kshared[9]);
	sha2_step1(g,h,a,b,c,d,e,f,in[10],Kshared[10]);
	sha2_step1(f,g,h,a,b,c,d,e,in[11],Kshared[11]);
	sha2_step1(e,f,g,h,a,b,c,d,in[12],Kshared[12]);
	sha2_step1(d,e,f,g,h,a,b,c,in[13],Kshared[13]);
	sha2_step1(c,d,e,f,g,h,a,b,in[14],Kshared[14]);
	sha2_step1(b,c,d,e,f,g,h,a,in[15],Kshared[15]);

	#pragma unroll 3
	for (int i=0; i<3; i++)
	{
		#pragma unroll 16
		for (int j = 0; j < 16; j++){
			in[j] = in[j] + in[(j + 9) & 15] + ssg2_0(in[(j + 1) & 15]) + ssg2_1(in[(j + 14) & 15]);
		}
		sha2_step1(a, b, c, d, e, f, g, h, in[0], Kshared[16 + 16 * i]);
		sha2_step1(h, a, b, c, d, e, f, g, in[1], Kshared[17 + 16 * i]);
		sha2_step1(g, h, a, b, c, d, e, f, in[2], Kshared[18 + 16 * i]);
		sha2_step1(f, g, h, a, b, c, d, e, in[3], Kshared[19 + 16 * i]);
		sha2_step1(e, f, g, h, a, b, c, d, in[4], Kshared[20 + 16 * i]);
		sha2_step1(d, e, f, g, h, a, b, c, in[5], Kshared[21 + 16 * i]);
		sha2_step1(c, d, e, f, g, h, a, b, in[6], Kshared[22 + 16 * i]);
		sha2_step1(b, c, d, e, f, g, h, a, in[7], Kshared[23 + 16 * i]);
		sha2_step1(a, b, c, d, e, f, g, h, in[8], Kshared[24 + 16 * i]);
		sha2_step1(h, a, b, c, d, e, f, g, in[9], Kshared[25 + 16 * i]);
		sha2_step1(g, h, a, b, c, d, e, f, in[10], Kshared[26 + 16 * i]);
		sha2_step1(f, g, h, a, b, c, d, e, in[11], Kshared[27 + 16 * i]);
		sha2_step1(e, f, g, h, a, b, c, d, in[12], Kshared[28 + 16 * i]);
		sha2_step1(d, e, f, g, h, a, b, c, in[13], Kshared[29 + 16 * i]);
		sha2_step1(c, d, e, f, g, h, a, b, in[14], Kshared[30 + 16 * i]);
		sha2_step1(b, c, d, e, f, g, h, a, in[15], Kshared[31 + 16 * i]);
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
static void sha256_round_body_final(uint32_t* state,const uint32_t* Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step2(a,b,c,d,e,f,g,h, Kshared[0]);
	sha2_step2(h,a,b,c,d,e,f,g, Kshared[1]);
	sha2_step2(g,h,a,b,c,d,e,f, Kshared[2]);
	sha2_step2(f,g,h,a,b,c,d,e, Kshared[3]);
	sha2_step2(e,f,g,h,a,b,c,d, Kshared[4]);
	sha2_step2(d,e,f,g,h,a,b,c, Kshared[5]);
	sha2_step2(c,d,e,f,g,h,a,b, Kshared[6]);
	sha2_step2(b,c,d,e,f,g,h,a, Kshared[7]);
	sha2_step2(a,b,c,d,e,f,g,h, Kshared[8]);
	sha2_step2(h,a,b,c,d,e,f,g, Kshared[9]);
	sha2_step2(g,h,a,b,c,d,e,f, Kshared[10]);
	sha2_step2(f,g,h,a,b,c,d,e, Kshared[11]);
	sha2_step2(e,f,g,h,a,b,c,d, Kshared[12]);
	sha2_step2(d,e,f,g,h,a,b,c, Kshared[13]);
	sha2_step2(c,d,e,f,g,h,a,b, Kshared[14]);
	sha2_step2(b,c,d,e,f,g,h,a, Kshared[15]);

	#pragma unroll
	for (int i=0; i<2; i++){

		sha2_step2(a, b, c, d, e, f, g, h, Kshared[16 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, Kshared[17 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, Kshared[18 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, Kshared[19 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, Kshared[20 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, Kshared[21 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, Kshared[22 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, Kshared[23 + 16 * i]);
		sha2_step2(a, b, c, d, e, f, g, h, Kshared[24 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, Kshared[25 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, Kshared[26 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, Kshared[27 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, Kshared[28 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, Kshared[29 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, Kshared[30 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, Kshared[31 + 16 * i]);
	}
	sha2_step2(a, b, c, d, e, f, g, h, Kshared[16 + 16 * 2]);
	sha2_step2(h, a, b, c, d, e, f, g, Kshared[17 + 16 * 2]);
	sha2_step2(g, h, a, b, c, d, e, f, Kshared[18 + 16 * 2]);
	sha2_step2(f, g, h, a, b, c, d, e, Kshared[19 + 16 * 2]);
	sha2_step2(e, f, g, h, a, b, c, d, Kshared[20 + 16 * 2]);
	sha2_step2(d, e, f, g, h, a, b, c, Kshared[21 + 16 * 2]);
	sha2_step2(c, d, e, f, g, h, a, b, Kshared[22 + 16 * 2]);
	sha2_step2(b, c, d, e, f, g, h, a, Kshared[23 + 16 * 2]);
	sha2_step2(a, b, c, d, e, f, g, h, Kshared[24 + 16 * 2]);
	sha2_step2(h, a, b, c, d, e, f, g, Kshared[25 + 16 * 2]);
	sha2_step2(g, h, a, b, c, d, e, f, Kshared[26 + 16 * 2]);
	sha2_step2(f, g, h, a, b, c, d, e, Kshared[27 + 16 * 2]);
	sha2_step2(e, f, g, h, a, b, c, d, Kshared[28 + 16 * 2]);
	sha2_step2(d, e, f, g, h, a, b, c, Kshared[29 + 16 * 2]);

	state[6]+= g;
	state[7]+= h;
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(1024,2) /* to force 32 regs */
#else
__launch_bounds__(768,2) /* to force 32 regs */
#endif
void myriadgroestl_gpu_hash_sha(uint32_t threads, uint32_t startNounce, uint32_t* hashBuffer, uint32_t *resNonces,const uint64_t target64){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t W[16];
		uint32_t *inpHash = &hashBuffer[thread<<4];

		*(uint2x4*)&W[ 0] = __ldg4((uint2x4*)&inpHash[ 0]);
		*(uint2x4*)&W[ 8] = __ldg4((uint2x4*)&inpHash[ 8]);

		uint32_t buf[ 8] = {
			0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
			0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
		};

		sha256_round_body(W,buf,sha256_constantTable);
	
		sha256_round_body_final(buf,sha256_constantTable2);

		#if 0
			// Full sha hash
			#pragma unroll
			for(int k=0; k<8; k++)
				W[k] = cuda_swab32(buf[k]);
		#else
			W[6] = cuda_swab32(buf[6]);
			W[7] = cuda_swab32(buf[7]);
		#endif

		if (*(uint64_t*)&W[6] <= target64){
			uint32_t tmp = atomicExch(&resNonces[0], startNounce + thread);
			if (tmp != UINT32_MAX)
				resNonces[1] = tmp;
		}
	}
}

#define TPB52 512
#define TPB50 512
#define THF 4

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52, 2)
#else
__launch_bounds__(TPB50, 2)
#endif
void myriadgroestl_gpu_hash_quad(uint32_t threads, uint32_t startNounce, uint32_t *d_hash){

	// durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x)>>2;
	if (thread < threads)
	{
		const uint32_t thr = threadIdx.x & 3;
		// GROESTL
		uint32_t input[8];
		uint32_t other[8];
		uint32_t msgBitsliced[8];
		uint32_t state[8];
		uint32_t output[16];

		*(uint2x4*)input = *(uint2x4*)&c_input[((threadIdx.x & 2)<<3)];
		*(uint2x4*)other = *(uint2x4*)&c_input[(((threadIdx.x+1)&3)<<3)];
		#pragma unroll 8
		for(int k=0; k<8; k++){
//			input[k] = c_input[k+((threadIdx.x & 2)<<3)];
//			other[k] = c_input[k+(((threadIdx.x+1)&3)<<3)];
			other[k] = __shfl(other[k], threadIdx.x & 2, 4);
		}

		if ((thr == 2) || (thr == 3))
			other[4] = cuda_swab32(startNounce + thread);

		uint32_t t;

		const uint32_t perm = (threadIdx.x & 1) ? 0x7362 : 0x5140;

		merge8(msgBitsliced[0], input[0], input[4], perm);
		merge8(msgBitsliced[1], other[0], other[4], perm);
		merge8(msgBitsliced[2], input[1], input[5], perm);
		merge8(msgBitsliced[3], other[1], other[5], perm);
		merge8(msgBitsliced[4], input[2], input[6], perm);
		merge8(msgBitsliced[5], other[2], other[6], perm);
		merge8(msgBitsliced[6], input[3], input[7], perm);
		merge8(msgBitsliced[7], other[3], other[7], perm);

		SWAP1(msgBitsliced[0], msgBitsliced[1]);
		SWAP1(msgBitsliced[2], msgBitsliced[3]);
		SWAP1(msgBitsliced[4], msgBitsliced[5]);
		SWAP1(msgBitsliced[6], msgBitsliced[7]);

		SWAP2(msgBitsliced[0], msgBitsliced[2]);
		SWAP2(msgBitsliced[1], msgBitsliced[3]);
		SWAP2(msgBitsliced[4], msgBitsliced[6]);
		SWAP2(msgBitsliced[5], msgBitsliced[7]);

		SWAP4(msgBitsliced[0], msgBitsliced[4]);
		SWAP4(msgBitsliced[1], msgBitsliced[5]);
		SWAP4(msgBitsliced[2], msgBitsliced[6]);
		SWAP4(msgBitsliced[3], msgBitsliced[7]);

	        groestl512_progressMessage_quad(state, msgBitsliced,thr);

		from_bitslice_quad52(state, output);

		uint2x4* outHash = (uint2x4*)&d_hash[thread<<4];
		
#if __CUDA_ARCH__ <= 500
		output[0] = __byte_perm(output[0], __shfl(output[0], (threadIdx.x + 1) & 3, 4), 0x0167);
		output[2] = __byte_perm(output[2], __shfl(output[2], (threadIdx.x + 1) & 3, 4), 0x0167);
		output[4] = __byte_perm(output[4], __shfl(output[4], (threadIdx.x + 1) & 3, 4), 0x2367);
		output[6] = __byte_perm(output[6], __shfl(output[6], (threadIdx.x + 1) & 3, 4), 0x2367);
		output[8] = __byte_perm(output[8], __shfl(output[8], (threadIdx.x + 1) & 3, 4), 0x0167);
		output[10] = __byte_perm(output[10], __shfl(output[10], (threadIdx.x + 1) & 3, 4), 0x0167);
		output[12] = __byte_perm(output[12], __shfl(output[12], (threadIdx.x + 1) & 3, 4), 0x2367);
		output[14] = __byte_perm(output[14], __shfl(output[14], (threadIdx.x + 1) & 3, 4), 0x2367);
		
		if (thr == 0 || thr == 2){
			output[0 + 1] = __shfl(output[0], (threadIdx.x + 2) & 3, 4);
			output[2 + 1] = __shfl(output[2], (threadIdx.x + 2) & 3, 4);
			output[4 + 1] = __shfl(output[4], (threadIdx.x + 2) & 3, 4);
			output[6 + 1] = __shfl(output[6], (threadIdx.x + 2) & 3, 4);
			output[8 + 1] = __shfl(output[8], (threadIdx.x + 2) & 3, 4);
			output[10 + 1] = __shfl(output[10], (threadIdx.x + 2) & 3, 4);
			output[12 + 1] = __shfl(output[12], (threadIdx.x + 2) & 3, 4);
			output[14 + 1] = __shfl(output[14], (threadIdx.x + 2) & 3, 4);		
			if(thr==0){
				outHash[0] = *(uint2x4*)&output[0];
				outHash[1] = *(uint2x4*)&output[8];
			}
		}
#else
		output[ 0] = __byte_perm(output[0], __shfl(output[0], (threadIdx.x + 1) & 3, 4), 0x0167);
		output[ 1] = __shfl(output[0], (threadIdx.x + 2) & 3, 4);

		output[ 2] = __byte_perm(output[2], __shfl(output[2], (threadIdx.x + 1) & 3, 4), 0x0167);
		output[ 3] = __shfl(output[2], (threadIdx.x + 2) & 3, 4);
		
		output[ 4] = __byte_perm(output[4], __shfl(output[4], (threadIdx.x + 1) & 3, 4), 0x2367);
		output[ 5] = __shfl(output[4], (threadIdx.x + 2) & 3, 4);
		
		output[ 6] = __byte_perm(output[6], __shfl(output[6], (threadIdx.x + 1) & 3, 4), 0x2367);
		output[ 7] = __shfl(output[6], (threadIdx.x + 2) & 3, 4);
		
		output[ 8] = __byte_perm(output[8], __shfl(output[8], (threadIdx.x + 1) & 3, 4), 0x0167);
		output[ 9] = __shfl(output[8], (threadIdx.x + 2) & 3, 4);

		output[10] = __byte_perm(output[10], __shfl(output[10], (threadIdx.x + 1) & 3, 4), 0x0167);
		output[11] = __shfl(output[10], (threadIdx.x + 2) & 3, 4);
		
		output[12] = __byte_perm(output[12], __shfl(output[12], (threadIdx.x + 1) & 3, 4), 0x2367);
		output[13] = __shfl(output[12], (threadIdx.x + 2) & 3, 4);
		
		output[14] = __byte_perm(output[14], __shfl(output[14], (threadIdx.x + 1) & 3, 4), 0x2367);
		output[15] = __shfl(output[14], (threadIdx.x + 2) & 3, 4);

		if(thr==0){
			outHash[0] = *(uint2x4*)&output[0];
			outHash[1] = *(uint2x4*)&output[8];
		}
#endif
    	}
}

// Setup Function
__host__
void myriadgroestl_cpu_init(int thr_id, uint32_t threads)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_outputHashes[thr_id], (size_t) 64 * threads));
}

__host__
void myriadgroestl_cpu_free(int thr_id)
{
	cudaFree(d_outputHashes[thr_id]);
}

__host__
void myriadgroestl_cpu_setBlock(int thr_id, void *data){

	uint32_t msgBlock[32] = { 0 };
	uint32_t paddedInput[32];
	memcpy(&msgBlock[0], data, 80);
	msgBlock[20] = 0x80;
	msgBlock[31] = 0x01000000;

	for(int thr=0;thr<4;thr++)
		for(int k=0; k<8; k++)
			paddedInput[k+(thr<<3)] = msgBlock[4*k+thr];

	for(int k=0;k<8;k++){
		uint32_t temp = paddedInput[k+(1<<3)];
		paddedInput[k+(1<<3)] = paddedInput[k+(2<<3)];
		paddedInput[k+(2<<3)] = temp;
	}

	cudaMemcpyToSymbol(c_input, paddedInput, 128);
}

__host__
void myriadgroestl_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_resNounce, const uint64_t target)
{
	// Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
	// mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
	uint32_t tpb = TPB52;
	int dev_id = device_map[thr_id];
	if (device_sm[dev_id] <= 500) tpb = TPB50;	
	const dim3 grid((THF*threads + tpb-1)/tpb);
	const dim3 block(tpb);

	myriadgroestl_gpu_hash_quad <<< grid, block >>> (threads, startNounce, d_outputHashes[thr_id]);

	tpb = (device_sm[dev_id] <= 500) ? 768 : 1024;

	dim3 grid2((threads + tpb - 1) / tpb);
	dim3 block2(tpb);
	
	myriadgroestl_gpu_hash_sha <<< grid2, block2 >>> (threads, startNounce, d_outputHashes[thr_id], d_resNounce, target);

}
