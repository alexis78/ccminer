/**
 * SKEIN512 80 + SHA256 64
 * by tpruvot@github - 2015
 * Merged skein512 80 + sha256 64 (in a single kernel) for SM 5+
 * based on sp and klaus work, adapted by tpruvot to keep skein2 compat
 * 
 * Inherited optimum TPB 1024 from sp and klaus work for 2nd generation maxwell
 * SP 1.5.80 on ASUS Strix 970:        272.0MH/s - 1290MHz / intensity 28
 * SP 1.5.80 on GB windforce 750ti OC:  90.9MH/s - 1320MHz / intensity 23
 *
 * Further improved under CUDA 7.5
 * ASUS Strix 970:        293.5MH/s - 1290MHz / intensity 28
 * GB windforce 750ti OC: 114.1MH/s - 1320MHz / intensity 23
 *
 * Both tests carried out, on 970, with --plimit=180W
 * --------------------------------------------
 *
 * Provos Alexis - 2016
 */

#include "sph/sph_skein.h"

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "skein_header.h"
#include <openssl/sha.h>

/* try 1024 for 970+ */
#define TPB52 1024
#define TPB50 768

#define maxResults 16
/* 16 adapters max */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern "C" void skeincoinhash(void *output, const void *input){
	sph_skein512_context ctx_skein;
	SHA256_CTX sha256;

	uint32_t hash[16];

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, (unsigned char *)hash, 64);
	SHA256_Final((unsigned char *)hash, &sha256);

	memcpy(output, hash, 32);
}

__constant__ uint2 c_message16[2];

__constant__ uint2 _ALIGN(16) c_buffer[56];
__constant__ const uint2 c_t[ 5]   = {{8,0},{0,0xFF000000},{8,0xFF000000},{0x50,0},{0,0xB0000000}};
__constant__ const uint2 c_add[18] = {{1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0},{8,0},{9,0},{10,0},{11,0},{12,0},{13,0},{14,0},{15,0},{16,0},{17,0},{18,0}};
// precomputed tables for SHA256

__constant__ const uint32_t sha256_hashTable[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};
__constant__  uint32_t _ALIGN(16) sha256_endingTable[64] = {
	0xc28a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf374,
	0x649b69c1, 0xf0fe4786, 0x0fe1edc6, 0x240cf254, 0x4fe9346f, 0x6cc984be, 0x61b9411e, 0x16f988fa,
	0xf2c65152, 0xa88e5a6d, 0xb019fc65, 0xb9d99ec7, 0x9a1231c3, 0xe70eeaa0, 0xfdb1232b, 0xc7353eb0,
	0x3069bad5, 0xcb976d5f, 0x5a0f118f, 0xdc1eeefd, 0x0a35b689, 0xde0b7a04, 0x58f4ca9d, 0xe15d5b16,
	0x007f3e86, 0x37088980, 0xa507ea32, 0x6fab9537, 0x17406110, 0x0d8cd6f1, 0xcdaa3b6d, 0xc0bbbe37,
	0x83613bda, 0xdb48a363, 0x0b02e931, 0x6fd15ca7, 0x521afaca, 0x31338431, 0x6ed41a95, 0x6d437890,
	0xc39c91f2, 0x9eccabbd, 0xb5c9a0e6, 0x532fb63c, 0xd2c741c6, 0x07237ea3, 0xa4954b68, 0x4c191d76
};

__constant__ uint32_t _ALIGN(16) sha256_constantTable[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* Elementary defines for SHA256 */
#define xor3b(a,b,c) ((a ^ b) ^ c)

#define R(x, n)       ((x) >> (n))
//#define Maj(x, y, z)  ((x & (y | z)) | (y & z)) //((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);
__device__ __forceinline__
uint32_t Maj(const uint32_t a,const uint32_t b,const uint32_t c){ //Sha256 - Maj - andor
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm ("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(result) : "r"(a), "r"(b),"r"(c)); // 0xE8 = ((0xF0 & (0xCC | 0xAA)) | (0xCC & 0xAA))
	#else
		result = ((a & (b | c)) | (b & c));
	#endif
	return result;	
}

#define Ch(x, y, z)   ((x & (y ^ z)) ^ z)
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

#if __CUDA_ARCH__ <= 500
__global__ __launch_bounds__(TPB50)
#else
__global__ __launch_bounds__(TPB52, 1)
#endif
void skeincoin_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t* resNonce, uint32_t target7)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	
	if (thread < threads)
	{
		uint32_t nonce = startNonce + thread;
		uint2 nonce2 = make_uint2(c_message16[1].x, cuda_swab32(nonce));

		uint2 h[ 9];
		uint2 p[8];

		*(uint2x4*)&h[ 0] = *(uint2x4*)&c_buffer[ 0];
		*(uint2x4*)&h[ 4] = *(uint2x4*)&c_buffer[ 4];
		h[ 8] = c_buffer[ 8];
		
		int i = 9;

		p[ 1] = nonce2+h[ 1];

		p[ 0] = c_buffer[i++]+p[ 1];
		p[ 1] = ROL2(p[ 1],46) ^ p[ 0];
		p[ 2] = c_buffer[i++] + p[ 1];
		p[ 3] = c_buffer[i++];
		p[ 4] = c_buffer[i++];
		p[ 5] = c_buffer[i++];
		p[ 6] = c_buffer[i++];
		p[ 7] = c_buffer[i++];
		p[ 1] = ROL2(p[ 1],33) ^ p[ 2];
		p[ 0]+= p[ 3];
		p[ 3] = c_buffer[i++] ^ p[ 0];
		p[ 4]+=p[ 1];p[ 6]+=p[ 3];p[ 0]+=p[ 5];p[ 2]+=p[ 7];p[ 1]=ROL2(p[ 1],17) ^ p[ 4];p[ 3]=ROL2(p[ 3],49) ^ p[ 6];p[ 5]=c_buffer[i++]  ^ p[ 0];p[ 7]=c_buffer[i++]^ p[ 2];
		p[ 6]+=p[ 1];p[ 0]+=p[ 7];p[ 2]+=p[ 5];p[ 4]+=p[ 3];p[ 1]=ROL2(p[ 1],44) ^ p[ 6];p[ 7]=ROL2(p[ 7], 9) ^ p[ 0];p[ 5]=ROL2(p[ 5],54) ^ p[ 2];p[ 3]=ROR8(p[ 3])  ^ p[ 4];
		
		addwBuff(1,2,3,4,5);TFBIGMIX8o();
		addwBuff(2,3,4,5,6);TFBIGMIX8e();
		addwBuff(3,4,5,6,7);TFBIGMIX8o();
		addwBuff(4,5,6,7,8);TFBIGMIX8e();
		addwBuff(5,6,7,8,0);TFBIGMIX8o();
		addwBuff(6,7,8,0,1);TFBIGMIX8e();
		addwBuff(7,8,0,1,2);TFBIGMIX8o();
		addwBuff(8,0,1,2,3);TFBIGMIX8e();
		addwBuff(0,1,2,3,4);TFBIGMIX8o();
		addwBuff(1,2,3,4,5);TFBIGMIX8e();
		addwBuff(2,3,4,5,6);TFBIGMIX8o();
		addwBuff(3,4,5,6,7);TFBIGMIX8e();
		addwBuff(4,5,6,7,8);TFBIGMIX8o();
		addwBuff(5,6,7,8,0);TFBIGMIX8e();
		addwBuff(6,7,8,0,1);TFBIGMIX8o();
		addwBuff(7,8,0,1,2);TFBIGMIX8e();
		addwBuff(8,0,1,2,3);TFBIGMIX8o();
		
		h[ 0] = c_message16[0] ^ (p[0]+h[ 0]);
		h[ 1] = nonce2 ^ (p[1]+h[ 1]);
		h[ 2]+= p[2];
		h[ 3]+= p[3];
		h[ 4]+= p[4];
		
		h[ 5]+= p[5] + c_t[ 3];//make_uint2(0x50,0);// SPH_T64(bcount << 6) + (sph_u64)(extra);
		h[ 6]+= p[6] + c_t[ 4];//make_uint2(0,0xB0000000);// (bcount >> 58) + ((sph_u64)(etype) << 55);
		h[ 7]+= p[7] + vectorize(18);

		h[ 8] = h[ 0] ^ h[ 1] ^ h[ 2] ^ h[ 3] ^ h[ 4] ^ h[ 5] ^ h[ 6] ^ h[ 7] ^ vectorize(0x1BD11BDAA9FC1A22);
		
		p[ 0] = h[ 0]; p[ 1] = h[ 1]; p[ 2] = h[ 2]; p[ 3] = h[ 3]; p[ 4] = h[ 4]; p[ 5] = h[ 5] + c_t[ 0]; p[ 6] = h[ 6] + c_t[ 1];p[ 7] = h[ 7];
		TFBIGMIX8e();
		addwCon(1,2,3,4,5,6,7,8, 1,2, 0);TFBIGMIX8o();
		addwCon(2,3,4,5,6,7,8,0, 2,0, 1);TFBIGMIX8e();
		addwCon(3,4,5,6,7,8,0,1, 0,1, 2);TFBIGMIX8o();
		addwCon(4,5,6,7,8,0,1,2, 1,2, 3);TFBIGMIX8e();
		addwCon(5,6,7,8,0,1,2,3, 2,0, 4);TFBIGMIX8o();
		addwCon(6,7,8,0,1,2,3,4, 0,1, 5);TFBIGMIX8e();
		addwCon(7,8,0,1,2,3,4,5, 1,2, 6);TFBIGMIX8o();
		addwCon(8,0,1,2,3,4,5,6, 2,0, 7);TFBIGMIX8e();
		addwCon(0,1,2,3,4,5,6,7, 0,1, 8);TFBIGMIX8o();
		addwCon(1,2,3,4,5,6,7,8, 1,2, 9);TFBIGMIX8e();
		addwCon(2,3,4,5,6,7,8,0, 2,0,10);TFBIGMIX8o();
		addwCon(3,4,5,6,7,8,0,1, 0,1,11);TFBIGMIX8e();
		addwCon(4,5,6,7,8,0,1,2, 1,2,12);TFBIGMIX8o();
		addwCon(5,6,7,8,0,1,2,3, 2,0,13);TFBIGMIX8e();
		addwCon(6,7,8,0,1,2,3,4, 0,1,14);TFBIGMIX8o();
		addwCon(7,8,0,1,2,3,4,5, 1,2,15);TFBIGMIX8e();
		addwCon(8,0,1,2,3,4,5,6, 2,0,16);TFBIGMIX8o();
		addwCon(0,1,2,3,4,5,6,7, 0,1,17);

		uint32_t W1[16];
		uint32_t W2[16];
		uint32_t regs[8];
		uint32_t hash[8];

		uint32_t T1;
		
		#pragma unroll 16
		for (int k = 0; k<16; k++)
			W1[k] = cuda_swab32(((uint32_t *)&p)[k]);

		// Init with Hash-Table
		#pragma unroll 8
		for (int k = 0; k < 8; k++) {
			hash[k] = regs[k] = sha256_hashTable[k];
		}

		// Progress W1
		#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			T1 = regs[7] + sha256_constantTable[j] + bsg2_1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + W1[j];
			#pragma unroll
			for (int k = 6; k >= 0; k--) regs[k + 1] = regs[k];
			regs[0] = T1 + bsg2_0(regs[1]) + Maj(regs[1], regs[2], regs[3]);
			regs[4] += T1;
		}

		// Progress W2...W3

		////// PART 1
		#pragma unroll 2
		for (int j = 0; j<2; j++)
			W2[j] = ssg2_1(W1[14 + j]) + W1[9 + j] + ssg2_0(W1[1 + j]) + W1[j];
		#pragma unroll 5
		for (int j = 2; j<7; j++)
			W2[j] = ssg2_1(W2[j - 2]) + W1[9 + j] + ssg2_0(W1[1 + j]) + W1[j];

		#pragma unroll 8
		for (int j = 7; j<15; j++)
			W2[j] = ssg2_1(W2[j - 2]) + W2[j - 7] + ssg2_0(W1[1 + j]) + W1[j];

		W2[15] = ssg2_1(W2[13]) + W2[8] + ssg2_0(W2[0]) + W1[15];

		// Round function
		#pragma unroll
		for (int j = 0; j<16; j++)
		{
			T1 = regs[7] + sha256_constantTable[j + 16] + bsg2_1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + W2[j];
			#pragma unroll
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + bsg2_0(regs[1]) + Maj(regs[1], regs[2], regs[3]);
			regs[4] += T1;
		}

		////// PART 2
		#pragma unroll
		for (int j = 0; j<2; j++)
			W1[j] = ssg2_1(W2[14 + j]) + W2[9 + j] + ssg2_0(W2[1 + j]) + W2[j];

		#pragma unroll
		for (int j = 2; j<7; j++)
			W1[j] = ssg2_1(W1[j - 2]) + W2[9 + j] + ssg2_0(W2[1 + j]) + W2[j];

		#pragma unroll
		for (int j = 7; j<15; j++)
			W1[j] = ssg2_1(W1[j - 2]) + W1[j - 7] + ssg2_0(W2[1 + j]) + W2[j];

		W1[15] = ssg2_1(W1[13]) + W1[8] + ssg2_0(W1[0]) + W2[15];

		// Round function
		#pragma unroll
		for (int j = 0; j<16; j++)
		{
			T1 = regs[7] + sha256_constantTable[j + 32] + bsg2_1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + W1[j];
			#pragma unroll
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + bsg2_0(regs[1]) + Maj(regs[1], regs[2], regs[3]);
			regs[4] += T1;
		}

		////// PART 3
		#pragma unroll
		for (int j = 0; j<2; j++)
			W2[j] = ssg2_1(W1[14 + j]) + W1[9 + j] + ssg2_0(W1[1 + j]) + W1[j];

		#pragma unroll
		for (int j = 2; j<7; j++)
			W2[j] = ssg2_1(W2[j - 2]) + W1[9 + j] + ssg2_0(W1[1 + j]) + W1[j];

		#pragma unroll
		for (int j = 7; j<15; j++)
			W2[j] = ssg2_1(W2[j - 2]) + W2[j - 7] + ssg2_0(W1[1 + j]) + W1[j];

		W2[15] = ssg2_1(W2[13]) + W2[8] + ssg2_0(W2[0]) + W1[15];

		// Round function
		#pragma unroll
		for (int j = 0; j<16; j++)
		{
			T1 = regs[7]  + sha256_constantTable[j + 48] + bsg2_1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + W2[j];
			#pragma unroll
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + bsg2_0(regs[1]) + Maj(regs[1], regs[2], regs[3]);
			regs[4] += T1;
		}

		#pragma unroll
		for (int k = 0; k<8; k++)
			regs[k] = (hash[k] += regs[k]);

		/////
		///// Second Pass (ending)
		/////

		// Progress W1
		#pragma unroll
		for (int j = 0; j<56; j++)//62
		{
			T1 = regs[7]  + sha256_endingTable[j] + bsg2_1(regs[4]) + Ch(regs[4], regs[5], regs[6]);
			#pragma unroll
			for (int k = 6; k >= 0; k--)
				regs[k + 1] = regs[k];
			regs[0] = T1 + bsg2_0(regs[1]) + Maj(regs[1], regs[2], regs[3]);
			regs[4] += T1;
		}
		T1 = regs[7] + sha256_endingTable[56] + bsg2_1(regs[4]) + Ch(regs[4], regs[5], regs[6]);
		regs[7] = T1 + bsg2_0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		regs[3]+= T1;

		T1 = regs[6] + sha256_endingTable[57] + bsg2_1(regs[3]) + Ch(regs[3], regs[4], regs[5]);
		regs[2]+= T1;
		//************
		regs[1]+= regs[5] + sha256_endingTable[58] + bsg2_1(regs[2]) + Ch(regs[2], regs[3], regs[4]);
		regs[0]+= regs[4] + sha256_endingTable[59] + bsg2_1(regs[1]) + Ch(regs[1], regs[2], regs[3]);

		uint32_t test = cuda_swab32(hash[7] + sha256_endingTable[60] + regs[7] + regs[3] + bsg2_1(regs[0]) + Ch(regs[0], regs[1], regs[2]));
		if (test <= target7){
			uint32_t pos = atomicInc(&resNonce[0],0xffffffff)+1;
			if(pos<maxResults)
				resNonce[pos]=nonce;
		}
	}
}

__host__
void skeincoin_setBlock_80(int thr_id, void *pdata)
{
	uint64_t message[16];
	memcpy(&message[0], pdata, 80);

	cudaMemcpyToSymbol(c_message16, &message[8], 16, 0, cudaMemcpyHostToDevice);

	uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
	uint64_t t0, t1, t2;

	h0 = 0x4903ADFF749C51CEull;
	h1 = 0x0D95DE399746DF03ull;
	h2 = 0x8FD1934127C79BCEull;
	h3 = 0x9A255629FF352CB1ull;
	h4 = 0x5DB62599DF6CA7B0ull;
	h5 = 0xEABE394CA9D5C3F4ull;
	h6 = 0x991112C71A75B523ull;
	h7 = 0xAE18A40B660FCC33ull;
	//h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);
	h8 = 0xcab2076d98173ec4ULL;

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	uint64_t p[8];
	for (int i = 0; i<8; i++)
		p[i] = message[i];

	TFBIG_4e_PRE(0);
	TFBIG_4o_PRE(1);
	TFBIG_4e_PRE(2);
	TFBIG_4o_PRE(3);
	TFBIG_4e_PRE(4);
	TFBIG_4o_PRE(5);
	TFBIG_4e_PRE(6);
	TFBIG_4o_PRE(7);
	TFBIG_4e_PRE(8);
	TFBIG_4o_PRE(9);
	TFBIG_4e_PRE(10);
	TFBIG_4o_PRE(11);
	TFBIG_4e_PRE(12);
	TFBIG_4o_PRE(13);
	TFBIG_4e_PRE(14);
	TFBIG_4o_PRE(15);
	TFBIG_4e_PRE(16);
	TFBIG_4o_PRE(17);
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

	h0 = p[ 0] ^ message[ 0];
	h1 = p[ 1] ^ message[ 1];
	h2 = p[ 2] ^ message[ 2];
	h3 = p[ 3] ^ message[ 3];
	h4 = p[ 4] ^ message[ 4];
	h5 = p[ 5] ^ message[ 5];
	h6 = p[ 6] ^ message[ 6];
	h7 = p[ 7] ^ message[ 7];
	h8 = ((h0 ^ h1) ^ (h2 ^ h3)) ^ ((h4 ^ h5) ^ (h6 ^ h7)) ^ SPH_C64(0x1BD11BDAA9FC1A22);
	
	t0 = 0x50ull; // SPH_T64(bcount << 6) + (sph_u64)(extra);
	t1 = 0xB000000000000000ul; // (bcount >> 58) + ((sph_u64)(etype) << 55);
	t2 = 0xB000000000000050ull;
	
	p[ 0] = message[ 8] + h0;
	p[ 2] = h2;
	p[ 3] = h3;
	p[ 4] = h4;
	p[ 5] = h5 + t0;
	p[ 6] = h6 + t1;
	p[ 7] = h7;
	
	p[ 2] += p[ 3];			p[ 3] = ROTL64(p[ 3],36) ^ p[ 2];
	p[ 4] += p[ 5];			p[ 5] = ROTL64(p[ 5],19) ^ p[ 4];
	p[ 6] += p[ 7];			p[ 7] = ROTL64(p[ 7],37) ^ p[ 6];

	p[ 4]+= p[ 7];			p[ 7] = ROTL64(p[ 7],27) ^ p[ 4];
	p[ 6]+= p[ 5];			p[ 5] = ROTL64(p[ 5],14) ^ p[ 6];
	
	uint64_t sk_buf[56];
	int i = 0;
	sk_buf[i++] = h0;
	sk_buf[i++] = h1;
	sk_buf[i++] = h2;
	sk_buf[i++] = h3;
	sk_buf[i++] = h4;
	sk_buf[i++] = h5;
	sk_buf[i++] = h6;
	sk_buf[i++] = h7;
	sk_buf[i++] = h8;
	sk_buf[i++] = p[ 0];//10
	sk_buf[i++] = p[ 2];
	sk_buf[i++] = p[ 3];
	sk_buf[i++] = p[ 4];
	sk_buf[i++] = p[ 5];
	sk_buf[i++] = p[ 6];
	sk_buf[i++] = p[ 7];
	sk_buf[i++] = ROTL64(p[ 3],42);
	sk_buf[i++] = ROTL64(p[ 5],36);
	sk_buf[i++] = ROTL64(p[ 7],39);
	sk_buf[i++] = h6 + t1;//20
	sk_buf[i++] = h8 + 1;
	sk_buf[i++] = h7 + t2;
	sk_buf[i++] = h0 + 2;
	sk_buf[i++] = h8 + t0;
	sk_buf[i++] = h1 + 3;
	sk_buf[i++] = h0 + t1;
	sk_buf[i++] = h2 + 4;
	sk_buf[i++] = h1 + t2;
	sk_buf[i++] = h3 + 5;
	sk_buf[i++] = h2 + t0;
	sk_buf[i++] = h4 + 6;
	sk_buf[i++] = h3 + t1;
	sk_buf[i++] = h5 + 7;
	sk_buf[i++] = h4 + t2;
	sk_buf[i++] = h6 + 8;
	sk_buf[i++] = h5 + t0;
	sk_buf[i++] = h7 + 9;
	sk_buf[i++] = h6 + t1;
	sk_buf[i++] = h8 + 10;
	sk_buf[i++] = h7 + t2;
	sk_buf[i++] = h0 + 11;
	sk_buf[i++] = h8 + t0;
	sk_buf[i++] = h1 + 12;
	sk_buf[i++] = h0 + t1;
	sk_buf[i++] = h2 + 13;
	sk_buf[i++] = h1 + t2;
	sk_buf[i++] = h3 + 14;
	sk_buf[i++] = h2 + t0;
	sk_buf[i++] = h4 + 15;
	sk_buf[i++] = h3 + t1;
	sk_buf[i++] = h5 + 16;
	sk_buf[i++] = h4 + t2;
	sk_buf[i++] = h6 + 17;
	sk_buf[i++] = h5 + t0;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_buffer, sk_buf, sizeof(sk_buf), 0, cudaMemcpyHostToDevice));
	
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_skeincoin(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){

	int dev_id = device_map[thr_id];
	
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];

	int intensity = (device_sm[dev_id] > 500) ? 27 : 23;
	uint32_t throughput = cuda_default_throughput(dev_id, 1UL << intensity);
	if (init[thr_id]) throughput = min(throughput, (max_nonce - first_nonce));

	uint32_t target7 = ptarget[7];

	if (opt_benchmark)
		((uint64_t*)ptarget)[3] = 0;

	if (!init[thr_id]){
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
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

	uint32_t tpb = TPB52;
	if (device_sm[dev_id] <= 500) tpb = TPB50;
	const dim3 grid((throughput + tpb - 1) / tpb);
	const dim3 block(tpb);
	
	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	skeincoin_setBlock_80(thr_id, (void*)endiandata);
	int rc=0;
	cudaMemset(d_resNonce[thr_id], 0x00, maxResults*sizeof(uint32_t));
	do {
		// GPU HASH
		skeincoin_gpu_hash <<< grid, block >>> (throughput, pdata[19], d_resNonce[thr_id], target7);
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
				uint32_t _ALIGN(64) vhash[8];
				be32enc(&endiandata[19],h_resNonce[thr_id][i]);
				skeincoinhash(vhash, endiandata);
				if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
					*hashes_done = pdata[19] - first_nonce + throughput + 1;
					work_set_target_ratio(work, vhash);
					pdata[19] = h_resNonce[thr_id][i];
					rc = 1;
					//check Extranonce
					for(uint32_t j=i+1;j<h_resNonce[thr_id][0]+1;j++){
						be32enc(&endiandata[19],h_resNonce[thr_id][j]);
						skeincoinhash(vhash, endiandata);
						if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
							pdata[21] = h_resNonce[thr_id][j];
//							if(!opt_quiet)
//								gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08X",pdata[21]);
							if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]){
								work_set_target_ratio(work, vhash);
								xchg(pdata[19],pdata[21]);
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
	} while (!work_restart[thr_id].restart && (uint64_t)max_nonce > (uint64_t)throughput + (uint64_t)pdata[19]);

	*hashes_done = pdata[19] - first_nonce + 1;
//	MyStreamSynchronize(NULL, 0, device_map[thr_id]);
	return rc;
}

// cleanup
extern "C" void free_skeincoin(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
