/*
 * Based on:
 * SKEIN512 80 + SKEIN512 64 (Woodcoin)
 * by tpruvot@github - 2015
 *
 * --------------------------------------------
 * Compiled under CUDA7.5 for compute 5.0 - 5.2
 *
 * Previous results:
 * GTX970   - 1239MHz - 179.5MH/s
 * GTX750ti - 1320MHz - 74.5MH/s
 *
 * Changelog:
 * Merged kernels (Higher intensities supported)
 * More precomputations (mainly additions)
 * Fixed uint2 vector addition (cuda_helper.h / generic fix for all hashing functions)
 * Increased register use
 * Increased default intensity
 * 
 * Current results:
 * GTX970   - 1278MHz - 258.5MH/s
 * GTX750ti - 1320MHz - 101.5MH/s
 *
 * Both tests carried out, on 970, with --plimit=180W
 * --------------------------------------------
 *
 * Provos Alexis - 2016
 *
 */
#include <string.h>

#include "sph/sph_skein.h"

#include "miner.h"
#include "cuda_helper.h"
#include "skein_header.h"

#define TPB52 1024
#define TPB50 768

#define NBN 2

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

void skein2hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_skein512_context ctx_skein;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, 64);
	sph_skein512_close(&ctx_skein, hash);

	memcpy(output, (void*) hash, 32);
}

__constant__ uint2 _ALIGN(16) c_buffer[128]; // padded message (80 bytes + 72 bytes midstate + align)

__constant__ const uint2 c_h[ 9] = {
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},
	{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0xA9FC1A22,0x1BD11BDA}
			};

__constant__ const uint2 c_t[ 5] = {	{8,0},{0,0xff000000},{8,0xff000000},{0x50,0},{0,0xB0000000} };

__constant__ const uint64_t c_add[18] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};

__global__
#if __CUDA_ARCH__ <= 500
__launch_bounds__(TPB50, 1)
#else
__launch_bounds__(TPB52, 1)
#endif
static void skein2_512_gpu_hash_80(const uint32_t threads,const uint32_t startNounce, uint32_t *resNonce,const uint64_t highTarget){

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	uint2 h[ 9];
	uint2 p[8];
	if (thread < threads){
	
		// Skein
		#pragma unroll 9
		for(int i=0;i<9;i++){
			h[ i] = c_buffer[ i];
		}
		
		uint2 nonce = make_uint2(c_buffer[ 9].x, startNounce + thread);
		
//		TFBIG_4e_UI2(0);
		p[ 1] = nonce + h[ 1];	p[ 0] = c_buffer[10] + p[ 1];
		p[ 2] = c_buffer[11];	p[ 3] = c_buffer[12];
		p[ 4] = c_buffer[13];	p[ 5] = c_buffer[14];
		p[ 6] = c_buffer[15];	p[ 7] = c_buffer[16];

//		TFBIGMIX8e();
		p[ 1]=ROL2(p[ 1],46) ^ p[ 0];
		
		p[ 2]+=p[ 1];			p[ 0]+=p[ 3];
		p[ 1]=ROL2(p[ 1],33) ^ p[ 2];	p[ 3]=c_buffer[17] ^ p[ 0];

		p[ 4]+=p[ 1];			p[ 6]+=p[ 3];
		p[ 0]+=p[ 5];			p[ 2]+=p[ 7];
		p[ 1]=ROL2(p[ 1],17) ^ p[ 4];	p[ 3]=ROL2(p[ 3],49) ^ p[ 6];
		p[ 5]=c_buffer[18] ^ p[ 0];	p[ 7]=c_buffer[19] ^ p[ 2];

		p[ 6]+=p[ 1];			p[ 0]+=p[ 7];
		p[ 2]+=p[ 5];			p[ 4]+=p[ 3];
		p[ 1]=ROL2(p[ 1],44) ^ p[ 6];	p[ 7]=ROL2(p[ 7], 9) ^ p[ 0];
		p[ 5]=ROL2(p[ 5],54) ^ p[ 2];	p[ 3]=ROR8(p[ 3])    ^ p[ 4];
		
		int i=20;
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
		i++;
		h[ 0] = c_buffer[i++] ^ (p[0]+h[ 0]);	
		h[ 1] = nonce ^ (p[1]+h[ 1]);
		h[ 2]+= p[2];
		h[ 3]+= p[3];
		h[ 4]+= p[4];
		h[ 5]+= p[5] + c_t[ 3];// SPH_T64(bcount << 6) + (sph_u64)(extra);
		h[ 6]+= p[6] + c_t[ 4];//make_uint2(0,0xB0000000);// (bcount >> 58) + ((sph_u64)(etype) << 55);
		h[ 7]+= p[7] + c_add[17];

		h[ 8] = h[ 0] ^ h[ 1] ^ h[ 2] ^ h[ 3] ^ h[ 4] ^ h[ 5] ^ h[ 6] ^ h[ 7] ^ c_h[ 8];
		
		p[ 0] = h[ 0];		p[ 1] = h[ 1];		p[ 2] = h[ 2];		p[ 3] = h[ 3];p[ 4] = h[ 4];		p[ 5] = h[ 5] + c_t[ 0];p[ 6] = h[ 6] + c_t[ 1];p[ 7] = h[ 7];
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

		// Initialisierung
		h[ 0] = c_h[ 0];		h[ 1] = c_h[ 1];		h[ 2] = c_h[ 2];		h[ 3] = c_h[ 3];
		h[ 4] = c_h[ 4];		h[ 5] = c_h[ 5];		h[ 6] = c_h[ 6];		h[ 7] = c_h[ 7];

		uint2 p2[ 8];
		#pragma unroll
		for(int i=0;i<8;i++)
			p2[ i] = p[ i];
		
		h[ 8] = c_buffer[i++];
		
		addwBuff(0,1,2,3,4);TFBIGMIX8e();
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
		addwBuff(0,1,2,3,4);

		h[ 0] = p2[0] ^ p[0];	h[ 1] = p2[1] ^ p[1];	h[ 2] = p2[2] ^ p[2];	h[ 3] = p2[3] ^ p[3];
		h[ 4] = p2[4] ^ p[4];	h[ 5] = p2[5] ^ p[5];	h[ 6] = p2[6] ^ p[6];	h[ 7] = p2[7] ^ p[7];

		h[ 8] = h[ 0] ^ h[ 1] ^ h[ 2] ^ h[ 3] ^ h[ 4] ^ h[ 5] ^ h[ 6] ^ h[ 7] ^ c_h[ 8];

		p[ 0] = h[ 0];		p[ 1] = h[ 1];		p[ 2] = h[ 2];		p[ 3] = h[ 3];	p[ 4] = h[ 4];		p[ 5] = h[ 5] + c_t[ 0];p[ 6] = h[ 6] + c_t[ 1];p[ 7] = h[ 7];
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
		addwCon(8,0,1,2,3,4,5,6, 2,0,16);

//		TFBIGMIX8o();
		p[ 0]+=p[ 1];p[ 2]+=p[ 3];p[ 4]+=p[ 5];p[ 6]+=p[ 7];p[ 1]=ROL2(p[ 1],39) ^ p[ 0];p[ 3]=ROL2(p[ 3],30) ^ p[ 2];p[ 5]=ROL2(p[ 5],34) ^ p[ 4];p[ 7]=ROL24(p[ 7])   ^ p[ 6];
		p[ 2]+=p[ 1];p[ 4]+=p[ 7];p[ 6]+=p[ 5];p[ 0]+=p[ 3];
				
		p[ 1]=ROL2(p[ 1],13) ^ p[ 2];
		p[ 3]=ROL2(p[ 3],17) ^ p[ 0];

		p[ 4]+=p[ 1];
		p[ 6]+=p[ 3];
		
		p[ 3]=ROL2(p[ 3],29) ^ p[ 6];
		
		p[ 4]+=p[ 3];
		p[ 3]=ROL2(p[ 3],22) ^ p[ 4];

		if(devectorize(p[3]+h[ 3])<=highTarget){
			uint32_t tmp = atomicExch(&resNonce[0], startNounce + thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}
__host__
void skein2_512_cpu_setBlock_80(void *pdata)
{
	uint64_t message[20];
	uint64_t p[8];
	uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
	uint64_t t0, t1, t2;

	memcpy(&message[0], pdata, 80);

	h0 = 0x4903ADFF749C51CEull;
	h1 = 0x0D95DE399746DF03ull;
	h2 = 0x8FD1934127C79BCEull;
	h3 = 0x9A255629FF352CB1ull;
	h4 = 0x5DB62599DF6CA7B0ull;
	h5 = 0xEABE394CA9D5C3F4ull;
	h6 = 0x991112C71A75B523ull;
	h7 = 0xAE18A40B660FCC33ull;
	// h[ 8] = h[ 0] ^ h[ 1] ^ h[ 2] ^ h[ 3] ^ h[ 4] ^ h[ 5] ^ h[ 6] ^ h[ 7] ^ SPH_C64(0x1BD11BDAA9FC1A22);
	h8 = 0xcab2076d98173ec4ULL;

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	memcpy(&p[0], &message[0], 64);

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

	h0 = message[0] ^ p[0];
	h1 = message[1] ^ p[1];
	h2 = message[2] ^ p[2];
	h3 = message[3] ^ p[3];
	h4 = message[4] ^ p[4];
	h5 = message[5] ^ p[5];
	h6 = message[6] ^ p[6];
	h7 = message[7] ^ p[7];

	h8 = ((h0 ^ h1) ^ (h2 ^ h3)) ^ ((h4 ^ h5) ^ (h6 ^ h7)) ^ 0x1BD11BDAA9FC1A22;
	
	t0 = 0x50ull;
	t1 = 0xB000000000000000ull;
	t2 = t0^t1;
	
	p[ 0] = message[ 8]+h0;
	p[ 2] = h2;
	p[ 3] = h3;
	p[ 4] = h4;
	p[ 5] = h5 + t0;
	p[ 6] = h6 + t1;
	p[ 7] = h7;

	p[ 2]+=p[ 3];	p[ 3]=ROTL64(p[ 3],36) ^ p[ 2];
	p[ 4]+=p[ 5];	p[ 5]=ROTL64(p[ 5],19) ^ p[ 4];
	p[ 6]+=p[ 7];	p[ 7]=ROTL64(p[ 7],37) ^ p[ 6];
	
	p[ 4]+=p[ 7];	p[ 7]=ROTL64(p[ 7],27) ^ p[ 4];
	p[ 6]+=p[ 5];	p[ 5]=ROTL64(p[ 5],14) ^ p[ 6];
	
	uint64_t buffer[128];
	int i=0;	
	buffer[i++] = h0;
	buffer[i++] = h1;
	buffer[i++] = h2;
	buffer[i++] = h3;
	buffer[i++] = h4;
	buffer[i++] = h5;
	buffer[i++] = h6;
	buffer[i++] = h7;
	buffer[i++] = h8;
	buffer[i++] = message[ 9];
	buffer[i++] = p[ 0];
	buffer[i++] = p[ 2];
	buffer[i++] = p[ 3];
	buffer[i++] = p[ 4];
	buffer[i++] = p[ 5];
	buffer[i++] = p[ 6];
	buffer[i++] = p[ 7];
	buffer[i++] = ROTL64(p[ 3],42);	
	buffer[i++] = ROTL64(p[ 5],36);
	buffer[i++] = ROTL64(p[ 7],39);
	buffer[i++] = h6 + t1;
	buffer[i++] = h8 + 1;
	buffer[i++] = h7 + t2;
	buffer[i++] = h0 + 2;
	buffer[i++] = h8 + t0;
	buffer[i++] = h1 + 3;
	buffer[i++] = h0 + t1;
	buffer[i++] = h2 + 4;
	buffer[i++] = h1 + t2;
	buffer[i++] = h3 + 5;
	buffer[i++] = h2 + t0;
	buffer[i++] = h4 + 6;
	buffer[i++] = h3 + t1;
	buffer[i++] = h5 + 7;
	buffer[i++] = h4 + t2;
	buffer[i++] = h6 + 8;
	buffer[i++] = h5 + t0;
	buffer[i++] = h7 + 9;
	buffer[i++] = h6 + t1;
	buffer[i++] = h8 + 10;
	buffer[i++] = h7 + t2;
	buffer[i++] = h0 + 11;
	buffer[i++] = h8 + t0;
	buffer[i++] = h1 + 12;
	buffer[i++] = h0 + t1;
	buffer[i++] = h2 + 13;
	buffer[i++] = h1 + t2;
	buffer[i++] = h3 + 14;
	buffer[i++] = h2 + t0;
	buffer[i++] = h4 + 15;
	buffer[i++] = h3 + t1;
	buffer[i++] = h5 + 16;
	buffer[i++] = h4 + t2;
	buffer[i++] = h6 + 17;
	buffer[i++] = h5 + t0;
	buffer[i++] = message[ 8];//i=30
	
	h0 = 0x4903ADFF749C51CEull;
	h1 = 0x0D95DE399746DF03ull;
	h2 = 0x8FD1934127C79BCEull;
	h3 = 0x9A255629FF352CB1ull;
	h4 = 0x5DB62599DF6CA7B0ull;
	h5 = 0xEABE394CA9D5C3F4ull;
	h6 = 0x991112C71A75B523ull;
	h7 = 0xAE18A40B660FCC33ull;
	
	t0 = 64; // ptr
	t1 = 0xf000000000000000ULL;// t1 = vectorize(480ull << 55); // etype
	t2 = t0 ^ t1;
	h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ 0x1BD11BDAA9FC1A22;
	
	buffer[i++] = h8;
	buffer[i++] = h5 + t0;
	buffer[i++] = h7;
	buffer[i++] = h6 + t1;
	buffer[i++] = h8 + 1;
	buffer[i++] = h7 + t2;
	buffer[i++] = h0 + 2;
	buffer[i++] = h8 + t0;	
	buffer[i++] = h1 + 3;
	buffer[i++] = h0 + t1;
	buffer[i++] = h2 + 4;
	buffer[i++] = h1 + t2;
	buffer[i++] = h3 + 5;
	buffer[i++] = h2 + t0;
	buffer[i++] = h4 + 6;
	buffer[i++] = h3 + t1;
	buffer[i++] = h5 + 7;
	buffer[i++] = h4 + t2;
	buffer[i++] = h6 + 8;
	buffer[i++] = h5 + t0;
	buffer[i++] = h7 + 9;
	buffer[i++] = h6 + t1;
	buffer[i++] = h8 + 10;
	buffer[i++] = h7 + t2;
	buffer[i++] = h0 + 11;
	buffer[i++] = h8 + t0;
	buffer[i++] = h1 + 12;
	buffer[i++] = h0 + t1;
	buffer[i++] = h2 + 13;
	buffer[i++] = h1 + t2;
	buffer[i++] = h3 + 14;
	buffer[i++] = h2 + t0;
	buffer[i++] = h4 + 15;
	buffer[i++] = h3 + t1;
	buffer[i++] = h5 + 16;
	buffer[i++] = h4 + t2;
	buffer[i++] = h6 + 17;
	buffer[i++] = h5 + t0;
	buffer[i++] = h7 + 18;
	buffer[i++] = h6 + t1;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_buffer, buffer, i*sizeof(uint64_t), 0));
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_skein2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	int dev_id = device_map[thr_id];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	int intensity = (device_sm[dev_id] > 500) ? 27 : 24;
	uint32_t throughput = cuda_default_throughput(dev_id, 1UL << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint64_t*)ptarget)[3] = 0;

	const uint64_t highTarget = *(uint64_t*)&ptarget[6];
	
	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		
		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
		if(h_resNonce[thr_id] == NULL){
			gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
			exit(EXIT_FAILURE);
		}
		init[thr_id] = true;
	}
	uint32_t tpb = TPB52;
	if (device_sm[dev_id] <= 500) tpb = TPB50;
	const dim3 grid((throughput + tpb-1)/tpb);
	const dim3 block(tpb);

	uint32_t endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	skein2_512_cpu_setBlock_80((void*)endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
	int rc=0;
	do {
		// Hash with CUDA
		skein2_512_gpu_hash_80 <<< grid, block >>> (throughput, pdata[19], d_resNonce[thr_id],highTarget);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0] != UINT32_MAX){
			uint32_t _ALIGN(64) vhash64[8];
			endiandata[19] = h_resNonce[thr_id][0];
			skein2hash(vhash64, endiandata);
			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				*hashes_done = pdata[19] - first_nonce + throughput + 1;
				work_set_target_ratio(work, vhash64);
				pdata[19] = swab32(h_resNonce[thr_id][0]);
				rc = 1;
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
//					if(!opt_quiet)
//						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", h_resNonce[thr_id][1]);
					endiandata[19] = h_resNonce[thr_id][1];
					skein2hash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0])
						work_set_target_ratio(work, vhash64);
					pdata[21] = swab32(h_resNonce[thr_id][1]);
					rc = 2;
				}
				return rc;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));				
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && (uint64_t)max_nonce > (uint64_t)throughput + (uint64_t)pdata[19]);

	*hashes_done = pdata[19] - first_nonce + 1;
	return rc;
}

// cleanup
void free_skein2(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
