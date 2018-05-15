/*
	Based on SP's BMW kernel
	Provos Alexis - 2016
*/

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "cuda_vectors.h"

#define CONST_EXP3d(i)   devectorize(ROL2(q[i+ 1], 5))    + devectorize(ROL2(q[i+ 3],11)) + devectorize(ROL2(q[i+5], 27)) + \
                         devectorize(SWAPDWORDS2(q[i+7])) + devectorize(ROL2(q[i+9], 37)) + devectorize(ROL2(q[i+11],43)) + \
                         devectorize(ROL2(q[i+13],53))    + devectorize(SHR2(q[i+14],1) ^ q[i+14]) + devectorize(SHR2(q[i+15],2) ^ q[i+15])

__device__ __forceinline__
static void bmw512_round1(uint2* q,uint2* h,const uint64_t* msg){
		const uint2 hash[16] =
		{
			{ 0x84858687, 0x80818283 },{ 0x8C8D8E8F, 0x88898A8B },{ 0x94959697, 0x90919293 },{ 0x9C9D9E9F, 0x98999A9B },
			{ 0xA4A5A6A7, 0xA0A1A2A3 },{ 0xACADAEAF, 0xA8A9AAAB },{ 0xB4B5B6B7, 0xB0B1B2B3 },{ 0xBCBDBEBF, 0xB8B9BABB },
			{ 0xC4C5C6C7, 0xC0C1C2C3 },{ 0xCCCDCECF, 0xC8C9CACB },{ 0xD4D5D6D7, 0xD0D1D2D3 },{ 0xDCDDDEDF, 0xD8D9DADB },
			{ 0xE4E5E6E7, 0xE0E1E2E3 },{ 0xECEDEEEF, 0xE8E9EAEB },{ 0xF4F5F6F7, 0xF0F1F2F3 },{ 0xFCFDFEFF, 0xF8F9FAFB }
		};

		const uint64_t hash2[16] =
		{
			0x8081828384858687     ,0x88898A8B8C8D8E8F,0x9091929394959697,0x98999A9B9C9D9E9F,
			0xA0A1A2A3A4A5A6A7     ,0xA8A9AAABACADAEAF,0xB0B1B2B3B4B5B6B7,0xB8B9BABBBCBDBEBF,
			0xC0C1C2C3C4C5C6C7^0x80,0xC8C9CACBCCCDCECF,0xD0D1D2D3D4D5D6D7,0xD8D9DADBDCDDDEDF,
			0xE0E1E2E3E4E5E6E7     ,0xE8E9EAEBECEDEEEF,0xF0F1F2F3F4F5F6F7,0xF8F9FAFBFCFDFEFF
		};
		
		const uint2 precalcf[9] =
		{
			{ 0x55555550, 0x55555555 },{ 0xAAAAAAA5, 0x5AAAAAAA },{ 0xFFFFFFFA, 0x5FFFFFFF },{ 0x5555554F, 0x65555555 },
			{ 0xAAAAAAA4, 0x6AAAAAAA },{ 0xFE00FFF9, 0x6FFFFFFF },{ 0xAAAAAAA1, 0x9AAAAAAA },{ 0xFFFEFFF6, 0x9FFFFFFF },{ 0x5755554B, 0xA5555555 }
		};
				
		uint2 tmp;
		uint64_t mxh[8];
		
		mxh[0] = msg[0] ^ hash2[0];
		mxh[1] = msg[1] ^ hash2[1];
		mxh[2] = msg[2] ^ hash2[2];
		mxh[3] = msg[3] ^ hash2[3];
		mxh[4] = msg[4] ^ hash2[4];
		mxh[5] = msg[5] ^ hash2[5];
		mxh[6] = msg[6] ^ hash2[6];
		mxh[7] = msg[7] ^ hash2[7];
		
		tmp = vectorize(mxh[5] - mxh[7]) + hash[10] + hash[13] + hash[14];
		q[0] = hash[1] + (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37));
		
		tmp = vectorize(mxh[6]) + hash[11] + hash[14] - (hash[15]^512) - (hash[8]^0x80);
		q[1] = hash[2] + (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43));
		
		tmp = vectorize(mxh[0] + mxh[7]) + hash[9] - hash[12] + (hash[15]^0x200);
		q[2] = hash[3] + (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53));
		
		q[16] = (SHR2(q[ 0], 1) ^ SHL2(q[ 0], 2) ^ ROL2(q[ 0],13) ^ ROL2(q[ 0],43)) + (SHR2(q[ 1], 2) ^ SHL2(q[ 1], 1) ^ ROL2(q[ 1],19) ^ ROL2(q[ 1],53));
		q[17] = (SHR2(q[ 1], 1) ^ SHL2(q[ 1], 2) ^ ROL2(q[ 1],13) ^ ROL2(q[ 1],43)) + (SHR2(q[ 2], 2) ^ SHL2(q[ 2], 1) ^ ROL2(q[ 2],19) ^ ROL2(q[ 2],53));
		
		tmp = vectorize((mxh[0] - mxh[1]) + hash2[8] - hash2[10] + hash2[13]);
		q[3] = hash[4] + (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59));
		
		tmp = vectorize((mxh[1] + mxh[2]) + hash2[9] - hash2[11] - hash2[14]);
		q[4] = hash[5] + (SHR2(tmp, 1) ^ tmp);
		
		q[16]+=(SHR2(q[ 2], 2) ^ SHL2(q[ 2], 2) ^ ROL2(q[ 2],28) ^ ROL2(q[ 2],59)) + (SHR2(q[ 3], 1) ^ SHL2(q[ 3], 3) ^ ROL2(q[ 3], 4) ^ ROL2(q[ 3],37));
		q[17]+=(SHR2(q[ 3], 2) ^ SHL2(q[ 3], 2) ^ ROL2(q[ 3],28) ^ ROL2(q[ 3],59)) + (SHR2(q[ 4], 1) ^ SHL2(q[ 4], 3) ^ ROL2(q[ 4], 4) ^ ROL2(q[ 4],37));
		
		tmp = vectorize((mxh[3] - mxh[2] + hash2[10] - hash2[12] + (512 ^ hash2[15])));
		q[5] = hash[6] + (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37));
		
		tmp = vectorize((mxh[4]) - (mxh[0]) - (mxh[3]) + hash2[13] - hash2[11]);
		q[6] = hash[7] + (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43));
		
		q[16]+=(SHR2(q[ 4], 1) ^ SHL2(q[ 4], 2) ^ ROL2(q[ 4],13) ^ ROL2(q[ 4],43)) + (SHR2(q[ 5], 2) ^ SHL2(q[ 5], 1) ^ ROL2(q[ 5],19) ^ ROL2(q[ 5],53));
		q[17]+=(SHR2(q[ 5], 1) ^ SHL2(q[ 5], 2) ^ ROL2(q[ 5],13) ^ ROL2(q[ 5],43)) + (SHR2(q[ 6], 2) ^ SHL2(q[ 6], 1) ^ ROL2(q[ 6],19) ^ ROL2(q[ 6],53));
		
		tmp = vectorize((mxh[1]) - (mxh[4]) - (mxh[5]) - hash2[12] - hash2[14]);
		q[7] = hash[8] + (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53));
		
		tmp = vectorize((mxh[2]) - (mxh[5]) - (mxh[6]) + hash2[13] - (512 ^ hash2[15]));
		q[8] = hash[9] + (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59));
		
		q[16]+=(SHR2(q[ 6], 2) ^ SHL2(q[ 6], 2) ^ ROL2(q[ 6],28) ^ ROL2(q[ 6],59)) + (SHR2(q[ 7], 1) ^ SHL2(q[ 7], 3) ^ ROL2(q[ 7], 4) ^ ROL2(q[ 7],37));
		q[17]+=(SHR2(q[ 7], 2) ^ SHL2(q[ 7], 2) ^ ROL2(q[ 7],28) ^ ROL2(q[ 7],59)) + (SHR2(q[ 8], 1) ^ SHL2(q[ 8], 3) ^ ROL2(q[ 8], 4) ^ ROL2(q[ 8],37));
		
		tmp = vectorize((mxh[0]) - (mxh[3]) + (mxh[6]) - (mxh[7]) + (hash2[14]));
		q[9] = hash[10] + (SHR2(tmp, 1) ^ tmp);
		
		tmp = vectorize((512 ^ hash2[15]) + hash2[8] - (mxh[1]) - (mxh[4]) - (mxh[7]));
		q[10] = hash[11] + (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37));
		
		q[16]+=(SHR2(q[ 8], 1) ^ SHL2(q[ 8], 2) ^ ROL2(q[ 8],13) ^ ROL2(q[ 8],43)) + (SHR2(q[ 9], 2) ^ SHL2(q[ 9], 1) ^ ROL2(q[ 9],19) ^ ROL2(q[ 9],53));
		q[17]+=(SHR2(q[ 9], 1) ^ SHL2(q[ 9], 2) ^ ROL2(q[ 9],13) ^ ROL2(q[ 9],43)) + (SHR2(q[10], 2) ^ SHL2(q[10], 1) ^ ROL2(q[10],19) ^ ROL2(q[10],53));
		
		tmp = vectorize(hash2[9] + hash2[8] - (mxh[0]) - (mxh[2]) - (mxh[5]));
		q[11] = hash[12] + (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp, 43));
		
		tmp = vectorize((mxh[1]) + (mxh[3]) - (mxh[6]) + hash2[10] - hash2[9]);
		q[12] = hash[13] + (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp, 53));
		
		q[16]+=(SHR2(q[10], 2) ^ SHL2(q[10], 2) ^ ROL2(q[10],28) ^ ROL2(q[10],59)) + (SHR2(q[11], 1) ^ SHL2(q[11], 3) ^ ROL2(q[11], 4) ^ ROL2(q[11],37));
		q[17]+=(SHR2(q[11], 2) ^ SHL2(q[11], 2) ^ ROL2(q[11],28) ^ ROL2(q[11],59)) + (SHR2(q[12], 1) ^ SHL2(q[12], 3) ^ ROL2(q[12], 4) ^ ROL2(q[12],37));
		
		tmp = vectorize((mxh[2]) + (mxh[4]) + (mxh[7]) + hash2[10] + hash2[11]);
		q[13] = hash[14] + (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp, 59));
		
		tmp = vectorize((mxh[3]) - (mxh[5]) + hash2[8] - hash2[11] - hash2[12]);
		q[14] = hash[15] + (SHR2(tmp, 1) ^ tmp);
		
		q[16]+=(SHR2(q[12], 1) ^ SHL2(q[12], 2) ^ ROL2(q[12],13) ^ ROL2(q[12],43)) + (SHR2(q[13], 2) ^ SHL2(q[13], 1) ^ ROL2(q[13],19) ^ ROL2(q[13],53));
		q[17]+=(SHR2(q[13], 1) ^ SHL2(q[13], 2) ^ ROL2(q[13],13) ^ ROL2(q[13],43)) + (SHR2(q[14], 2) ^ SHL2(q[14], 1) ^ ROL2(q[14],19) ^ ROL2(q[14],53));
		
		tmp = vectorize(hash2[12] - hash2[9] + hash2[13] - (mxh[4]) - (mxh[6]));
		q[15] = hash[0] + (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp, 37));

		q[16]+= (SHR2(q[14], 2) ^ SHL2(q[14], 2) ^ ROL2(q[14],28) ^ ROL2(q[14],59)) + (SHR2(q[15], 1) ^ SHL2(q[15], 3) ^ ROL2(q[15], 4) ^ ROL2(q[15],37)) +
			((precalcf[0] + ROTL64(msg[0], 1) + ROTL64(msg[ 3], 4)) ^ hash[ 7]);

		q[17]+=
			(SHR2(q[15], 2) ^ SHL2(q[15], 2) ^ ROL2(q[15],28) ^ ROL2(q[15],59)) + (SHR2(q[16], 1) ^ SHL2(q[16], 3) ^ ROL2(q[16], 4) ^ ROL2(q[16],37)) +
			((precalcf[1] + ROTL64(msg[ 1], 2) + ROTL64(msg[ 4], 5)) ^ hash[ 8]);

		uint64_t add1 = devectorize(q[ 2] + q[ 4] + q[ 6] + q[ 8] + q[10] + q[12] + q[14]);
		uint64_t add2 = devectorize(q[ 3] + q[ 5] + q[ 7] + q[ 9] + q[11] + q[13] + q[15]);

		uint2 XL64 = q[16] ^ q[17];
		
		q[18] = vectorize(CONST_EXP3d(2) + add1 + devectorize((precalcf[2] + ROTL64(msg[ 2], 3) + ROTL64(msg[ 5], 6)) ^ hash[ 9]));
		q[19] = vectorize(CONST_EXP3d(3) + add2 + devectorize((precalcf[3] + ROTL64(msg[ 3], 4) + ROTL64(msg[ 6], 7)) ^ hash[10]));
		
		add1+= devectorize(q[16] - q[ 2]);
		add2+= devectorize(q[17] - q[ 3]);

		XL64= xor3x(XL64,q[18],q[19]);

		q[20] = vectorize(CONST_EXP3d(4) + add1 + devectorize((precalcf[ 4] + ROTL64(msg[ 4], 5) + ROTL64(msg[ 7],8)) ^ hash[11]));
		q[21] = vectorize(CONST_EXP3d(5) + add2 + devectorize((precalcf[ 5] + ROTL64(msg[ 5], 6)) ^ hash[5 + 7]));

		add1+= devectorize(q[18] - q[ 4]);
		add2+= devectorize(q[19] - q[ 5]);

		XL64= xor3x(XL64,q[20],q[21]);

		q[22] = vectorize(CONST_EXP3d(6) + add1 + devectorize((vectorize((22)*(0x0555555555555555ull)) + ROTL64(msg[ 6], 7) - ROTL64(msg[ 0], 1)) ^ hash[13]));
		q[23] = vectorize(CONST_EXP3d(7) + add2 + devectorize((vectorize((23)*(0x0555555555555555ull)) + ROTL64(msg[ 7],8)    - ROTL64(msg[ 1], 2)) ^ hash[14]));

		add1+= devectorize(q[20] - q[ 6]);
		add2+= devectorize(q[21] - q[ 7]);

		XL64= xor3x(XL64,q[22],q[23]);

		q[24] = vectorize(CONST_EXP3d(8) + add1 + devectorize((vectorize((24)*(0x0555555555555555ull) + 0x10000) - ROTL64(msg[ 2], 3)) ^ hash[15]));
		q[25] = vectorize(CONST_EXP3d(9) + add2 + devectorize((vectorize((25)*(0x0555555555555555ull)) - ROTL64(msg[3], 4)) ^ hash[0]));

		add1+= devectorize(q[22] - q[ 8]);
		add2+= devectorize(q[23] - q[ 9]);

		uint2 XH64= xor3x(XL64,q[24],q[25]);

		q[26] = vectorize(CONST_EXP3d(10) + add1 + devectorize((vectorize((26)*(0x0555555555555555ull)) - ROTL64(msg[ 4], 5)) ^ hash[ 1]));
		q[27] = vectorize(CONST_EXP3d(11) + add2 + devectorize((vectorize((27)*(0x0555555555555555ull)) - ROTL64(msg[ 5], 6)) ^ hash[ 2]));

		add1+= devectorize(q[24] - q[10]);
		add2+= devectorize(q[25] - q[11]);

		XH64= xor3x(XH64,q[26],q[27]);

		q[28] = vectorize(CONST_EXP3d(12) + add1 + devectorize((vectorize(0x955555555755554C) - ROTL64(msg[ 6], 7)) ^ hash[3]));
		q[29] = vectorize(CONST_EXP3d(13) + add2 + devectorize((precalcf[6] + ROTL64(msg[ 0], 1) - ROTL64(msg[ 7],8)) ^ hash[ 4]));

		add1+= devectorize(q[26] - q[12]);
		add2+= devectorize(q[27] - q[13]);

		XH64= xor3x(XH64,q[28],q[29]);

		q[30] = vectorize(CONST_EXP3d(14) + add1 + devectorize((precalcf[ 7] + ROTL64(msg[ 1], 2)) ^ hash[ 5]));
		q[31] = vectorize(CONST_EXP3d(15) + add2 + devectorize((precalcf[ 8] + ROTL64(msg[ 2], 3)) ^ hash[ 6]));

		XH64= xor3x(XH64,q[30],q[31]);

		h[0] = (SHL2(XH64, 5) ^ SHR2(q[16], 5) ^ vectorize(msg[0])) + (XL64 ^ q[24] ^ q[0]);
		h[1] = (SHR2(XH64, 7) ^ SHL8(q[17])    ^ vectorize(msg[1])) + (XL64 ^ q[25] ^ q[1]);
		h[2] = (SHR2(XH64, 5) ^ SHL2(q[18], 5) ^ vectorize(msg[2])) + (XL64 ^ q[26] ^ q[2]);
		h[3] = (SHR2(XH64, 1) ^ SHL2(q[19], 5) ^ vectorize(msg[3])) + (XL64 ^ q[27] ^ q[3]);
		h[4] = (SHR2(XH64, 3) ^     q[20]      ^ vectorize(msg[4])) + (XL64 ^ q[28] ^ q[4]);
		h[5] = (SHL2(XH64, 6) ^ SHR2(q[21], 6) ^ vectorize(msg[5])) + (XL64 ^ q[29] ^ q[5]);
		h[6] = (SHR2(XH64, 4) ^ SHL2(q[22], 6) ^ vectorize(msg[6])) + (XL64 ^ q[30] ^ q[6]);
		h[7] = (SHR2(XH64,11) ^ SHL2(q[23], 2) ^ vectorize(msg[7])) + (XL64 ^ q[31] ^ q[7]);

		h[ 8] = (ROL2(h[ 4], 9)) + (XH64 ^ q[24] ^ 0x80) + (SHL8(XL64)   ^ q[23] ^ q[ 8]);
		h[ 9] = (ROL2(h[ 5],10)) + (XH64 ^ q[25])        + (SHR2(XL64, 6) ^ q[16] ^ q[ 9]);
		h[10] = (ROL2(h[ 6],11)) + (XH64 ^ q[26])        + (SHL2(XL64, 6) ^ q[17] ^ q[10]);
		h[11] = (ROL2(h[ 7],12)) + (XH64 ^ q[27])        + (SHL2(XL64, 4) ^ q[18] ^ q[11]);
		h[12] = (ROL2(h[ 0],13)) + (XH64 ^ q[28])        + (SHR2(XL64, 3) ^ q[19] ^ q[12]);
		h[13] = (ROL2(h[ 1],14)) + (XH64 ^ q[29])        + (SHR2(XL64, 4) ^ q[20] ^ q[13]);
		h[14] = (ROL2(h[ 2],15)) + (XH64 ^ q[30])        + (SHR2(XL64, 7) ^ q[21] ^ q[14]);
		h[15] = (ROL16(h[ 3]))   + (XH64 ^ q[31] ^ 512)  + (SHR2(XL64, 2) ^ q[22] ^ q[15]);
}

__global__ __launch_bounds__(32,8)
void quark_bmw512_gpu_hash_64(uint32_t threads, uint64_t *const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){

		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];

		uint64_t *inpHash = &g_hash[8 * hashPosition];

		uint64_t msg[16];
		uint2    h[16];

		uint2x4* phash = (uint2x4*)inpHash;
		uint2x4* outpt = (uint2x4*)msg;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		uint2 q[32];

		bmw512_round1(q,h,msg);

		const uint2 cmsg[16] ={
			0xaaaaaaa0, 0xaaaaaaaa,	0xaaaaaaa1, 0xaaaaaaaa,	0xaaaaaaa2, 0xaaaaaaaa,	0xaaaaaaa3, 0xaaaaaaaa,
			0xaaaaaaa4, 0xaaaaaaaa, 0xaaaaaaa5, 0xaaaaaaaa,	0xaaaaaaa6, 0xaaaaaaaa,	0xaaaaaaa7, 0xaaaaaaaa,
			0xaaaaaaa8, 0xaaaaaaaa,	0xaaaaaaa9, 0xaaaaaaaa,	0xaaaaaaaa, 0xaaaaaaaa,	0xaaaaaaab, 0xaaaaaaaa,
			0xaaaaaaac, 0xaaaaaaaa,	0xaaaaaaad, 0xaaaaaaaa,	0xaaaaaaae, 0xaaaaaaaa,	0xaaaaaaaf, 0xaaaaaaaa
		};

		#pragma unroll 16
		for (int i = 0; i < 16; i++)
			msg[i] = devectorize(cmsg[i] ^ h[i]);

		const uint2 precalc[16] = {
			{ 0x55555550, 0x55555555 },{ 0xAAAAAAA5, 0x5AAAAAAA },{ 0xFFFFFFFA, 0x5FFFFFFF },{ 0x5555554F, 0x65555555 },
			{ 0xAAAAAAA4, 0x6AAAAAAA },{ 0xFFFFFFF9, 0x6FFFFFFF },{ 0x5555554E, 0x75555555 },{ 0xAAAAAAA3, 0x7AAAAAAA },
			{ 0xFFFFFFF8, 0x7FFFFFFF },{ 0x5555554D, 0x85555555 },{ 0xAAAAAAA2, 0x8AAAAAAA },{ 0xFFFFFFF7, 0x8FFFFFFF },
			{ 0x5555554C, 0x95555555 },{ 0xAAAAAAA1, 0x9AAAAAAA },{ 0xFFFFFFF6, 0x9FFFFFFF },{ 0x5555554B, 0xA5555555 }
		};

		const uint64_t p2 = msg[15] - msg[12];
		const uint64_t p3 = msg[14] - msg[7];
		const uint64_t p4 = msg[6] + msg[9];
		const uint64_t p5 = msg[8] - msg[5];
		const uint64_t p6 = msg[1] - msg[14];
		const uint64_t p7 = msg[8] - msg[1];
		const uint64_t p8 = msg[3] + msg[10];


		uint2 tmp = vectorize((msg[5]) + (msg[10]) + (msg[13]) + p3);
		q[0] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp,37)) + cmsg[ 1];
		
		tmp = vectorize((msg[6]) - (msg[8]) + (msg[11]) + (msg[14]) - (msg[15]));
		q[1] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp,43)) + cmsg[ 2];
		
		tmp = vectorize((msg[0]) + (msg[7]) + (msg[9]) + p2);
		q[2] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp,53)) + cmsg[ 3];
		
		tmp = vectorize((msg[0]) + p7 - (msg[10]) + (msg[13]));
		q[3] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp,59)) + cmsg[ 4];
		
		tmp = vectorize((msg[2]) + (msg[9]) - (msg[11]) + p6);
		q[4] = (SHR2(tmp, 1) ^ tmp) + cmsg[5];
		
		tmp = vectorize(p8 + p2 - (msg[2]));
		q[5] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp,37)) + cmsg[ 6];
		
		tmp = vectorize((msg[4]) - (msg[0]) - (msg[3]) - (msg[11]) + (msg[13]));
		q[6] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp,43)) + cmsg[ 7];
		
		tmp = vectorize(p6 - (msg[4]) - (msg[5]) - (msg[12]));
		q[7] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp,53)) + cmsg[ 8];
		
		tmp = vectorize((msg[2]) - (msg[5]) - (msg[6]) + (msg[13]) - (msg[15]));
		q[8] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp,59)) + cmsg[ 9];
		
		tmp = vectorize((msg[0]) - (msg[3]) + (msg[6]) + p3);
		q[9] = (SHR2(tmp, 1) ^ tmp) + cmsg[10];
		
		tmp = vectorize(p7 - (msg[4]) - (msg[7]) + (msg[15]));
		q[10] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp,37)) + cmsg[11];
		
		tmp = vectorize(p5 - (msg[0]) - (msg[2]) + (msg[9]));
		q[11] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp,43)) + cmsg[12];
		
		tmp = vectorize(p8 + msg[1] - p4);
		q[12] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp,53)) + cmsg[13];
		
		tmp = vectorize((msg[2]) + (msg[4]) + (msg[7]) + (msg[10]) + (msg[11]));
		q[13] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp,59)) + cmsg[14];
		
		tmp = vectorize((msg[3]) + p5 - (msg[11]) - (msg[12]));
		q[14] = (SHR2(tmp, 1) ^ tmp) + cmsg[15];
		
		tmp = vectorize((msg[12]) - (msg[4]) - p4 + (msg[13]));
		q[15] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp,37)) + cmsg[0];

		q[16] =
			vectorize(devectorize(SHR2(q[ 0], 1) ^ SHL2(q[ 0], 2) ^ ROL2(q[ 0],13) ^ ROL2(q[ 0],43)) + devectorize(SHR2(q[ 1], 2) ^ SHL2(q[ 1], 1) ^ ROL2(q[ 1],19) ^ ROL2(q[ 1],53)) +
			devectorize(SHR2(q[ 2], 2) ^ SHL2(q[ 2], 2) ^ ROL2(q[ 2],28) ^ ROL2(q[ 2],59)) + devectorize(SHR2(q[ 3], 1) ^ SHL2(q[ 3], 3) ^ ROL2(q[ 3], 4) ^ ROL2(q[ 3],37)) +
			devectorize(SHR2(q[ 4], 1) ^ SHL2(q[ 4], 2) ^ ROL2(q[ 4],13) ^ ROL2(q[ 4],43)) + devectorize(SHR2(q[ 5], 2) ^ SHL2(q[ 5], 1) ^ ROL2(q[ 5],19) ^ ROL2(q[ 5],53)) +
			devectorize(SHR2(q[ 6], 2) ^ SHL2(q[ 6], 2) ^ ROL2(q[ 6],28) ^ ROL2(q[ 6],59)) + devectorize(SHR2(q[ 7], 1) ^ SHL2(q[ 7], 3) ^ ROL2(q[ 7], 4) ^ ROL2(q[ 7],37)) +
			devectorize(SHR2(q[ 8], 1) ^ SHL2(q[ 8], 2) ^ ROL2(q[ 8],13) ^ ROL2(q[ 8],43)) + devectorize(SHR2(q[ 9], 2) ^ SHL2(q[ 9], 1) ^ ROL2(q[ 9],19) ^ ROL2(q[ 9],53)) +
			devectorize(SHR2(q[10], 2) ^ SHL2(q[10], 2) ^ ROL2(q[10],28) ^ ROL2(q[10],59)) + devectorize(SHR2(q[11], 1) ^ SHL2(q[11], 3) ^ ROL2(q[11], 4) ^ ROL2(q[11],37)) +
			devectorize(SHR2(q[12], 1) ^ SHL2(q[12], 2) ^ ROL2(q[12],13) ^ ROL2(q[12],43)) + devectorize(SHR2(q[13], 2) ^ SHL2(q[13], 1) ^ ROL2(q[13],19) ^ ROL2(q[13],53)) +
			devectorize(SHR2(q[14], 2) ^ SHL2(q[14], 2) ^ ROL2(q[14],28) ^ ROL2(q[14],59)) + devectorize(SHR2(q[15], 1) ^ SHL2(q[15], 3) ^ ROL2(q[15], 4) ^ ROL2(q[15],37)) +
			devectorize((precalc[0] + ROL2(h[0], 1) + ROL2(h[ 3], 4) - ROL2(h[10],11)) ^ cmsg[ 7]));
		q[17] =
			vectorize(devectorize(SHR2(q[ 1], 1) ^ SHL2(q[ 1], 2) ^ ROL2(q[ 1],13) ^ ROL2(q[ 1],43)) + devectorize(SHR2(q[ 2], 2) ^ SHL2(q[ 2], 1) ^ ROL2(q[ 2],19) ^ ROL2(q[ 2],53)) +
			devectorize(SHR2(q[ 3], 2) ^ SHL2(q[ 3], 2) ^ ROL2(q[ 3],28) ^ ROL2(q[ 3],59)) + devectorize(SHR2(q[ 4], 1) ^ SHL2(q[ 4], 3) ^ ROL2(q[ 4], 4) ^ ROL2(q[ 4],37)) +
			devectorize(SHR2(q[ 5], 1) ^ SHL2(q[ 5], 2) ^ ROL2(q[ 5],13) ^ ROL2(q[ 5],43)) + devectorize(SHR2(q[ 6], 2) ^ SHL2(q[ 6], 1) ^ ROL2(q[ 6],19) ^ ROL2(q[ 6],53)) +
			devectorize(SHR2(q[ 7], 2) ^ SHL2(q[ 7], 2) ^ ROL2(q[ 7],28) ^ ROL2(q[ 7],59)) + devectorize(SHR2(q[ 8], 1) ^ SHL2(q[ 8], 3) ^ ROL2(q[ 8], 4) ^ ROL2(q[ 8],37)) +
			devectorize(SHR2(q[ 9], 1) ^ SHL2(q[ 9], 2) ^ ROL2(q[ 9],13) ^ ROL2(q[ 9],43)) + devectorize(SHR2(q[10], 2) ^ SHL2(q[10], 1) ^ ROL2(q[10],19) ^ ROL2(q[10],53)) +
			devectorize(SHR2(q[11], 2) ^ SHL2(q[11], 2) ^ ROL2(q[11],28) ^ ROL2(q[11],59)) + devectorize(SHR2(q[12], 1) ^ SHL2(q[12], 3) ^ ROL2(q[12], 4) ^ ROL2(q[12],37)) +
			devectorize(SHR2(q[13], 1) ^ SHL2(q[13], 2) ^ ROL2(q[13],13) ^ ROL2(q[13],43)) + devectorize(SHR2(q[14], 2) ^ SHL2(q[14], 1) ^ ROL2(q[14],19) ^ ROL2(q[14],53)) +
			devectorize(SHR2(q[15], 2) ^ SHL2(q[15], 2) ^ ROL2(q[15],28) ^ ROL2(q[15],59)) + devectorize(SHR2(q[16], 1) ^ SHL2(q[16], 3) ^ ROL2(q[16], 4) ^ ROL2(q[16],37)) +
			devectorize((precalc[1] + ROL2(h[ 1], 2) + ROL2(h[ 4], 5) - ROL2(h[11],12)) ^ cmsg[ 8]));

		uint64_t add1 = devectorize(q[ 2] + q[ 4] + q[ 6] + q[ 8] + q[10] + q[12] + q[14]);
		uint64_t add2 = devectorize(q[ 3] + q[ 5] + q[ 7] + q[ 9] + q[11] + q[13] + q[15]);

		uint2 XL64 = q[16] ^ q[17];

		q[18] = vectorize(add1 + CONST_EXP3d(2) + devectorize((precalc[2] + ROL2(h[2], 3) + ROL2(h[ 5], 6) - ROL2(h[12],13)) ^ cmsg[ 9]));
		q[19] = vectorize(add2 + CONST_EXP3d(3) + devectorize((precalc[3] + ROL2(h[3], 4) + ROL2(h[ 6], 7) - ROL2(h[13],14)) ^ cmsg[10]));

		add1 = add1 - devectorize(q[ 2] - q[16]);
		add2 = add2 - devectorize(q[ 3] - q[17]);

		XL64 = xor3x(XL64,q[18],q[19]);

		q[20] = vectorize(add1 + CONST_EXP3d(4) + devectorize((precalc[4] + ROL2(h[4], 5) + ROL8(h[ 7])    - ROL2(h[14],15)) ^ cmsg[11]));
		q[21] = vectorize(add2 + CONST_EXP3d(5) + devectorize((precalc[5] + ROL2(h[5], 6) + ROL2(h[ 8], 9) - ROL16(h[15]))   ^ cmsg[12]));

		add1 = add1 - devectorize(q[ 4] - q[18]);
		add2 = add2 - devectorize(q[ 5] -q[19]);

		XL64 = xor3x(XL64,q[20],q[21]);

		q[22] = vectorize(add1 + CONST_EXP3d(6) + devectorize((precalc[6] + ROL2(h[ 6], 7) + ROL2(h[ 9],10) - ROL2(h[ 0], 1)) ^ cmsg[13]));
		q[23] = vectorize(add2 + CONST_EXP3d(7) + devectorize((precalc[7] + ROL8(h[ 7])    + ROL2(h[10],11) - ROL2(h[ 1], 2)) ^ cmsg[14]));

		add1 -= devectorize(q[ 6] - q[20]);
		add2 -= devectorize(q[ 7] - q[21]);
		
		XL64 = xor3x(XL64,q[22],q[23]);

		q[24] = vectorize(add1 + CONST_EXP3d(8) + devectorize((precalc[8] + ROL2(h[8], 9) + ROL2(h[11],12) - ROL2(h[ 2], 3)) ^ cmsg[15]));
		q[25] = vectorize(add2 + CONST_EXP3d(9) + devectorize((precalc[9] + ROL2(h[9],10) + ROL2(h[12],13) - ROL2(h[ 3], 4)) ^ cmsg[ 0]));

		add1 -= devectorize(q[ 8] - q[22]);
		add2 -= devectorize(q[ 9] - q[23]);

		uint2 XH64 = xor3x(XL64,q[24],q[25]);

		q[26] = vectorize(add1 + CONST_EXP3d(10) + devectorize((precalc[10] + ROL2(h[10],11) + ROL2(h[13],14) - ROL2(h[ 4], 5)) ^ cmsg[ 1]));
		q[27] = vectorize(add2 + CONST_EXP3d(11) + devectorize((precalc[11] + ROL2(h[11],12) + ROL2(h[14],15) - ROL2(h[ 5], 6)) ^ cmsg[ 2]));

		add1 -= devectorize(q[10] - q[24]);
		add2 -= devectorize(q[11] - q[25]);

		XH64 = xor3x(XH64,q[26],q[27]);

		q[28] = vectorize(add1 + CONST_EXP3d(12) + devectorize((precalc[12] + ROL2(h[12], 13) + ROL16(h[15])  - ROL2(h[ 6], 7)) ^ cmsg[ 3]));
		q[29] = vectorize(add2 + CONST_EXP3d(13) + devectorize((precalc[13] + ROL2(h[13], 14) + ROL2(h[ 0], 1)- ROL8(h[ 7]))    ^ cmsg[ 4]));

		add1 -= devectorize(q[12] - q[26]);
		add2 -= devectorize(q[13] - q[27]);

		XH64 = xor3x(XH64,q[28],q[29]);

		q[30] = vectorize(add1 + CONST_EXP3d(14) + devectorize((precalc[14] + ROL2(h[14],15) + ROL2(h[ 1], 2) - ROL2(h[ 8], 9)) ^ cmsg[ 5]));
		q[31] = vectorize(add2 + CONST_EXP3d(15) + devectorize((precalc[15] + ROL16(h[15])   + ROL2(h[ 2], 3) - ROL2(h[ 9],10)) ^ cmsg[ 6]));

		XH64 = xor3x(XH64,q[30],q[31]);

		msg[0] = devectorize((SHL2(XH64, 5) ^ SHR2(q[16], 5) ^ h[ 0]) + (XL64 ^ q[24] ^ q[ 0]));
		msg[1] = devectorize((SHR2(XH64, 7) ^ SHL8(q[17])   ^ h[ 1]) + (XL64 ^ q[25] ^ q[ 1]));
		msg[2] = devectorize((SHR2(XH64, 5) ^ SHL2(q[18], 5) ^ h[ 2]) + (XL64 ^ q[26] ^ q[ 2]));
		msg[3] = devectorize((SHR2(XH64, 1) ^ SHL2(q[19], 5) ^ h[ 3]) + (XL64 ^ q[27] ^ q[ 3]));
		msg[4] = devectorize((SHR2(XH64, 3) ^ q[20] ^ h[4])          + (XL64 ^ q[28] ^ q[ 4]));
		msg[5] = devectorize((SHL2(XH64, 6) ^ SHR2(q[21], 6) ^ h[ 5]) + (XL64 ^ q[29] ^ q[ 5]));
		msg[6] = devectorize((SHR2(XH64, 4) ^ SHL2(q[22], 6) ^ h[ 6]) + (XL64 ^ q[30] ^ q[ 6]));
		msg[7] = devectorize((SHR2(XH64,11) ^ SHL2(q[23], 2) ^ h[ 7]) + (XL64 ^ q[31] ^ q[ 7]));
		msg[8] = devectorize((XH64 ^ q[24] ^ h[8]) + (SHL8(XL64) ^ q[23] ^ q[8]) + ROTL64(msg[4], 9));
		
		msg[ 9] = devectorize((XH64 ^ q[25] ^ h[ 9]) + (SHR2(XL64, 6) ^ q[16] ^ q[9]) + ROTL64(msg[ 5],10));
		msg[10] = devectorize((XH64 ^ q[26] ^ h[10]) + (SHL2(XL64, 6) ^ q[17] ^ q[10]) + ROTL64(msg[ 6],11));
		msg[11] = devectorize((XH64 ^ q[27] ^ h[11]) + (SHL2(XL64, 4) ^ q[18] ^ q[11]) + ROTL64(msg[ 7],12));

		#if __CUDA_ARCH__ > 500
		*(uint2x4*)&inpHash[0] = *(uint2x4*)&msg[8];
		#endif

		msg[12] = devectorize((XH64 ^ q[28] ^ h[12]) + (SHR2(XL64, 3) ^ q[19] ^ q[12]) + ROTL64(msg[ 0],13));
		msg[13] = devectorize((XH64 ^ q[29] ^ h[13]) + (SHR2(XL64, 4) ^ q[20] ^ q[13]) + ROTL64(msg[ 1],14));
		msg[14] = devectorize((XH64 ^ q[30] ^ h[14]) + (SHR2(XL64, 7) ^ q[21] ^ q[14]) + ROTL64(msg[ 2],15));
		msg[15] = devectorize((XH64 ^ q[31] ^ h[15]) + (SHR2(XL64, 2) ^ q[22] ^ q[15]) + ROTL64(msg[ 3],16));

		#if __CUDA_ARCH__ > 500
		*(uint2x4*)&inpHash[4] = *(uint2x4*)&msg[12];
		#else
		*(uint2x4*)&inpHash[0] = *(uint2x4*)&msg[8];			
		*(uint2x4*)&inpHash[4] = *(uint2x4*)&msg[12];
		#endif
	}
}

__global__ __launch_bounds__(32,8)
void quark_bmw512_gpu_hash_64_quark(const uint32_t threads, uint64_t *const __restrict__ g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 *inpHash = (uint2*)&g_hash[thread<<3];

		uint64_t msg[16];
		uint2 h[16];

		uint2x4* phash = (uint2x4*)inpHash;
		uint2x4* outpt = (uint2x4*)msg;		
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		uint2 q[32];

		bmw512_round1(q,h,msg);

		const uint2 cmsg[16] = {
			0xaaaaaaa0, 0xaaaaaaaa,	0xaaaaaaa1, 0xaaaaaaaa,	0xaaaaaaa2, 0xaaaaaaaa,	0xaaaaaaa3, 0xaaaaaaaa,
			0xaaaaaaa4, 0xaaaaaaaa,	0xaaaaaaa5, 0xaaaaaaaa,	0xaaaaaaa6, 0xaaaaaaaa,	0xaaaaaaa7, 0xaaaaaaaa,
			0xaaaaaaa8, 0xaaaaaaaa,	0xaaaaaaa9, 0xaaaaaaaa,	0xaaaaaaaa, 0xaaaaaaaa,	0xaaaaaaab, 0xaaaaaaaa,
			0xaaaaaaac, 0xaaaaaaaa,	0xaaaaaaad, 0xaaaaaaaa,	0xaaaaaaae, 0xaaaaaaaa,	0xaaaaaaaf, 0xaaaaaaaa
		};

		#pragma unroll 16
		for (int i = 0; i < 16; i++) {
			msg[i] = devectorize(cmsg[i] ^ h[i]);
		}

		const uint2 precalc[16] = {
			{ 0x55555550, 0x55555555 },{ 0xAAAAAAA5, 0x5AAAAAAA },{ 0xFFFFFFFA, 0x5FFFFFFF },{ 0x5555554F, 0x65555555 },
			{ 0xAAAAAAA4, 0x6AAAAAAA },{ 0xFFFFFFF9, 0x6FFFFFFF },{ 0x5555554E, 0x75555555 },{ 0xAAAAAAA3, 0x7AAAAAAA },
			{ 0xFFFFFFF8, 0x7FFFFFFF },{ 0x5555554D, 0x85555555 },{ 0xAAAAAAA2, 0x8AAAAAAA },{ 0xFFFFFFF7, 0x8FFFFFFF },
			{ 0x5555554C, 0x95555555 },{ 0xAAAAAAA1, 0x9AAAAAAA },{ 0xFFFFFFF6, 0x9FFFFFFF },{ 0x5555554B, 0xA5555555 },
		};

		const uint64_t p2 = msg[15] - msg[12];
		const uint64_t p3 = msg[14] - msg[7];
		const uint64_t p4 = msg[6] + msg[9];
		const uint64_t p5 = msg[8] - msg[5];
		const uint64_t p6 = msg[1] - msg[14];
		const uint64_t p7 = msg[8] - msg[1];
		const uint64_t p8 = msg[3] + msg[10];


		uint2 tmp = vectorize((msg[5]) + (msg[10]) + (msg[13]) + p3);
		q[0] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp,37)) + cmsg[ 1];
		
		tmp = vectorize((msg[6]) - (msg[8]) + (msg[11]) + (msg[14]) - (msg[15]));
		q[1] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp,43)) + cmsg[ 2];
		
		tmp = vectorize((msg[0]) + (msg[7]) + (msg[9]) + p2);
		q[2] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp,53)) + cmsg[ 3];
		
		tmp = vectorize((msg[0]) + p7 - (msg[10]) + (msg[13]));
		q[3] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp,59)) + cmsg[ 4];
		
		tmp = vectorize((msg[2]) + (msg[9]) - (msg[11]) + p6);
		q[4] = (SHR2(tmp, 1) ^ tmp) + cmsg[5];
		
		tmp = vectorize(p8 + p2 - (msg[2]));
		q[5] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp,37)) + cmsg[ 6];
		
		tmp = vectorize((msg[4]) - (msg[0]) - (msg[3]) - (msg[11]) + (msg[13]));
		q[6] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp,43)) + cmsg[ 7];
		
		tmp = vectorize(p6 - (msg[4]) - (msg[5]) - (msg[12]));
		q[7] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp,53)) + cmsg[ 8];
		
		tmp = vectorize((msg[2]) - (msg[5]) - (msg[6]) + (msg[13]) - (msg[15]));
		q[8] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp,59)) + cmsg[ 9];
		
		tmp = vectorize((msg[0]) - (msg[3]) + (msg[6]) + p3);
		q[9] = (SHR2(tmp, 1) ^ tmp) + cmsg[10];
		
		tmp = vectorize(p7 - (msg[4]) - (msg[7]) + (msg[15]));
		q[10] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp,37)) + cmsg[11];
		
		tmp = vectorize(p5 - (msg[0]) - (msg[2]) + (msg[9]));
		q[11] = (SHR2(tmp, 1) ^ SHL2(tmp, 2) ^ ROL2(tmp, 13) ^ ROL2(tmp,43)) + cmsg[12];
		
		tmp = vectorize(p8 + msg[1] - p4);
		q[12] = (SHR2(tmp, 2) ^ SHL2(tmp, 1) ^ ROL2(tmp, 19) ^ ROL2(tmp,53)) + cmsg[13];
		
		tmp = vectorize((msg[2]) + (msg[4]) + (msg[7]) + (msg[10]) + (msg[11]));
		q[13] = (SHR2(tmp, 2) ^ SHL2(tmp, 2) ^ ROL2(tmp, 28) ^ ROL2(tmp,59)) + cmsg[14];
		
		tmp = vectorize((msg[3]) + p5 - (msg[11]) - (msg[12]));
		q[14] = (SHR2(tmp, 1) ^ tmp) + cmsg[15];
		
		tmp = vectorize((msg[12]) - (msg[4]) - p4 + (msg[13]));
		q[15] = (SHR2(tmp, 1) ^ SHL2(tmp, 3) ^ ROL2(tmp, 4) ^ ROL2(tmp,37)) + cmsg[0];

		q[16] =
			vectorize(devectorize(SHR2(q[ 0], 1) ^ SHL2(q[ 0], 2) ^ ROL2(q[ 0],13) ^ ROL2(q[ 0],43)) + devectorize(SHR2(q[ 1], 2) ^ SHL2(q[ 1], 1) ^ ROL2(q[ 1],19) ^ ROL2(q[ 1],53)) +
			devectorize(SHR2(q[ 2], 2) ^ SHL2(q[ 2], 2) ^ ROL2(q[ 2],28) ^ ROL2(q[ 2],59)) + devectorize(SHR2(q[ 3], 1) ^ SHL2(q[ 3], 3) ^ ROL2(q[ 3], 4) ^ ROL2(q[ 3],37)) +
			devectorize(SHR2(q[ 4], 1) ^ SHL2(q[ 4], 2) ^ ROL2(q[ 4],13) ^ ROL2(q[ 4],43)) + devectorize(SHR2(q[ 5], 2) ^ SHL2(q[ 5], 1) ^ ROL2(q[ 5],19) ^ ROL2(q[ 5],53)) +
			devectorize(SHR2(q[ 6], 2) ^ SHL2(q[ 6], 2) ^ ROL2(q[ 6],28) ^ ROL2(q[ 6],59)) + devectorize(SHR2(q[ 7], 1) ^ SHL2(q[ 7], 3) ^ ROL2(q[ 7], 4) ^ ROL2(q[ 7],37)) +
			devectorize(SHR2(q[ 8], 1) ^ SHL2(q[ 8], 2) ^ ROL2(q[ 8],13) ^ ROL2(q[ 8],43)) + devectorize(SHR2(q[ 9], 2) ^ SHL2(q[ 9], 1) ^ ROL2(q[ 9],19) ^ ROL2(q[ 9],53)) +
			devectorize(SHR2(q[10], 2) ^ SHL2(q[10], 2) ^ ROL2(q[10],28) ^ ROL2(q[10],59)) + devectorize(SHR2(q[11], 1) ^ SHL2(q[11], 3) ^ ROL2(q[11], 4) ^ ROL2(q[11],37)) +
			devectorize(SHR2(q[12], 1) ^ SHL2(q[12], 2) ^ ROL2(q[12],13) ^ ROL2(q[12],43)) + devectorize(SHR2(q[13], 2) ^ SHL2(q[13], 1) ^ ROL2(q[13],19) ^ ROL2(q[13],53)) +
			devectorize(SHR2(q[14], 2) ^ SHL2(q[14], 2) ^ ROL2(q[14],28) ^ ROL2(q[14],59)) + devectorize(SHR2(q[15], 1) ^ SHL2(q[15], 3) ^ ROL2(q[15], 4) ^ ROL2(q[15],37)) +
			devectorize((precalc[0] + ROL2(h[0], 1) + ROL2(h[ 3], 4) - ROL2(h[10],11)) ^ cmsg[ 7]));
		q[17] =
			vectorize(devectorize(SHR2(q[ 1], 1) ^ SHL2(q[ 1], 2) ^ ROL2(q[ 1],13) ^ ROL2(q[ 1],43)) + devectorize(SHR2(q[ 2], 2) ^ SHL2(q[ 2], 1) ^ ROL2(q[ 2],19) ^ ROL2(q[ 2],53)) +
			devectorize(SHR2(q[ 3], 2) ^ SHL2(q[ 3], 2) ^ ROL2(q[ 3],28) ^ ROL2(q[ 3],59)) + devectorize(SHR2(q[ 4], 1) ^ SHL2(q[ 4], 3) ^ ROL2(q[ 4], 4) ^ ROL2(q[ 4],37)) +
			devectorize(SHR2(q[ 5], 1) ^ SHL2(q[ 5], 2) ^ ROL2(q[ 5],13) ^ ROL2(q[ 5],43)) + devectorize(SHR2(q[ 6], 2) ^ SHL2(q[ 6], 1) ^ ROL2(q[ 6],19) ^ ROL2(q[ 6],53)) +
			devectorize(SHR2(q[ 7], 2) ^ SHL2(q[ 7], 2) ^ ROL2(q[ 7],28) ^ ROL2(q[ 7],59)) + devectorize(SHR2(q[ 8], 1) ^ SHL2(q[ 8], 3) ^ ROL2(q[ 8], 4) ^ ROL2(q[ 8],37)) +
			devectorize(SHR2(q[ 9], 1) ^ SHL2(q[ 9], 2) ^ ROL2(q[ 9],13) ^ ROL2(q[ 9],43)) + devectorize(SHR2(q[10], 2) ^ SHL2(q[10], 1) ^ ROL2(q[10],19) ^ ROL2(q[10],53)) +
			devectorize(SHR2(q[11], 2) ^ SHL2(q[11], 2) ^ ROL2(q[11],28) ^ ROL2(q[11],59)) + devectorize(SHR2(q[12], 1) ^ SHL2(q[12], 3) ^ ROL2(q[12], 4) ^ ROL2(q[12],37)) +
			devectorize(SHR2(q[13], 1) ^ SHL2(q[13], 2) ^ ROL2(q[13],13) ^ ROL2(q[13],43)) + devectorize(SHR2(q[14], 2) ^ SHL2(q[14], 1) ^ ROL2(q[14],19) ^ ROL2(q[14],53)) +
			devectorize(SHR2(q[15], 2) ^ SHL2(q[15], 2) ^ ROL2(q[15],28) ^ ROL2(q[15],59)) + devectorize(SHR2(q[16], 1) ^ SHL2(q[16], 3) ^ ROL2(q[16], 4) ^ ROL2(q[16],37)) +
			devectorize((precalc[1] + ROL2(h[ 1], 2) + ROL2(h[ 4], 5) - ROL2(h[11],12)) ^ cmsg[ 8]));

		uint64_t add1 = devectorize(q[ 2] + q[ 4] + q[ 6] + q[ 8] + q[10] + q[12] + q[14]);
		uint64_t add2 = devectorize(q[ 3] + q[ 5] + q[ 7] + q[ 9] + q[11] + q[13] + q[15]);

		uint2 XL64 = q[16] ^ q[17];

		q[18] = vectorize(add1 + CONST_EXP3d(2) + devectorize((precalc[2] + ROL2(h[2], 3) + ROL2(h[ 5], 6) - ROL2(h[12],13)) ^ cmsg[ 9]));
		q[19] = vectorize(add2 + CONST_EXP3d(3) + devectorize((precalc[3] + ROL2(h[3], 4) + ROL2(h[ 6], 7) - ROL2(h[13],14)) ^ cmsg[10]));

		add1+= devectorize(q[16] - q[ 2]);
		add2+= devectorize(q[17] - q[ 3]);

		XL64 = xor3x(XL64, q[18], q[19]);

		q[20] = vectorize(add1 + CONST_EXP3d(4) + devectorize((precalc[4] + ROL2(h[4], 5) + ROL8(h[ 7])    - ROL2(h[14],15)) ^ cmsg[11]));
		q[21] = vectorize(add2 + CONST_EXP3d(5) + devectorize((precalc[5] + ROL2(h[5], 6) + ROL2(h[ 8], 9) - ROL16(h[15]))   ^ cmsg[12]));

		add1+= devectorize(q[18] - q[ 4]);
		add2+= devectorize(q[19] - q[ 5]);

		XL64 = xor3x(XL64, q[20], q[21]);

		q[22] = vectorize(add1 + CONST_EXP3d(6) + devectorize((precalc[6] + ROL2(h[ 6], 7) + ROL2(h[ 9],10) - ROL2(h[ 0], 1)) ^ cmsg[13]));
		q[23] = vectorize(add2 + CONST_EXP3d(7) + devectorize((precalc[7] + ROL8(h[ 7])    + ROL2(h[10],11) - ROL2(h[ 1], 2)) ^ cmsg[14]));

		add1+= devectorize(q[20] - q[ 6]);
		add2+= devectorize(q[21] - q[ 7]);
		
		XL64 = xor3x(XL64, q[22], q[23]);

		q[24] = vectorize(add1 + CONST_EXP3d(8) + devectorize((precalc[8] + ROL2(h[8], 9) + ROL2(h[11],12) - ROL2(h[ 2], 3)) ^ cmsg[15]));
		q[25] = vectorize(add2 + CONST_EXP3d(9) + devectorize((precalc[9] + ROL2(h[9],10) + ROL2(h[12],13) - ROL2(h[ 3], 4)) ^ cmsg[ 0]));

		add1+= devectorize(q[22] - q[ 8]);
		add2+= devectorize(q[23] - q[ 9]);
		
		uint2 XH64 = xor3x(XL64, q[24], q[25]);

		q[26] = vectorize(add1 + CONST_EXP3d(10) + devectorize((precalc[10] + ROL2(h[10],11) + ROL2(h[13],14) - ROL2(h[ 4], 5)) ^ cmsg[ 1]));
		q[27] = vectorize(add2 + CONST_EXP3d(11) + devectorize((precalc[11] + ROL2(h[11],12) + ROL2(h[14],15) - ROL2(h[ 5], 6)) ^ cmsg[ 2]));

		add1+= devectorize(q[24] - q[10]);
		add2+= devectorize(q[25] - q[11]);

		XH64 = xor3x(XH64, q[26], q[27]);

		q[28] = vectorize(add1 + CONST_EXP3d(12) + devectorize((precalc[12] + ROL2(h[12], 13) + ROL16(h[15])  - ROL2(h[ 6], 7)) ^ cmsg[ 3]));
		q[29] = vectorize(add2 + CONST_EXP3d(13) + devectorize((precalc[13] + ROL2(h[13], 14) + ROL2(h[ 0], 1)- ROL8(h[ 7]))    ^ cmsg[ 4]));

		add1+= devectorize(q[26] - q[12]);
		add2+= devectorize(q[27] - q[13]);

		XH64 = xor3x(XH64, q[28], q[29]);

		q[30] = vectorize(add1 + CONST_EXP3d(14) + devectorize((precalc[14] + ROL2(h[14],15) + ROL2(h[ 1], 2) - ROL2(h[ 8], 9)) ^ cmsg[ 5]));
		q[31] = vectorize(add2 + CONST_EXP3d(15) + devectorize((precalc[15] + ROL16(h[15])   + ROL2(h[ 2], 3) - ROL2(h[ 9],10)) ^ cmsg[ 6]));

		XH64 = xor3x(XH64, q[30], q[31]);

		msg[4] = devectorize((SHR2(XH64, 3) ^ q[20] ^ h[4]) + (XL64 ^ q[28] ^ q[4]));
		msg[8] = devectorize((SHL8(XL64) ^ q[23] ^ q[8]) + (XH64 ^ q[24] ^ h[8]) + ROTL64(msg[4], 9));

		inpHash[0].x = (vectorize(msg[8])).x;

		if (!(((vectorize(msg[8])).x) & 0x8)){

			msg[0] = devectorize((h[ 0] ^ SHL2(XH64, 5) ^ SHR2(q[16], 5)) + (XL64 ^ q[ 0] ^ q[24]));
			msg[1] = devectorize((h[ 1] ^ SHR2(XH64, 7) ^ SHL8(q[17])   ) + (XL64 ^ q[ 1] ^ q[25]));
			msg[2] = devectorize((h[ 2] ^ SHR2(XH64, 5) ^ SHL2(q[18], 5)) + (XL64 ^ q[ 2] ^ q[26]));
			msg[3] = devectorize((h[ 3] ^ SHR2(XH64, 1) ^ SHL2(q[19], 5)) + (XL64 ^ q[ 3] ^ q[27]));
			msg[5] = devectorize((h[ 5] ^ SHL2(XH64, 6) ^ SHR2(q[21], 6)) + (XL64 ^ q[ 5] ^ q[29]));
			msg[6] = devectorize((h[ 6] ^ SHR2(XH64, 4) ^ SHL2(q[22], 6)) + (XL64 ^ q[ 6] ^ q[30]));
			msg[7] = devectorize((h[ 7] ^ SHR2(XH64,11) ^ SHL2(q[23], 2)) + (XL64 ^ q[ 7] ^ q[31]));

			msg[ 9] = devectorize((q[ 9] ^ q[16] ^ SHR2(XL64, 6)) + (XH64 ^ q[25] ^ h[ 9]) + ROTL64(msg[ 5],10));
			msg[10] = devectorize((q[10] ^ q[17] ^ SHL2(XL64, 6)) + (XH64 ^ q[26] ^ h[10]) + ROTL64(msg[ 6],11));
			msg[11] = devectorize((q[11] ^ q[18] ^ SHL2(XL64, 4)) + (XH64 ^ q[27] ^ h[11]) + ROTL64(msg[ 7],12));

			#if __CUDA_ARCH__ > 500
			*(uint2x4*)&inpHash[0] = *(uint2x4*)&msg[8];
			#endif
			
			msg[12] = devectorize((q[12] ^ q[19] ^ SHR2(XL64, 3)) + (XH64 ^ q[28] ^ h[12]) + ROTL64(msg[ 0],13));
			msg[13] = devectorize((q[13] ^ q[20] ^ SHR2(XL64, 4)) + (XH64 ^ q[29] ^ h[13]) + ROTL64(msg[ 1],14));
			msg[14] = devectorize((q[14] ^ q[21] ^ SHR2(XL64, 7)) + (XH64 ^ q[30] ^ h[14]) + ROTL64(msg[ 2],15));
			msg[15] = devectorize((q[15] ^ q[22] ^ SHR2(XL64, 2)) + (XH64 ^ q[31] ^ h[15]) + ROTL64(msg[ 3],16));

			#if __CUDA_ARCH__ > 500
			*(uint2x4*)&inpHash[4] = *(uint2x4*)&msg[12];
			#else
			*(uint2x4*)&inpHash[0] = *(uint2x4*)&msg[8];			
			*(uint2x4*)&inpHash[4] = *(uint2x4*)&msg[12];
			#endif
		}
	}
}

__host__
void quark_bmw512_cpu_init(int thr_id, uint32_t threads)
{

}

__host__ void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 32;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    quark_bmw512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash, d_nonceVector);
}
__host__ void quark_bmw512_cpu_hash_64_quark(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 32;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	quark_bmw512_gpu_hash_64_quark <<<grid, block >>>(threads, (uint64_t*)d_hash);
}
