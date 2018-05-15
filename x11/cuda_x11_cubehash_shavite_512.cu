/*
 * Merged cubehash and Shavite kernels in order to further decrease shared memory bottleneck
 * Built/tested under CUDA7.5 for compute 5.0/5.2
 * Provos Alexis - 2016
 */

#include "cuda_helper.h"
#include "cuda_vectors.h"
#include "cuda_x11_aes.cuh"

#define TPB 128

//--SHAVITE--------------------------------------------
__device__
static void round_3_7_11(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
	KEY_EXPAND_ELT(sharedMemory, &r[ 0]);
	*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
	x = p[ 2] ^ *(uint4*)&r[ 0];
	KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
	r[4] ^= r[0];
	r[5] ^= r[1];
	r[6] ^= r[2];
	r[7] ^= r[3];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x = x ^ *(uint4*)&r[4];
	KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
	r[8] ^= r[4];
	r[9] ^= r[5];
	r[10]^= r[6];
	r[11]^= r[7];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[8];
	x.y ^= r[9];
	x.z ^= r[10];
	x.w ^= r[11];
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	r[12] ^= r[8];
	r[13] ^= r[9];
	r[14]^= r[10];
	r[15]^= r[11];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[12];
	x.y ^= r[13];
	x.z ^= r[14];
	x.w ^= r[15];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[ 1].x ^= x.x;
	p[ 1].y ^= x.y;
	p[ 1].z ^= x.z;
	p[ 1].w ^= x.w;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[ 0] ^ *(uint4*)&r[16];
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[24];
	KEY_EXPAND_ELT(sharedMemory,&r[28]);
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[ 3] ^= x;
}

__device__
static void round_4_8_12(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
	*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
	x = p[ 1] ^ *(uint4*)&r[ 0];
	AES_ROUND_NOKEY(sharedMemory, &x);

	r[ 4] ^= r[29];	r[ 5] ^= r[30];
	r[ 6] ^= r[31];	r[ 7] ^= r[ 0];

	x ^= *(uint4*)&r[ 4];
	*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[ 8];
	*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[ 0] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
	x = p[ 3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[ 2] ^= x;
}

//--END OF SHAVITE MACROS------------------------------------

//--CUBEHASH512 MACROS----------------------------------

#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }

__device__
unsigned int rotate(const unsigned int val, const unsigned int shift)
{
    unsigned int ret;
    asm ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(val), "r"(shift));
    return ret;
}

__device__
static void rrounds(uint32_t *x){
	#pragma unroll 1
	for (uint32_t r = 0; r < 16; r++) {
		/* "add x_0jklm into x_1jklmn modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = rotate(x[ 0], 7);x[17] = x[17] + x[ 1];x[ 1] = rotate(x[ 1], 7);
		x[18] = x[18] + x[ 2]; x[ 2] = rotate(x[ 2], 7);x[19] = x[19] + x[ 3];x[ 3] = rotate(x[ 3], 7);
		x[20] = x[20] + x[ 4]; x[ 4] = rotate(x[ 4], 7);x[21] = x[21] + x[ 5];x[ 5] = rotate(x[ 5], 7);
		x[22] = x[22] + x[ 6]; x[ 6] = rotate(x[ 6], 7);x[23] = x[23] + x[ 7];x[ 7] = rotate(x[ 7], 7);
		x[24] = x[24] + x[ 8]; x[ 8] = rotate(x[ 8], 7);x[25] = x[25] + x[ 9];x[ 9] = rotate(x[ 9], 7);
		x[26] = x[26] + x[10]; x[10] = rotate(x[10], 7);x[27] = x[27] + x[11];x[11] = rotate(x[11], 7);
		x[28] = x[28] + x[12]; x[12] = rotate(x[12], 7);x[29] = x[29] + x[13];x[13] = rotate(x[13], 7);
		x[30] = x[30] + x[14]; x[14] = rotate(x[14], 7);x[31] = x[31] + x[15];x[15] = rotate(x[15], 7);
		/* "swap x_00klm with x_01klm" */
		SWAP(x[ 0], x[ 8]);x[ 0] ^= x[16];x[ 8] ^= x[24];SWAP(x[ 1], x[ 9]);x[ 1] ^= x[17];x[ 9] ^= x[25];
		SWAP(x[ 2], x[10]);x[ 2] ^= x[18];x[10] ^= x[26];SWAP(x[ 3], x[11]);x[ 3] ^= x[19];x[11] ^= x[27];
		SWAP(x[ 4], x[12]);x[ 4] ^= x[20];x[12] ^= x[28];SWAP(x[ 5], x[13]);x[ 5] ^= x[21];x[13] ^= x[29];
		SWAP(x[ 6], x[14]);x[ 6] ^= x[22];x[14] ^= x[30];SWAP(x[ 7], x[15]);x[ 7] ^= x[23];x[15] ^= x[31];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[16], x[18]); SWAP(x[17], x[19]); SWAP(x[20], x[22]); SWAP(x[21], x[23]);SWAP(x[24], x[26]); SWAP(x[25], x[27]); SWAP(x[28], x[30]); SWAP(x[29], x[31]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = rotate(x[ 0],11);x[17] = x[17] + x[ 1];x[ 1] = rotate(x[ 1],11);
		x[18] = x[18] + x[ 2]; x[ 2] = rotate(x[ 2],11);x[19] = x[19] + x[ 3];x[ 3] = rotate(x[ 3],11);
		x[20] = x[20] + x[ 4]; x[ 4] = rotate(x[ 4],11);x[21] = x[21] + x[ 5];x[ 5] = rotate(x[ 5],11);
		x[22] = x[22] + x[ 6]; x[ 6] = rotate(x[ 6],11);x[23] = x[23] + x[ 7];x[ 7] = rotate(x[ 7],11);
		x[24] = x[24] + x[ 8]; x[ 8] = rotate(x[ 8],11);x[25] = x[25] + x[ 9];x[ 9] = rotate(x[ 9],11);
		x[26] = x[26] + x[10]; x[10] = rotate(x[10],11);x[27] = x[27] + x[11];x[11] = rotate(x[11],11);
		x[28] = x[28] + x[12]; x[12] = rotate(x[12],11);x[29] = x[29] + x[13];x[13] = rotate(x[13],11);
		x[30] = x[30] + x[14]; x[14] = rotate(x[14],11);x[31] = x[31] + x[15];x[15] = rotate(x[15],11);
		/* "swap x_0j0lm with x_0j1lm" */
		SWAP(x[ 0], x[ 4]); x[ 0] ^= x[16]; x[ 4] ^= x[20]; SWAP(x[ 1], x[ 5]); x[ 1] ^= x[17]; x[ 5] ^= x[21];
		SWAP(x[ 2], x[ 6]); x[ 2] ^= x[18]; x[ 6] ^= x[22]; SWAP(x[ 3], x[ 7]); x[ 3] ^= x[19]; x[ 7] ^= x[23];
		SWAP(x[ 8], x[12]); x[ 8] ^= x[24]; x[12] ^= x[28]; SWAP(x[ 9], x[13]); x[ 9] ^= x[25]; x[13] ^= x[29];
		SWAP(x[10], x[14]); x[10] ^= x[26]; x[14] ^= x[30]; SWAP(x[11], x[15]); x[11] ^= x[27]; x[15] ^= x[31];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[16], x[17]); SWAP(x[18], x[19]); SWAP(x[20], x[21]); SWAP(x[22], x[23]);SWAP(x[24], x[25]); SWAP(x[26], x[27]); SWAP(x[28], x[29]); SWAP(x[30], x[31]);
	}
}

//--END OF CUBEHASH512 MACROS----------------------------------

__global__
__launch_bounds__(TPB,3)
void x11_cubehashShavite512_gpu_hash_64(uint32_t threads, uint32_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	
	__shared__ uint32_t sharedMemory[4][256];

	aes_gpu_init128(sharedMemory);
	__syncthreads();
//	if (thread < threads)
//	{
		uint32_t *const hash = &g_hash[thread * 16U];

		//Cubehash

		uint32_t x[32] = {
			0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,	0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
			0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,	0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
			0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,	0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
			0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,	0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
		};
		uint32_t Hash[16];
		*(uint2x4*)&Hash[0] = __ldg4((uint2x4*)&hash[0]);
		*(uint2x4*)&Hash[8] = __ldg4((uint2x4*)&hash[8]);

		*(uint2x4*)&x[ 0] ^= *(uint2x4*)&Hash[0];

		rrounds(x);

		*(uint2x4*)&x[ 0] ^= *(uint2x4*)&Hash[8];

		rrounds(x);
		x[0] ^= 0x80;

		rrounds(x);
		x[31] ^= 1;

//		#pragma unroll 10
		for (int i = 0;i < 9;++i)
			rrounds(x);


		rrounds(x);
		
		uint4 y;
		uint32_t r[32];
		uint4 msg[ 4];
		// kopiere init-state
		uint4 p[ 4];
		const uint32_t state[16] = {
			0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC,	0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
			0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47,	0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
		};
		*(uint2x4*)&p[ 0] = *(uint2x4*)&state[ 0];
		*(uint2x4*)&p[ 2] = *(uint2x4*)&state[ 8];

		#pragma unroll 4
		for(int i=0;i<4;i++){
			*(uint4*)&msg[ i] = *(uint4*)&x[i<<2];
			*(uint4*)&r[i<<2] = *(uint4*)&x[i<<2];
		}
		r[16] = 0x80; r[17] = 0; r[18] = 0; r[19] = 0;
		r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 0;
		r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02000000;
		r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;
		y = p[ 1] ^ msg[ 0];
		AES_ROUND_NOKEY(sharedMemory, &y);
		y ^= msg[ 1];
		AES_ROUND_NOKEY(sharedMemory, &y);
		y ^= msg[ 2];
		AES_ROUND_NOKEY(sharedMemory, &y);
		y ^= msg[ 3];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 0] ^= y;
		y = p[ 3];
		y.x ^= 0x80;
		AES_ROUND_NOKEY(sharedMemory, &y);
		AES_ROUND_NOKEY(sharedMemory, &y);
		y.w ^= 0x02000000;
		AES_ROUND_NOKEY(sharedMemory, &y);
		y.w ^= 0x02000000;
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 2]^= y;

		// 1
		KEY_EXPAND_ELT(sharedMemory, &r[ 0]);
		*(uint4*)&r[ 0]^=*(uint4*)&r[28];
		r[ 0] ^= 0x200;
		r[ 3] ^= 0xFFFFFFFF;
		y = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		y ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		y ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		y ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 3] ^= y;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		y = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		y ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		y ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		y ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 1] ^= y;
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		y = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &y);

		r[ 4] ^= r[29]; r[ 5] ^= r[30];
		r[ 6] ^= r[31]; r[ 7] ^= r[ 0];

		y ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		y ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		y ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 2] ^= y;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		y = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		y ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		y ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		y ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &y);

		p[ 0] ^= y;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,y);
		
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,y);

		// 2
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		y = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		r[ 7] ^= (~0x200);
		y ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		y ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		y ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 3] ^= y;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		y = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		y ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		y ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		y ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 1] ^= y;
	
		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		y = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &y);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		y ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		y ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		y ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 2] ^= y;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		y = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		y ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		y ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		y ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 0] ^= y;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,y);

		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,y);

		// 3
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		y = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		y ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		y ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		y ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 3] ^= y;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		y = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		y^=*(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24]^=*(uint4*)&r[20];
		y ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		r[30] ^= 0x200;
		r[31] ^= 0xFFFFFFFF;
		y ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 1] ^= y;

		*(uint4*)&r[ 0] ^= *(uint4*)&r[25];
		y = p[ 3] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &y);
		r[ 4] ^= r[29];
		r[ 5] ^= r[30];
		r[ 6] ^= r[31];
		r[ 7] ^= r[ 0];
		y ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 1];
		y ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 5];
		y ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 2] ^= y;
		*(uint4*)&r[16] ^= *(uint4*)&r[ 9];
		y = p[ 1] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[20] ^= *(uint4*)&r[13];
		y ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[24] ^= *(uint4*)&r[17];
		y ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &y);
		*(uint4*)&r[28] ^= *(uint4*)&r[21];
		y ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 0] ^= y;

		/* round 3, 7, 11 */
		round_3_7_11(sharedMemory,r,p,y);
		
		/* round 4, 8, 12 */
		round_4_8_12(sharedMemory,r,p,y);

		/* round 13 */
		KEY_EXPAND_ELT(sharedMemory,&r[ 0]);
		*(uint4*)&r[ 0] ^= *(uint4*)&r[28];
		y = p[ 0] ^ *(uint4*)&r[ 0];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[ 4]);
		*(uint4*)&r[ 4] ^= *(uint4*)&r[ 0];
		y ^= *(uint4*)&r[ 4];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[ 8]);
		*(uint4*)&r[ 8] ^= *(uint4*)&r[ 4];
		y ^= *(uint4*)&r[ 8];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[12]);
		*(uint4*)&r[12] ^= *(uint4*)&r[ 8];
		y ^= *(uint4*)&r[12];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 3] ^= y;
		KEY_EXPAND_ELT(sharedMemory, &r[16]);
		*(uint4*)&r[16] ^= *(uint4*)&r[12];
		y = p[ 2] ^ *(uint4*)&r[16];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[20]);
		*(uint4*)&r[20] ^= *(uint4*)&r[16];
		y ^= *(uint4*)&r[20];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory, &r[24]);
		*(uint4*)&r[24] ^= *(uint4*)&r[20];
		r[25] ^= 0x200;
		r[27] ^= 0xFFFFFFFF;
		y ^= *(uint4*)&r[24];
		AES_ROUND_NOKEY(sharedMemory, &y);
		KEY_EXPAND_ELT(sharedMemory,&r[28]);
		*(uint4*)&r[28] ^= *(uint4*)&r[24];
		y ^= *(uint4*)&r[28];
		AES_ROUND_NOKEY(sharedMemory, &y);
		p[ 1] ^= y;

		*(uint2x4*)&hash[ 0] = *(uint2x4*)&state[ 0] ^ *(uint2x4*)&p[ 2];
		*(uint2x4*)&hash[ 8] = *(uint2x4*)&state[ 8] ^ *(uint2x4*)&p[ 0];
//	}
}

__host__
void x11_cubehash_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	x11_cubehashShavite512_gpu_hash_64 <<<grid, block>>> (threads, d_hash);
}
