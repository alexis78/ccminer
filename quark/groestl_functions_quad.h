#include "cuda_helper.h"
#include "cuda_vectors.h"

__device__ __forceinline__
static void G256_AddRoundConstantQ_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, const int round){

	const uint32_t andmask = ((-((threadIdx.x & 0x03) == 3)) & 0xffff0000);
	
	x0 = (~x0) ^ ((-(round & 0x01)) & andmask);
	x1 = (~x1) ^ ((-(round & 0x02)) & andmask);
	x2 = (~x2) ^ ((-(round & 0x04)) & andmask);
	x3 = (~x3) ^ ((-(round & 0x08)) & andmask);
	x4 = (~x4) ^ (0xAAAA0000 & andmask);
	x5 = (~x5) ^ (0xCCCC0000 & andmask);
	x6 = (~x6) ^ (0xF0F00000 & andmask);
	x7 = (~x7) ^ (0xFF000000 & andmask);
}

__device__ __forceinline__
static void G256_AddRoundConstantP_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, const int round){

	const uint32_t andmask1 = ((threadIdx.x & 0x03) - 1) >> 16;

	x0 = x0 ^ ((-(round & 0x01)) & andmask1);
	x1 = x1 ^ ((-((round & 0x02) >> 1)) & andmask1);
	x2 = x2 ^ ((-((round & 0x04) >> 2)) & andmask1);
	x3 = x3 ^ ((-((round & 0x08) >> 3)) & andmask1);
	x4 = x4 ^ (0xAAAA & andmask1);
	x5 = x5 ^ (0xCCCC & andmask1);
	x6 = x6 ^ (0xF0F0 & andmask1);
	x7 = x7 ^ (0xFF00 & andmask1);
}

__device__ __forceinline__
static void G16mul_quad(uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0,const uint32_t &y3, const uint32_t &y2, const uint32_t &y1, const uint32_t &y0)
{
    uint32_t t0,t1,t2;
    
    t0 = ((x2 ^ x0) ^ (x3 ^ x1)) & ((y2 ^ y0) ^ (y3 ^ y1));
    t1 = ((x2 ^ x0) & (y2 ^ y0)) ^ t0;
    t2 = ((x3 ^ x1) & (y3 ^ y1)) ^ t0 ^ t1;

    t0 = (x2^x3) & (y2^y3);
    x3 = (x3 & y3) ^ t1 ^ t0;
    x2 = (x2 & y2) ^ t2 ^ t0;

    t0 = (x0^x1) & (y0^y1);
    x1 = (x1 & y1) ^ t1 ^ t0;
    x0 = (x0 & y0) ^ t2 ^ t0;
}

__device__ __forceinline__
static void G256_inv_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0){

    uint32_t t0,t1,t2,t3,t4,t5,t6,a,b;

    t3 = x7;
    t2 = x6;
    t1 = x5;
    t0 = x4;

    G16mul_quad(t3, t2, t1, t0, x3, x2, x1, x0);

    a = (x4 ^ x0);
    t0 ^= a;
    t2 ^= (x7 ^ x3) ^ (x5 ^ x1); 
    t1 ^= (x5 ^ x1) ^ a;
    t3 ^= (x6 ^ x2) ^ a;

    b = t0 ^ t1;
    t4 = (t2 ^ t3) & b;
    a = t4 ^ t3 ^ t1;
    t5 = (t3 & t1) ^ a;
    t6 = (t2 & t0) ^ a ^ (t2 ^ t0);

    t4 = (t5 ^ t6) & b;
    t1 = (t6 & t1) ^ t4;
    t0 = (t5 & t0) ^ t4;

    t4 = (t5 ^ t6) & (t2^t3);
    t3 = (t6 & t3) ^ t4;
    t2 = (t2 & t5) ^ t4;

    G16mul_quad(x3, x2, x1, x0, t1, t0, t3, t2);

    G16mul_quad(x7, x6, x5, x4, t1, t0, t3, t2);
}

__device__
static void transAtoX_quad(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7){

    uint32_t t0, t1;
    t0 = x0 ^ x1 ^ x2;
    t1 = x5 ^ x6;
    x2 = x7 ^ t1 ^ t0;
    x6 = x6 ^ x3 ^ t0;
    x3 = x0 ^ x1 ^ x3 ^ x4 ^ x7;    
    x4 = x0 ^ x4 ^ t1;
    x2 = t0 ^ t1 ^ x7;
    x1 = x0 ^ x1 ^ t1;
    x7 = x0 ^ t1 ^ x7;
    x5 = x0 ^ t1;

}

__device__
static void transXtoA_quad(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7){

    uint32_t t0,t2,t3,t5;

    x1 ^= x4;
    t0 = x1 ^ x6;
    x1 ^= x5;

    t2 = x0 ^ x2;
    x2 = x3 ^ x5;
    t2 ^= x2 ^ x6;
    x2 ^= x7;
    t3 = x4 ^ x2 ^ x6;

    t5 = x0 ^ x6;
    x4 = x3 ^ x7;
    x0 = x3 ^ x5;

    x6 = t0;    
    x3 = t2;
    x7 = t3;    
    x5 = t5;    
}

__device__
static void G256_ShiftBytesP_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0){

	uint32_t t0,t1;

	const uint32_t tpos = threadIdx.x & 0x03;
	const uint32_t shift1 = tpos << 1;
	const uint32_t shift2 = shift1 + 1 + ((tpos == 3) << 2);

	t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
	x0 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
	x1 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
	x2 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
	x3 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
	x4 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
	x5 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
	x6 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
	x7 = __byte_perm(t0, t1, 0x5410);
}

__device__
static void G256_ShiftBytesQ_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0){

	uint32_t t0,t1;

	const uint32_t tpos = threadIdx.x & 0x03;
	const uint32_t shift1 = (1 - (tpos >> 1)) + ((tpos & 0x01) << 2);
	const uint32_t shift2 = shift1 + 2 + ((tpos == 1) << 2);

	t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
	x0 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
	x1 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
	x2 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
	x3 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
	x4 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
	x5 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
	x6 = __byte_perm(t0, t1, 0x5410);

	t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
	t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
	x7 = __byte_perm(t0, t1, 0x5410);
}


__device__ __forceinline__
static void G256_MixFunction_quad(uint32_t *const __restrict__ r){

#define A(idx, l)	__shfl(r[idx], (threadIdx.x+l)&3, 4)
#define S(idx, l)	__byte_perm(A(idx,l),A(idx,l+1),0x5432)


	uint32_t b[8];

#if __CUDA_ARCH__ > 500

	const uint32_t tmp = S(7, 1) ^ S(7, 3) ^ A(7, 2) ^ A(7, 3);

	b[6] = S(6, 1) ^ S(6, 3) ^ A(6, 2) ^ A(6, 3);

	const uint32_t tmp2= b[6] ^ S(7, 3) ^ A(7, 4) ^ S(7, 4) ^ A(7, 5) ^ S(7, 6);

	b[5] = S(5, 1) ^ S(5, 3) ^ A(5, 2) ^ A(5, 3);
	b[6] = b[5] ^ S(6, 3) ^ A(6, 4) ^ S(6, 4) ^ A(6, 5) ^ S(6, 6);
	r[7] = b[6] ^ S(7, 2) ^ A(7, 2) ^ S(7, 3) ^ A(7, 3) ^ A(7, 5);

	b[4] = S(4, 1) ^ S(4, 3) ^ A(4, 2) ^ A(4, 3);
	b[5] = b[4] ^ S(5, 3) ^ A(5, 4) ^ S(5, 4) ^ A(5, 5) ^ S(5, 6);
	r[6] = b[5] ^ S(6, 2) ^ A(6, 2) ^ S(6, 3) ^ A(6, 3) ^ A(6, 5);

	b[3] = S(3, 1) ^ S(3, 3) ^ A(3, 2) ^ A(3, 3);
	b[4] = b[3] ^ S(4, 3) ^ A(4, 4) ^ S(4, 4) ^ A(4, 5) ^ S(4, 6) ^ tmp;
	r[5] = b[4] ^ S(5, 2) ^ A(5, 2) ^ S(5, 3) ^ A(5, 3) ^ A(5, 5);

	b[2] = S(2, 1) ^ S(2, 3) ^ A(2, 2) ^ A(2, 3);
	b[3] = b[2] ^ S(3, 3) ^ A(3, 4) ^ S(3, 4) ^ A(3, 5) ^ S(3, 6) ^ tmp;
	r[4] = b[3] ^ S(4, 2) ^ A(4, 2) ^ S(4, 3) ^ A(4, 3) ^ A(4, 5) ^ tmp2;

	b[1] = S(1, 1) ^ S(1, 3) ^ A(1, 2) ^ A(1, 3);
	b[2] = b[1] ^ S(2, 3) ^ A(2, 4) ^ S(2, 4) ^ A(2, 5) ^ S(2, 6);
	r[3] = b[2] ^ S(3, 2) ^ A(3, 2) ^ S(3, 3) ^ A(3, 3) ^ A(3, 5) ^ tmp2;

	b[0] = S(0, 1) ^ S(0, 3) ^ A(0, 2) ^ A(0, 3);
	b[1] = b[0] ^ S(1, 3) ^ A(1, 4) ^ S(1, 4) ^ A(1, 5) ^ S(1, 6) ^ tmp;
	r[2] = b[1] ^ S(2, 2) ^ A(2, 2) ^ S(2, 3) ^ A(2, 3) ^ A(2, 5);

	b[0] = tmp  ^ S(0, 3) ^ A(0, 4) ^ S(0, 4) ^ A(0, 5) ^ S(0, 6);
	r[1] = b[0] ^ S(1, 2) ^ A(1, 2) ^ S(1, 3) ^ A(1, 3) ^ A(1, 5)^tmp2;

	r[0] = tmp2  ^ S(0, 2) ^ A(0, 2) ^ S(0, 3) ^ A(0, 3) ^ A(0, 5);

#else

	b[0] = S(0, 1) ^ A(0, 2) ^ S(0, 3) ^ A(0, 3);
	b[1] = S(1, 1) ^ A(1, 2) ^ S(1, 3) ^ A(1, 3);
	b[2] = S(2, 1) ^ A(2, 2) ^ S(2, 3) ^ A(2, 3);
	b[3] = S(3, 1) ^ A(3, 2) ^ S(3, 3) ^ A(3, 3);
	b[4] = S(4, 1) ^ A(4, 2) ^ S(4, 3) ^ A(4, 3);
	b[5] = S(5, 1) ^ A(5, 2) ^ S(5, 3) ^ A(5, 3);
	b[6] = S(6, 1) ^ A(6, 2) ^ S(6, 3) ^ A(6, 3);
	b[7] = S(7, 1) ^ A(7, 2) ^ S(7, 3) ^ A(7, 3);

	uint32_t tmp = b[7];
	b[7] = b[6] ^ S(7, 3) ^ A(7, 4) ^ S(7, 4) ^ A(7, 5) ^ S(7, 6);
	b[6] = b[5] ^ S(6, 3) ^ A(6, 4) ^ S(6, 4) ^ A(6, 5) ^ S(6, 6);
	b[5] = b[4] ^ S(5, 3) ^ A(5, 4) ^ S(5, 4) ^ A(5, 5) ^ S(5, 6);
	b[4] = b[3] ^ S(4, 3) ^ A(4, 4) ^ S(4, 4) ^ A(4, 5) ^ S(4, 6) ^ tmp;
	b[3] = b[2] ^ S(3, 3) ^ A(3, 4) ^ S(3, 4) ^ A(3, 5) ^ S(3, 6) ^ tmp;
	b[2] = b[1] ^ S(2, 3) ^ A(2, 4) ^ S(2, 4) ^ A(2, 5) ^ S(2, 6);
	b[1] = b[0] ^ S(1, 3) ^ A(1, 4) ^ S(1, 4) ^ A(1, 5) ^ S(1, 6) ^ tmp;
	b[0] = tmp  ^ S(0, 3) ^ A(0, 4) ^ S(0, 4) ^ A(0, 5) ^ S(0, 6);

	tmp = b[7];
	r[7] = b[6] ^ S(7, 2) ^ A(7, 2) ^ S(7, 3) ^ A(7, 3) ^ A(7, 5);
	r[6] = b[5] ^ S(6, 2) ^ A(6, 2) ^ S(6, 3) ^ A(6, 3) ^ A(6, 5);
	r[5] = b[4] ^ S(5, 2) ^ A(5, 2) ^ S(5, 3) ^ A(5, 3) ^ A(5, 5);
	r[4] = b[3] ^ S(4, 2) ^ A(4, 2) ^ S(4, 3) ^ A(4, 3) ^ A(4, 5) ^ tmp;
	r[3] = b[2] ^ S(3, 2) ^ A(3, 2) ^ S(3, 3) ^ A(3, 3) ^ A(3, 5) ^ tmp;
	r[2] = b[1] ^ S(2, 2) ^ A(2, 2) ^ S(2, 3) ^ A(2, 3) ^ A(2, 5);
	r[1] = b[0] ^ S(1, 2) ^ A(1, 2) ^ S(1, 3) ^ A(1, 3) ^ A(1, 5)^tmp;
	r[0] = tmp  ^ S(0, 2) ^ A(0, 2) ^ S(0, 3) ^ A(0, 3) ^ A(0, 5);

#endif

#undef S
#undef A
}

__device__ __forceinline__
static void sbox_quad(uint32_t *const __restrict__ r){

    transAtoX_quad(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);

    G256_inv_quad(r[2], r[4], r[1], r[7], r[3], r[0], r[5], r[6]);

    transXtoA_quad(r[7], r[1], r[4], r[2], r[6], r[5], r[0], r[3]);
    
    r[0] = r[0] ^ 0xFFFFFFFF;
    r[1] = r[1] ^ 0xFFFFFFFF;
    r[5] = r[5] ^ 0xFFFFFFFF;
    r[6] = r[6] ^ 0xFFFFFFFF;
}

__device__ __forceinline__
static void groestl512_perm_P_quad(uint32_t *const __restrict__ r){

	#if __CUDA_ARCH__ >500
		#pragma unroll 10
	#else
		#pragma unroll 1
	#endif
	for (int round = 0; round<14; round++)
	{
		G256_AddRoundConstantP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
		sbox_quad(r);
		G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
		G256_MixFunction_quad(r);
	}
}

__device__ __forceinline__
static void groestl512_perm_Q_quad(uint32_t *const __restrict__ r){

	#if __CUDA_ARCH__ >500
		#pragma unroll 10
	#else
		#pragma unroll 1
	#endif
	for (int round = 0; round<14; round++)
	{
		G256_AddRoundConstantQ_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
		sbox_quad(r);
		G256_ShiftBytesQ_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
		G256_MixFunction_quad(r);
	}
}

__device__ __forceinline__
static void groestl512_progressMessage_quad(uint32_t *const __restrict__ state, uint32_t *const __restrict__ message, const uint32_t thr){

	((uint8*)state)[0] = ((uint8*)message)[0];

	if (thr == 3) state[ 1] ^= 0x00008000;
	groestl512_perm_P_quad(state);
	if (thr == 3) state[ 1] ^= 0x00008000;
	groestl512_perm_Q_quad(message);
	((uint8*)state)[0] ^= ((uint8*)message)[0];
	((uint8*)message)[0] = ((uint8*)state)[0];

	groestl512_perm_P_quad(message);
		
	((uint8*)state)[0] ^= ((uint8*)message)[0];
}
