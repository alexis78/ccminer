static __constant__ const uint8_t c_perm[8][8] = {
	{ 2, 3, 6, 7, 0, 1, 4, 5 },{ 6, 7, 2, 3, 4, 5, 0, 1 },{ 7, 6, 5, 4, 3, 2, 1, 0 },{ 1, 0, 3, 2, 5, 4, 7, 6 },
	{ 0, 1, 4, 5, 6, 7, 2, 3 },{ 6, 7, 2, 3, 0, 1, 4, 5 },{ 6, 7, 0, 1, 4, 5, 2, 3 },{ 4, 5, 2, 3, 6, 7, 0, 1 }
};

static __constant__ const uint32_t c_IV_512[32] = {
	0x0ba16b95, 0x72f999ad, 0x9fecc2ae, 0xba3264fc, 0x5e894929, 0x8e9f30e5, 0x2f1daa37, 0xf0f2c558,
	0xac506643, 0xa90635a5, 0xe25b878b, 0xaab7878f, 0x88817f7a, 0x0a02892b, 0x559a7550, 0x598f657e,
	0x7eef60a1, 0x6b70e3e8, 0x9c1714d1, 0xb958e2a8, 0xab02675e, 0xed1c014f, 0xcd8d65bb, 0xfdb7a257,
	0x09254899, 0xd699c7bc, 0x9019b6dc, 0x2b9022e4, 0x8fa14956, 0x21bf9bd3, 0xb94d0943, 0x6ffddc22
};

static __constant__ const int16_t c_FFT128_8_16_Twiddle[128] = {
	1,   1,   1,   1,   1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,  60,   2, 120,   4,  -17,   8, -34,  16, -68,  32, 121,  64, -15, 128, -30,
	1,  46,  60, -67,   2,   92, 120, 123,   4, -73, -17, -11,   8, 111, -34, -22, 1, -67, 120, -73,   8,  -22, -68, -70,  64,  81, -30, -46,  -2,-123,  17,-111,
	1,-118,  46, -31,  60,  116, -67, -61,   2,  21,  92, -62, 120, -25, 123,-122, 1, 116,  92,-122, -17,   84, -22,  18,  32, 114, 117, -49, -30, 118,  67,  62,
	1, -31, -67,  21, 120, -122, -73, -50,   8,   9, -22, -89, -68,  52, -70, 114, 1, -61, 123, -50, -34,   18, -70, -99, 128, -98,  67,  25,  17,  -9,  35, -79
};

static __constant__ const int16_t c_FFT256_2_128_Twiddle[128] = {
	  1,  41,-118,  45,  46,  87, -31,  14, 60,-110, 116,-127, -67,  80, -61,  69,  2,  82,  21,  90,  92, -83, -62,  28,120,  37, -25,   3, 123, -97,-122,-119,
	  4, -93,  42, -77, -73,  91,-124,  56,-17,  74, -50,   6, -11,  63,  13,  19,  8,  71,  84, 103, 111, -75,   9, 112,-34,-109,-100,  12, -22, 126,  26,  38,
	 16,-115, -89, -51, -35, 107,  18, -33,-68,  39,  57,  24, -44,  -5,  52,  76, 32,  27,  79,-102, -70, -43,  36, -66,121,  78, 114,  48, -88, -10, 104,-105,
	 64,  54, -99,  53, 117, -86,  72, 125,-15,-101, -29,  96,  81, -20, -49,  47,128, 108,  59, 106, -23,  85,-113,  -7,-30,  55, -58, -65, -95, -40, -98,  94
};

__device__ __forceinline__
static uint32_t IF(uint32_t x,uint32_t y,uint32_t z){
/*
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		uint32_t result;
		asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(result)	: "r"(x), "r"(y), "r"(z));	// x=F0, y=CC, z=AA // 0xCA = ((CC⊻AA)∧F0)⊻AA
		return result;
	#else
*/		return (((y ^ z) & x) ^ z);
//	#endif
}


__device__ __forceinline__
static uint32_t MAJ(const uint32_t x,const uint32_t y,const uint32_t z){

	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		uint32_t result;
		asm ("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(result)	: "r"(x), "r"(y), "r"(z));	// x=AA, y=CC, z=F0 // 0xCA = ((CC⊻AA)∧F0)⊻AA
		return result;
	#else
		return ((z &y) | ((z|y) & x));
	#endif
}

#define p8_xor(x) ( ((x)%7) == 0 ? 1 : \
	((x)%7) == 1 ? 6 : \
	((x)%7) == 2 ? 2 : \
	((x)%7) == 3 ? 3 : \
	((x)%7) == 4 ? 5 : \
	((x)%7) == 5 ? 7 : 4 )

__device__ __forceinline__
static void STEP8_IF(const uint32_t *w, const uint32_t i, const uint32_t r, const uint32_t s, uint32_t *A, const uint32_t *B, const uint32_t *C, uint32_t *D)
{
	uint32_t R[8];

	#pragma unroll 8
	for(int j=0; j<8; j++)
		R[j] = ROTL32(A[j], r);

	uint32_t W[8];
	*(uint2x4*)&W[0] = *(uint2x4*)&w[0];
	#pragma unroll 8
	for(int j=0; j<8; j++)
		D[j]+= W[j] + IF(A[j], B[j], C[j]);
	#pragma unroll 8
	for(int j=0; j<8; j++)
		D[j] = R[j^p8_xor(i)] + ROTL32(D[j], s);
	#pragma unroll 8
	for(int j=0; j<8; j++)
		A[j] = R[j];
}

__device__ __forceinline__
static void STEP8_MAJ(const uint32_t *w, const uint32_t i, const uint32_t r, const uint32_t s, uint32_t *A, const uint32_t *B, const uint32_t *C, uint32_t *D)
{
	uint32_t R[8];

	uint32_t W[8];
	*(uint2x4*)&W[0] = *(uint2x4*)&w[0];
	
	#pragma unroll 8
	for(int j=0; j<8; j++)
		R[j] = ROTL32(A[j], r);

	#pragma unroll 8
	for(int j=0; j<8; j++)
		D[j]+= W[j] + MAJ(A[j], B[j], C[j]);
	#pragma unroll 8
	for(int j=0; j<8; j++)
		D[j] = R[j^p8_xor(i)] + ROTL32(D[j], s);
	#pragma unroll 8
	for(int j=0; j<8; j++)
		A[j] = R[j];
}

static __constant__ uint32_t d_cw[4][8][8] = {
	0x531B1720, 0xAC2CDE09, 0x0B902D87, 0x2369B1F4, 0x2931AA01, 0x02E4B082, 0xC914C914, 0xC1DAE1A6,	0xF18C2B5C, 0x08AC306B, 0x27BFC914, 0xCEDC548D, 0xC630C4BE, 0xF18C4335, 0xF0D3427C, 0xBE3DA380,
	0x143C02E4, 0xA948C630, 0xA4F2DE09, 0xA71D2085, 0xA439BD84, 0x109FCD6A, 0xEEA8EF61, 0xA5AB1CE8,	0x0B90D4A4, 0x3D6D039D, 0x25944D53, 0xBAA0E034, 0x5BC71E5A, 0xB1F4F2FE, 0x12CADE09, 0x548D41C3,
	0x3CB4F80D, 0x36ECEBC4, 0xA66443EE, 0x43351ABD, 0xC7A20C49, 0xEB0BB366, 0xF5293F98, 0x49B6DE09,	0x531B29EA, 0x02E402E4, 0xDB25C405, 0x53D4E543, 0x0AD71720, 0xE1A61A04, 0xB87534C1, 0x3EDF43EE,
	0x213E50F0, 0x39173EDF, 0xA9485B0E, 0xEEA82EF9, 0x14F55771, 0xFAF15546, 0x3D6DD9B3, 0xAB73B92E,	0x582A48FD, 0xEEA81892, 0x4F7EAA01, 0xAF10A88F, 0x11581720, 0x34C124DB, 0xD1C0AB73, 0x1E5AF0D3,
	0xC34C07F3, 0xC914143C, 0x599CBC12, 0xBCCBE543, 0x385EF3B7, 0x14F54C9A, 0x0AD7C068, 0xB64A21F7,	0xDEC2AF10, 0xC6E9C121, 0x56B8A4F2, 0x1158D107, 0xEB0BA88F, 0x050FAABA, 0xC293264D, 0x548D46D2,
	0xACE5E8E0, 0x53D421F7, 0xF470D279, 0xDC974E0C, 0xD6CF55FF, 0xFD1C4F7E, 0x36EC36EC, 0x3E261E5A,	0xEBC4FD1C, 0x56B839D0, 0x5B0E21F7, 0x58E3DF7B, 0x5BC7427C, 0xEF613296, 0x1158109F, 0x5A55E318,
	0xA7D6B703, 0x1158E76E, 0xB08255FF, 0x50F05771, 0xEEA8E8E0, 0xCB3FDB25, 0x2E40548D, 0xE1A60F2D,	0xACE5D616, 0xFD1CFD1C, 0x24DB3BFB, 0xAC2C1ABD, 0xF529E8E0, 0x1E5AE5FC, 0x478BCB3F, 0xC121BC12,
	0xF4702B5C, 0xC293FC63, 0xDA6CB2AD, 0x45601FCC, 0xA439E1A6, 0x4E0C0D02, 0xED3621F7, 0xAB73BE3D,	0x0E74D4A4, 0xF754CF95, 0xD84136EC, 0x3124AB73, 0x39D03B42, 0x0E74BCCB, 0x0F2DBD84, 0x41C35C80,
	0xA4135BED, 0xE10E1EF2, 0x6C4F93B1, 0x6E2191DF, 0xE2E01D20, 0xD1952E6B, 0x6A7D9583, 0x131DECE3,	0x369CC964, 0xFB73048D, 0x9E9D6163, 0x280CD7F4, 0xD9C6263A, 0x1062EF9E, 0x2AC7D539, 0xAD2D52D3,
	0x0A03F5FD, 0x197CE684, 0xAA72558E, 0xDE5321AD, 0xF0870F79, 0x607A9F86, 0xAFE85018, 0x2AC7D539,	0xE2E01D20, 0x2AC7D539, 0xC6A93957, 0x624C9DB4, 0x6C4F93B1, 0x641E9BE2, 0x452CBAD4, 0x263AD9C6,
	0xC964369C, 0xC3053CFB, 0x452CBAD4, 0x95836A7D, 0x4AA2B55E, 0xAB5B54A5, 0xAC4453BC, 0x74808B80,	0xCB3634CA, 0xFC5C03A4, 0x4B8BB475, 0x21ADDE53, 0xE2E01D20, 0xDF3C20C4, 0xBD8F4271, 0xAA72558E,
	0xFC5C03A4, 0x48D0B730, 0x2AC7D539, 0xD70B28F5, 0x53BCAC44, 0x3FB6C04A, 0x14EFEB11, 0xDB982468,	0x9A1065F0, 0xB0D14F2F, 0x8D5272AE, 0xC4D73B29, 0x91DF6E21, 0x949A6B66, 0x303DCFC3, 0x5932A6CE,
	0x1234EDCC, 0xF5140AEC, 0xCDF1320F, 0x3DE4C21C, 0x48D0B730, 0x1234EDCC, 0x131DECE3, 0x52D3AD2D,	0xE684197C, 0x6D3892C8, 0x72AE8D52, 0x6FF3900D, 0x73978C69, 0xEB1114EF, 0x15D8EA28, 0x71C58E3B,
	0x90F66F0A, 0x15D8EA28, 0x9BE2641E, 0x65F09A10, 0xEA2815D8, 0xBD8F4271, 0x3A40C5C0, 0xD9C6263A,	0xB38C4C74, 0xBAD4452C, 0x70DC8F24, 0xAB5B54A5, 0x46FEB902, 0x1A65E59B, 0x0DA7F259, 0xA32A5CD6,
	0xD62229DE, 0xB81947E7, 0x6D3892C8, 0x15D8EA28, 0xE59B1A65, 0x065FF9A1, 0xB2A34D5D, 0x6A7D9583,	0x975568AB, 0xFC5C03A4, 0x2E6BD195, 0x966C6994, 0xF2590DA7, 0x263AD9C6, 0x5A1BA5E5, 0xB0D14F2F,
	0x975568AB, 0x6994966C, 0xF1700E90, 0xD3672C99, 0xCC1F33E1, 0xFC5C03A4, 0x452CBAD4, 0x4E46B1BA,	0xF1700E90, 0xB2A34D5D, 0xD0AC2F54, 0x5760A8A0, 0x8C697397, 0x624C9DB4, 0xE85617AA, 0x95836A7D
		
};

__device__ __forceinline__
static void Round8_0_final(uint32_t* A,const uint32_t r,const  uint32_t s,const uint32_t t,const uint32_t u){

	STEP8_IF(d_cw[0][0],0, r, s, A, &A[ 8], &A[16], &A[24]);
	STEP8_IF(d_cw[0][1],1, s, t, &A[24], A, &A[ 8], &A[16]);
	STEP8_IF(d_cw[0][2],2, t, u, &A[16], &A[24], A, &A[ 8]);
	STEP8_IF(d_cw[0][3],3, u, r, &A[ 8], &A[16], &A[24], A);
	STEP8_MAJ(d_cw[0][4],4, r, s, A, &A[ 8], &A[16], &A[24]);
	STEP8_MAJ(d_cw[0][5],5, s, t, &A[24], A, &A[ 8], &A[16]);
	STEP8_MAJ(d_cw[0][6],6, t, u, &A[16], &A[24], A, &A[ 8]);
	STEP8_MAJ(d_cw[0][7],7, u, r, &A[ 8], &A[16], &A[24], A);
}

__device__ __forceinline__
static void Round8_1_final(uint32_t* A,const uint32_t r,const  uint32_t s,const uint32_t t,const uint32_t u){

	STEP8_IF(d_cw[1][0], 8, r, s, A, &A[ 8], &A[16], &A[24]);
	STEP8_IF(d_cw[1][1], 9, s, t, &A[24], A, &A[ 8], &A[16]);
	STEP8_IF(d_cw[1][2],10, t, u, &A[16], &A[24], A, &A[ 8]);
	STEP8_IF(d_cw[1][3],11, u, r, &A[ 8], &A[16], &A[24], A);
	STEP8_MAJ(d_cw[1][4],12, r, s, A, &A[ 8], &A[16], &A[24]);
	STEP8_MAJ(d_cw[1][5],13, s, t, &A[24], A, &A[ 8], &A[16]);
	STEP8_MAJ(d_cw[1][6],14, t, u, &A[16], &A[24], A, &A[ 8]);
	STEP8_MAJ(d_cw[1][7],15, u, r, &A[ 8], &A[16], &A[24], A);
}

__device__ __forceinline__
static void Round8_2_final(uint32_t* A,const uint32_t r,const  uint32_t s,const uint32_t t,const uint32_t u){

	STEP8_IF(d_cw[2][0],16, r, s, A, &A[ 8], &A[16], &A[24]);
	STEP8_IF(d_cw[2][1],17, s, t, &A[24], A, &A[ 8], &A[16]);
	STEP8_IF(d_cw[2][2],18, t, u, &A[16], &A[24], A, &A[ 8]);
	STEP8_IF(d_cw[2][3],19, u, r, &A[ 8], &A[16], &A[24], A);
	STEP8_MAJ(d_cw[2][4],20, r, s, A, &A[ 8], &A[16], &A[24]);
	STEP8_MAJ(d_cw[2][5],21, s, t, &A[24], A, &A[ 8], &A[16]);
	STEP8_MAJ(d_cw[2][6],22, t, u, &A[16], &A[24], A, &A[ 8]);
	STEP8_MAJ(d_cw[2][7],23, u, r, &A[ 8], &A[16], &A[24], A);
}

__device__ __forceinline__
static void Round8_3_final(uint32_t* A,const uint32_t r,const  uint32_t s,const uint32_t t,const uint32_t u){

	STEP8_IF(d_cw[3][0],24, r, s, A, &A[ 8], &A[16], &A[24]);
	STEP8_IF(d_cw[3][1],25, s, t, &A[24], A, &A[ 8], &A[16]);
	STEP8_IF(d_cw[3][2],26, t, u, &A[16], &A[24], A, &A[ 8]);
	STEP8_IF(d_cw[3][3],27, u, r, &A[ 8], &A[16], &A[24], A);
	STEP8_MAJ(d_cw[3][4],28, r, s, A, &A[ 8], &A[16], &A[24]);
	STEP8_MAJ(d_cw[3][5],29, s, t, &A[24], A, &A[ 8], &A[16]);
	STEP8_MAJ(d_cw[3][6],30, t, u, &A[16], &A[24], A, &A[ 8]);
	STEP8_MAJ(d_cw[3][7],31, u, r, &A[ 8], &A[16], &A[24], A);
}

//#define expanded_vector(x) __ldg(&g_fft4[x])
static __device__ __forceinline__ void expanded_vector(uint32_t* w,const uint4* ptr){
	asm volatile ("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(w[0]), "=r"(w[1]),"=r"(w[2]), "=r"(w[3]) : __LDG_PTR(ptr));
}

__device__ __forceinline__
static void Round8(uint32_t* A, const uint32_t thr_offset, const uint4 *const __restrict__ g_fft4) {

	uint32_t w[8];
	uint32_t tmp = thr_offset;

	uint32_t r = 3, s = 23, t = 17, u = 27;
	
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,0, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,1, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,2, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,3, u, r, &A[8], &A[16], &A[24], A);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,4, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,5, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,6, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,7, u, r, &A[8], &A[16], &A[24], A);

	r = 28; s = 19; t = 22; u = 7;
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,8, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,9, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,10, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,11, u, r, &A[8], &A[16], &A[24], A);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,12, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,13, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,14, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,15, u, r, &A[8], &A[16], &A[24], A);

	r = 29; s = 9; t = 15; u = 5;
	
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,16, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,17, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,18, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,19, u, r, &A[8], &A[16], &A[24], A);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,20, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,21, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,22, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,23, u, r, &A[8], &A[16], &A[24], A);

	r =  4; s = 13; t = 10; u = 25;

 	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,24, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,25, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,26, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_IF(w,27, u, r, &A[8], &A[16], &A[24], A);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,28, r, s, A, &A[8], &A[16], &A[24]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,29, s, t, &A[24], A, &A[8], &A[16]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,30, t, u, &A[16], &A[24], A, &A[8]);
	expanded_vector(&w[0],&g_fft4[tmp++]);
	expanded_vector(&w[4],&g_fft4[tmp++]);
	STEP8_MAJ(w,31, u, r, &A[8], &A[16], &A[24], A);

}

/********************* Message expansion ************************/

/*
 * Reduce modulo 257; result is in [-127; 383]
 * REDUCE(x) := (x&255) - (x>>8)
 */
#define REDUCE(x) \
	(((x)&255) - ((x)>>8))

/*
 * Reduce from [-127; 383] to [-128; 128]
 * EXTRA_REDUCE_S(x) := x<=128 ? x : x-257
 */
#define EXTRA_REDUCE_S(x) \
	((x)<=128 ? (x) : (x)-257)

/*
 * Reduce modulo 257; result is in [-128; 128]
 */
#define REDUCE_FULL_S(x) \
	EXTRA_REDUCE_S(REDUCE(x))

// Parallelization:
//
// FFT_8  wird 2 times 8-fach parallel ausgeführt (in FFT_64)
//        and  1 time 16-fach parallel (in FFT_128_full)
//
// STEP8_IF and STEP8_MAJ beinhalten je 2x 8-fach parallel Operations

/**
 * FFT_8 using w=4 as 8th root of unity
 * Unrolled decimation in frequency (DIF) radix-2 NTT.
 * Output data is in revbin_permuted order.
 */
__device__ __forceinline__
static void FFT_8(int *y,const uint8_t stripe){

#define BUTTERFLY(i,j,n) \
do { \
	int u= y[stripe*i]; \
	int v= y[stripe*j]; \
	y[stripe*i] = u+v; \
	y[stripe*j] = (u-v) << (n<<1); \
} while(0)

	BUTTERFLY(0, 4, 0);
	BUTTERFLY(1, 5, 1);
	BUTTERFLY(2, 6, 2);
	BUTTERFLY(3, 7, 3);

	y[stripe*6] = REDUCE(y[stripe*6]);
	y[stripe*7] = REDUCE(y[stripe*7]);

	BUTTERFLY(0, 2, 0);
	BUTTERFLY(4, 6, 0);
	BUTTERFLY(1, 3, 2);
	BUTTERFLY(5, 7, 2);

	y[stripe*7] = REDUCE(y[stripe*7]);

	BUTTERFLY(0, 1, 0);
	BUTTERFLY(2, 3, 0);
	BUTTERFLY(4, 5, 0);
	BUTTERFLY(6, 7, 0);

	y[ 0] = REDUCE(y[ 0]);
	y[stripe] = REDUCE(y[stripe]);
	y[stripe<<1] = REDUCE(y[stripe<<1]);
	y[stripe*3] = REDUCE(y[stripe*3]);
	y[stripe<<2] = REDUCE(y[stripe<<2]);
	y[stripe*5] = REDUCE(y[stripe*5]);
	y[stripe*6] = REDUCE(y[stripe*6]);
	y[stripe*7] = REDUCE(y[stripe*7]);
	
	y[ 0] = EXTRA_REDUCE_S(y[ 0]);
	y[stripe] = EXTRA_REDUCE_S(y[stripe]);
	y[stripe<<1] = EXTRA_REDUCE_S(y[stripe<<1]);
	y[stripe*3] = EXTRA_REDUCE_S(y[stripe*3]);
	y[stripe<<2] = EXTRA_REDUCE_S(y[stripe<<2]);
	y[stripe*5] = EXTRA_REDUCE_S(y[stripe*5]);
	y[stripe*6] = EXTRA_REDUCE_S(y[stripe*6]);
	y[stripe*7] = EXTRA_REDUCE_S(y[stripe*7]);

#undef BUTTERFLY
}

/**
 * FFT_16 using w=2 as 16th root of unity
 * Unrolled decimation in frequency (DIF) radix-2 NTT.
 * Output data is in revbin_permuted order.
 */
__device__ __forceinline__
static void FFT_16(int *y){

#define DO_REDUCE_FULL_S(i) \
	do { \
		y[i] = REDUCE(y[i]); \
		y[i] = EXTRA_REDUCE_S(y[i]); \
	} while(0)

	int u,v;

	const uint8_t thr = threadIdx.x&7;

	u = y[0]; // 0..7
	v = y[1]; // 8..15
	y[0] = u+v;
	y[1] = (u-v) << (thr);

	if ((thr) >=3) y[1] = REDUCE(y[1]);  // 11...15

	u = __shfl(y[0],  (threadIdx.x&3),8); // 0,1,2,3  0,1,2,3
	v = __shfl(y[0],4+(threadIdx.x&3),8); // 4,5,6,7  4,5,6,7
	y[0] = ((thr) < 4) ? (u+v) : ((u-v) << ((threadIdx.x&3)<<1));

	u = __shfl(y[1],  (threadIdx.x&3),8); // 8,9,10,11    8,9,10,11
	v = __shfl(y[1],4+(threadIdx.x&3),8); // 12,13,14,15  12,13,14,15
	y[1] = ((thr) < 4) ? (u+v) : ((u-v) << ((threadIdx.x&3)<<1));

	if ((threadIdx.x&1) && (thr >= 4)) {
		y[0] = REDUCE(y[0]);  // 5, 7
		y[1] = REDUCE(y[1]);  // 13, 15
	}

	u = __shfl(y[0],  (threadIdx.x&5),8); // 0,1,0,1  4,5,4,5
	v = __shfl(y[0],2+(threadIdx.x&5),8); // 2,3,2,3  6,7,6,7
	y[0] = ((threadIdx.x&3) < 2) ? (u+v) : ((u-v) << ((threadIdx.x&1)<<2));

	u = __shfl(y[1],  (threadIdx.x&5),8); // 8,9,8,9      12,13,12,13
	v = __shfl(y[1],2+(threadIdx.x&5),8); // 10,11,10,11  14,15,14,15
	y[1] = ((threadIdx.x&3) < 2) ? (u+v) : ((u-v) << ((threadIdx.x&1)<<2));

	u = __shfl(y[0],  (threadIdx.x&6),8); // 0,0,2,2      4,4,6,6
	v = __shfl(y[0],1+(threadIdx.x&6),8); // 1,1,3,3      5,5,7,7
	y[0] = ((threadIdx.x&1) < 1) ? (u+v) : (u-v);

	u = __shfl(y[1],  (threadIdx.x&6),8); // 8,8,10,10    12,12,14,14
	v = __shfl(y[1],1+(threadIdx.x&6),8); // 9,9,11,11    13,13,15,15
	y[1] = ((threadIdx.x&1) < 1) ? (u+v) : (u-v);

	DO_REDUCE_FULL_S( 0); // 0...7
	DO_REDUCE_FULL_S( 1); // 8...15

#undef DO_REDUCE_FULL_S
}

/***************************************************/
#if __CUDA_ARCH__ > 500
__global__ __launch_bounds__(TPB52_1,9)
#else
__global__ __launch_bounds__(TPB50_1,9)
#endif
static void x11_simd512_gpu_expand_64(uint32_t threads,const uint32_t* __restrict__ g_hash, uint4 *g_temp4)
{
	const uint32_t threadBloc = (blockDim.x * blockIdx.x + threadIdx.x)>>3;
	const uint8_t thr        = (threadIdx.x & 7);
	/* Message Expansion using Number Theoretical Transform similar to FFT */
	int expanded[32];

	uint4 vec0;
	int P, Q, P1, Q1, P2, Q2;

	const bool even = (threadIdx.x & 1) == 0;
	const bool hi = (thr)>=4;
	const bool lo = (thr)<4;
	const bool sel = ((threadIdx.x+2)&7) >= 4;  // 2,3,4,5
	
	if (threadBloc < threads){
		
		const uint32_t hashPosition = threadBloc<<4;

		const uint32_t *inpHash = &g_hash[hashPosition];

		const uint32_t data0 = __ldg(&inpHash[thr]);
		const uint32_t data1 = __ldg(&inpHash[thr + 8]);

		// Puffer für expandierte Nachricht
		uint4 *temp4 = &g_temp4[hashPosition<<2];

		#pragma unroll 4
		for (uint32_t i=0; i < 4; i++) {
			expanded[  i] = bfe(__byte_perm(__shfl(data0, i<<1, 8), __shfl(data0, (i<<1)+1, 8), thr),0,8);
		}
		#pragma unroll 4
		for (uint32_t i=0; i < 4; i++) {			
			expanded[4+i] = bfe(__byte_perm(__shfl(data1, i<<1, 8), __shfl(data1, (i<<1)+1, 8), thr),0,8);
		}
		#pragma unroll 8
		for (uint32_t i=8; i < 16; i++) {			
			expanded[ i] = 0;
		}
		/*
		 * FFT_256 using w=41 as 256th root of unity. Decimation in frequency (DIF) NTT. Output data is in revbin_permuted order. In place.
		 */
		#pragma unroll 8
		for (uint32_t i=0; i<8; i++)
			expanded[16+i] = REDUCE(expanded[i] * c_FFT256_2_128_Twiddle[8*i+(thr)]);

		#pragma unroll 8
		for (uint32_t i=24; i < 32; i++) {			
			expanded[ i] = 0;
		}		
		/* handle X^255 with an additional butterfly */
		if (thr==7){
			expanded[15] = 1;
			expanded[31] = REDUCE((-1) * c_FFT256_2_128_Twiddle[127]);
		}

//		FFT_128_full(expanded);
		FFT_8(expanded,2); // eight parallel FFT8's
		FFT_8(&expanded[16],2); // eight parallel FFT8's
		FFT_8(&expanded[ 1],2); // eight parallel FFT8's
		FFT_8(&expanded[17],2); // eight parallel FFT8's
		
		#pragma unroll 16
		for (uint32_t i=0; i<16; i++){
			expanded[i] = REDUCE(expanded[i]*c_FFT128_8_16_Twiddle[i*8+(thr)]);
			expanded[i+16] = REDUCE(expanded[i+16]*c_FFT128_8_16_Twiddle[i*8+(thr)]);			
		}

		#pragma unroll 8
		for (uint32_t i=0; i<8; i++){
			FFT_16(expanded+(i<<1));  // eight sequential FFT16's, each one executed in parallel by 8 threads
			FFT_16(expanded+16+(i<<1));  // eight sequential FFT16's, each one executed in parallel by 8 threads			
		}

		// store w matrices in global memory
		P1 = expanded[ 0]; P2 = __shfl(expanded[ 2], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[16]; Q2 = __shfl(expanded[18], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[0][thr], 8);
		P1 = expanded[ 8]; P2 = __shfl(expanded[10], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[24]; Q2 = __shfl(expanded[26], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[0][thr], 8);
		P1 = expanded[ 4]; P2 = __shfl(expanded[ 6], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[20]; Q2 = __shfl(expanded[22], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[0][thr], 8);
		P1 = expanded[12]; P2 = __shfl(expanded[14], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[28]; Q2 = __shfl(expanded[30], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[0][thr], 8);
		temp4[thr] = vec0;

		P1 = expanded[ 1]; P2 = __shfl(expanded[ 3], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[17]; Q2 = __shfl(expanded[19], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[1][thr], 8);
		P1 = expanded[ 9]; P2 = __shfl(expanded[11], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[25]; Q2 = __shfl(expanded[27], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[1][thr], 8);
		P1 = expanded[ 5]; P2 = __shfl(expanded[ 7], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[21]; Q2 = __shfl(expanded[23], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[1][thr], 8);
		P1 = expanded[13]; P2 = __shfl(expanded[15], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
		Q1 = expanded[29]; Q2 = __shfl(expanded[31], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[1][thr], 8);
		temp4[8+(thr)] = vec0;

		P1 = hi?expanded[ 1]:expanded[ 0]; P2 = __shfl(hi?expanded[ 3]:expanded[ 2], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = hi?expanded[17]:expanded[16]; Q2 = __shfl(hi?expanded[19]:expanded[18], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[2][thr], 8);
		P1 = hi?expanded[ 9]:expanded[ 8]; P2 = __shfl(hi?expanded[11]:expanded[10], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = hi?expanded[25]:expanded[24]; Q2 = __shfl(hi?expanded[27]:expanded[26], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[2][thr], 8);
		P1 = hi?expanded[ 5]:expanded[ 4]; P2 = __shfl(hi?expanded[ 7]:expanded[ 6], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = hi?expanded[21]:expanded[20]; Q2 = __shfl(hi?expanded[23]:expanded[22], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[2][thr], 8);
		P1 = hi?expanded[13]:expanded[12]; P2 = __shfl(hi?expanded[15]:expanded[14], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = hi?expanded[29]:expanded[28]; Q2 = __shfl(hi?expanded[31]:expanded[30], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[2][thr], 8);
		temp4[16+(thr)] = vec0;

		P1 = lo?expanded[ 1]:expanded[ 0]; P2 = __shfl(lo?expanded[ 3]:expanded[ 2], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = lo?expanded[17]:expanded[16]; Q2 = __shfl(lo?expanded[19]:expanded[18], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[3][thr], 8);
		P1 = lo?expanded[ 9]:expanded[ 8]; P2 = __shfl(lo?expanded[11]:expanded[10], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = lo?expanded[25]:expanded[24]; Q2 = __shfl(lo?expanded[27]:expanded[26], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[3][thr], 8);
		P1 = lo?expanded[ 5]:expanded[ 4]; P2 = __shfl(lo?expanded[ 7]:expanded[ 6], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = lo?expanded[21]:expanded[20]; Q2 = __shfl(lo?expanded[23]:expanded[22], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[3][thr], 8);
		P1 = lo?expanded[13]:expanded[12]; P2 = __shfl(lo?expanded[15]:expanded[14], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
		Q1 = lo?expanded[29]:expanded[28]; Q2 = __shfl(lo?expanded[31]:expanded[30], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(185*P,  185*Q , 0x5410), c_perm[3][thr], 8);
		temp4[24+(thr)] = vec0;

		P1 = sel?expanded[0]:expanded[1]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[2]:expanded[3]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[4][thr], 8);
		P1 = sel?expanded[8]:expanded[9]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[10]:expanded[11]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[4][thr], 8);
		P1 = sel?expanded[4]:expanded[5]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[6]:expanded[7]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[4][thr], 8);
		P1 = sel?expanded[12]:expanded[13]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[14]:expanded[15]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[4][thr], 8);

		temp4[32+thr] = vec0;

		P1 = sel?expanded[1]:expanded[0]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[3]:expanded[2]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[5][thr], 8);
		P1 = sel?expanded[9]:expanded[8]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[11]:expanded[10]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[5][thr], 8);
		P1 = sel?expanded[5]:expanded[4]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[7]:expanded[6]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[5][thr], 8);
		P1 = sel?expanded[13]:expanded[12]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		Q2 = sel?expanded[15]:expanded[14]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[5][thr], 8);

		temp4[40+thr] = vec0;

		uint32_t t;
		t = __shfl(expanded[17],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[16]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[19],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[18]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[6][thr], 8);
		t = __shfl(expanded[25],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[24]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[27],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[26]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[6][thr], 8);
		t = __shfl(expanded[21],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[20]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[23],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[22]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[6][thr], 8);
		t = __shfl(expanded[29],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[28]; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[31],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[30]; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[6][thr], 8);

		temp4[48+thr] = vec0;

		t = __shfl(expanded[16],(threadIdx.x+4)&7,8); P1 = sel?expanded[17]:t; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[18],(threadIdx.x+4)&7,8); Q2 = sel?expanded[19]:t; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.x = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[7][thr], 8);
		t = __shfl(expanded[24],(threadIdx.x+4)&7,8); P1 = sel?expanded[25]:t; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[26],(threadIdx.x+4)&7,8); Q2 = sel?expanded[27]:t; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.y = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[7][thr], 8);
		t = __shfl(expanded[20],(threadIdx.x+4)&7,8); P1 = sel?expanded[21]:t; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[22],(threadIdx.x+4)&7,8); Q2 = sel?expanded[23]:t; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.z = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[7][thr], 8);
		t = __shfl(expanded[28],(threadIdx.x+4)&7,8); P1 = sel?expanded[29]:t; Q1 = __shfl(P1, (threadIdx.x^1)&7, 8);
		t = __shfl(expanded[30],(threadIdx.x+4)&7,8); Q2 = sel?expanded[31]:t; P2 = __shfl(Q2, (threadIdx.x^1)&7, 8);
		P = even? P1 : P2; Q = even? Q1 : Q2;
		vec0.w = __shfl(__byte_perm(233*P,  233*Q , 0x5410), c_perm[7][thr], 8);

		temp4[56+thr] = vec0;
	}
}
