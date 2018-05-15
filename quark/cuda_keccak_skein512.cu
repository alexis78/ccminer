/*
 * Based on SP's keccak-skein merged implementation
 * For compute5.0/5.2 under CUDA7.5
 * Quarkcoin's throughput increased by 0.27%
 *
 * Provos Alexis - 2016
 */

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h"

#define TPB52 256
#define TPB50 128

__constant__ const uint2 keccak_round_constants[24] = {
		{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 }, { 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
		{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 }, { 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
		{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 },	{ 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
		{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 },	{ 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
		{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 }, { 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
		{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 },	{ 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

__device__
static void macro1(uint2 *const __restrict__ p){
	p[0] += p[1];p[2] += p[3];p[4] += p[5];p[6] += p[7];p[1] = ROL2(p[1],46) ^ p[0];p[3] = ROL2(p[3],36) ^ p[2];p[5] = ROL2(p[5],19) ^ p[4];p[7] = ROL2(p[7], 37) ^ p[6];
	p[2] += p[1];p[4] += p[7];p[6] += p[5];p[0] += p[3];p[1] = ROL2(p[1],33) ^ p[2];p[7] = ROL2(p[7],27) ^ p[4];p[5] = ROL2(p[5],14) ^ p[6];p[3] = ROL2(p[3], 42) ^ p[0];
	p[4] += p[1];p[6] += p[3];p[0] += p[5];p[2] += p[7];p[1] = ROL2(p[1],17) ^ p[4];p[3] = ROL2(p[3],49) ^ p[6];p[5] = ROL2(p[5],36) ^ p[0];p[7] = ROL2(p[7], 39) ^ p[2];
	p[6] += p[1];p[0] += p[7];p[2] += p[5];p[4] += p[3];p[1] = ROL2(p[1],44) ^ p[6];p[7] = ROL2(p[7], 9) ^ p[0];p[5] = ROL2(p[5],54) ^ p[2];p[3] = ROR8(p[3]) ^ p[4];
}

__device__
static void macro2(uint2 *const __restrict__ p){
	p[0] += p[1];p[2] += p[3];p[4] += p[5];p[6] += p[7];p[1] = ROL2(p[1], 39) ^ p[0];p[3] = ROL2(p[3], 30) ^ p[2];p[5] = ROL2(p[5], 34) ^ p[4];p[7] = ROL24(p[7]) ^ p[6];
	p[2] += p[1];p[4] += p[7];p[6] += p[5];p[0] += p[3];p[1] = ROL2(p[1], 13) ^ p[2];p[7] = ROL2(p[7], 50) ^ p[4];p[5] = ROL2(p[5], 10) ^ p[6];p[3] = ROL2(p[3], 17) ^ p[0];
	p[4] += p[1];p[6] += p[3];p[0] += p[5];p[2] += p[7];p[1] = ROL2(p[1], 25) ^ p[4];p[3] = ROL2(p[3], 29) ^ p[6];p[5] = ROL2(p[5], 39) ^ p[0];p[7] = ROL2(p[7], 43) ^ p[2];
	p[6] += p[1];p[0] += p[7];p[2] += p[5];p[4] += p[3];p[1] = ROL8(p[1]) ^ p[6];p[7] = ROL2(p[7], 35) ^ p[0];p[5] = ROR8(p[5]) ^ p[2];p[3] = ROL2(p[3], 22) ^ p[4];
}

__constant__ const uint2 buffer[152] = {
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC33,0xAE18A40B},
	{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC73,0x9E18A40B},{0x98173EC5,0xCAB2076D},
	{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC73,0x9E18A40B},{0x98173F04,0xCAB2076D},{0x749C51D0,0x4903ADFF},
	{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173F04,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF06,0x0D95DE39},
	{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79BD2,0x8FD19341},
	{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CB6,0x9A255629},
	{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7B6,0x5DB62599},
	{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C3FB,0xEABE394C},
	{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B52B,0x991112C7},
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC3C,0xAE18A40B},
	{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC73,0x9E18A40B},{0x98173ece,0xcab2076d},
	{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC73,0x9E18A40B},{0x98173F04,0xCAB2076D},{0x749C51D9,0x4903ADFF},
	{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173F04,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF0F,0x0D95DE39},
	{0xDF6CA7B0,0x5DB62599},{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x3903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79BDB,0x8FD19341},
	{0xA9D5C3F4,0xEABE394C},{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF43,0xFD95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CBF,0x9A255629},
	{0x1A75B523,0x991112C7},{0x660FCC33,0xAE18A40B},{0x98173EC4,0xCAB2076D},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79C0E,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7BF,0x5DB62599},
	{0x660FCC33,0xAE18A40B},{0x98173ec4,0xcab2076d},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x8A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C404,0xEABE394C},
	{0x98173ec4,0xcab2076d},{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7F0,0x4DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B534,0x991112C7},
	{0x749C51CE,0x4903ADFF},{0x9746DF03,0x0D95DE39},{0x27C79BCE,0x8FD19341},{0xFF352CB1,0x9A255629},{0xDF6CA7B0,0x5DB62599},{0xA9D5C434,0xEABE394C},{0x1A75B523,0x891112C7},{0x660FCC45,0xAE18A40B}
};

__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52,3)
#else
__launch_bounds__(TPB50,7)
#endif
void quark_keccakskein512_gpu_hash_64(uint32_t threads, uint2 *g_hash,const uint32_t * g_nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint2 t[5], u[5], v, w;
	uint2 s[25];
	
	if (thread < threads)
	{
		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];

		uint2x4* phash = (uint2x4 *)&g_hash[hashPosition * 8];

		*(uint2x4*)&s[ 0] = __ldg4(&phash[ 0]);
		*(uint2x4*)&s[ 4] = __ldg4(&phash[ 1]);
		
		s[8] = make_uint2(1,0x80000000);

		/*theta*/
		t[ 0] = s[ 0]^s[ 5];
		t[ 1] = s[ 1]^s[ 6];
		t[ 2] = s[ 2]^s[ 7];
		t[ 3] = s[ 3]^s[ 8];
		t[ 4] = s[4];
		
		/*theta*/
		#pragma unroll 5
		for(int j=0;j<5;j++){
			u[ j] = ROL2(t[ j], 1);
		}
		
		s[ 4] = xor3x(s[ 4], t[3], u[ 0]);
		s[24] = s[19] = s[14] = s[ 9] = t[ 3] ^ u[ 0];

		s[ 0] = xor3x(s[ 0], t[4], u[ 1]);
		s[ 5] = xor3x(s[ 5], t[4], u[ 1]);
		s[20] = s[15] = s[10] = t[4] ^ u[ 1];

		s[ 1] = xor3x(s[ 1], t[0], u[ 2]);
		s[ 6] = xor3x(s[ 6], t[0], u[ 2]);
		s[21] = s[16] = s[11] = t[0] ^ u[ 2];
		
		s[ 2] = xor3x(s[ 2], t[1], u[ 3]);
		s[ 7] = xor3x(s[ 7], t[1], u[ 3]);
		s[22] = s[17] = s[12] = t[1] ^ u[ 3];
		
		s[ 3] = xor3x(s[ 3], t[2], u[ 4]);s[ 8] = xor3x(s[ 8], t[2], u[ 4]);
		s[23] = s[18] = s[13] = t[2] ^ u[ 4];
		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1]  = ROL2(s[6], 44);
		s[6]  = ROL2(s[9], 20);
		s[9]  = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2]  = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL8(s[19]);
		s[19] = ROR8(s[23]);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4]  = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8]  = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5]  = ROL2(s[3], 28);
		s[3]  = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7]  = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);
		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		#pragma unroll 5
		for(int j=0;j<25;j+=5){
			v=s[j];w=s[j + 1];s[j] = chi(v,w,s[j+2]);s[j+1] = chi(w,s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
		}
		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[ 0];

		for (int i = 1; i < 23; i++) {
			/*theta*/
			#pragma unroll
			for(int j=0;j<5;j++){
				t[ j] = vectorize(xor5(devectorize(s[ j]),devectorize(s[j+5]),devectorize(s[j+10]),devectorize(s[j+15]),devectorize(s[j+20])));
			}

			/*theta*/
			#pragma unroll
			for(int j=0;j<5;j++){
				u[ j] = ROL2(t[ j], 1);
			}
			s[ 4] = xor3x(s[ 4], t[3], u[ 0]);s[ 9] = xor3x(s[ 9], t[3], u[ 0]);s[14] = xor3x(s[14], t[3], u[ 0]);s[19] = xor3x(s[19], t[3], u[ 0]);s[24] = xor3x(s[24], t[3], u[ 0]);
			s[ 0] = xor3x(s[ 0], t[4], u[ 1]);s[ 5] = xor3x(s[ 5], t[4], u[ 1]);s[10] = xor3x(s[10], t[4], u[ 1]);s[15] = xor3x(s[15], t[4], u[ 1]);s[20] = xor3x(s[20], t[4], u[ 1]);
			s[ 1] = xor3x(s[ 1], t[0], u[ 2]);s[ 6] = xor3x(s[ 6], t[0], u[ 2]);s[11] = xor3x(s[11], t[0], u[ 2]);s[16] = xor3x(s[16], t[0], u[ 2]);s[21] = xor3x(s[21], t[0], u[ 2]);
			s[ 2] = xor3x(s[ 2], t[1], u[ 3]);s[ 7] = xor3x(s[ 7], t[1], u[ 3]);s[12] = xor3x(s[12], t[1], u[ 3]);s[17] = xor3x(s[17], t[1], u[ 3]);s[22] = xor3x(s[22], t[1], u[ 3]);
			s[ 3] = xor3x(s[ 3], t[2], u[ 4]);s[ 8] = xor3x(s[ 8], t[2], u[ 4]);s[13] = xor3x(s[13], t[2], u[ 4]);s[18] = xor3x(s[18], t[2], u[ 4]);s[23] = xor3x(s[23], t[2], u[ 4]);

			/* rho pi: b[..] = rotl(a[..], ..) */
			v = s[1];
			s[1]  = ROL2(s[6], 44);
			s[6]  = ROL2(s[9], 20);
			s[9]  = ROL2(s[22], 61);
			s[22] = ROL2(s[14], 39);
			s[14] = ROL2(s[20], 18);
			s[20] = ROL2(s[2], 62);
			s[2]  = ROL2(s[12], 43);
			s[12] = ROL2(s[13], 25);
			s[13] = ROL8(s[19]);
			s[19] = ROR8(s[23]);
			s[23] = ROL2(s[15], 41);
			s[15] = ROL2(s[4], 27);
			s[4]  = ROL2(s[24], 14);
			s[24] = ROL2(s[21], 2);
			s[21] = ROL2(s[8], 55);
			s[8]  = ROL2(s[16], 45);
			s[16] = ROL2(s[5], 36);
			s[5]  = ROL2(s[3], 28);
			s[3]  = ROL2(s[18], 21);
			s[18] = ROL2(s[17], 15);
			s[17] = ROL2(s[11], 10);
			s[11] = ROL2(s[7], 6);
			s[7]  = ROL2(s[10], 3);
			s[10] = ROL2(v, 1);

			/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
			#pragma unroll
			for(int j=0;j<25;j+=5){
				v=s[j];w=s[j + 1];s[j] = chi(v,w,s[j+2]);s[j+1] = chi(w,s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
			}

			/* iota: a[0,0] ^= round constant */
			s[0] ^= keccak_round_constants[i];
		}
		/*theta*/
		#pragma unroll
		for(int j=0;j<5;j++){
			t[ j] = xor3x(xor3x(s[j+0],s[j+5],s[j+10]),s[j+15],s[j+20]);
		}
		/*theta*/
		#pragma unroll
		for(int j=0;j<5;j++){
			u[ j] = ROL2(t[ j], 1);
		}
		s[ 9] = xor3x(s[ 9], t[3], u[ 0]);
		s[24] = xor3x(s[24], t[3], u[ 0]);
		s[ 0] = xor3x(s[ 0], t[4], u[ 1]);
		s[10] = xor3x(s[10], t[4], u[ 1]);
		s[ 6] = xor3x(s[ 6], t[0], u[ 2]);
		s[16] = xor3x(s[16], t[0], u[ 2]);
		s[12] = xor3x(s[12], t[1], u[ 3]);
		s[22] = xor3x(s[22], t[1], u[ 3]);
		s[ 3] = xor3x(s[ 3], t[2], u[ 4]);
		s[18] = xor3x(s[18], t[2], u[ 4]);
		/* rho pi: b[..] = rotl(a[..], ..) */
		s[ 1]  = ROL2(s[ 6], 44);
		s[ 2]  = ROL2(s[12], 43);
		s[ 5]  = ROL2(s[ 3], 28);
		s[ 7]  = ROL2(s[10], 3);
		s[ 3]  = ROL2(s[18], 21);
		s[ 4]  = ROL2(s[24], 14);
		s[ 6]  = ROL2(s[ 9], 20);
		s[ 8]  = ROL2(s[16], 45);
		s[ 9]  = ROL2(s[22], 61);

		uint2 p[8],h[9];
		
		uint32_t t0;
		uint2 t1,t2;
		t0 = 8;
		t1 = vectorize(0xFF00000000000000);
		t2 = t1+t0;
				
		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		p[ 0] = chi(s[ 0],s[ 1],s[ 2]);
		p[ 1] = chi(s[ 1],s[ 2],s[ 3]);
		p[ 2] = chi(s[ 2],s[ 3],s[ 4]);
		p[ 3] = chi(s[ 3],s[ 4],s[ 0]);
		p[ 4] = chi(s[ 4],s[ 0],s[ 1]);
		p[ 5] = chi(s[ 5],s[ 6],s[ 7]);
		p[ 6] = chi(s[ 6],s[ 7],s[ 8]);
		p[ 7] = chi(s[ 7],s[ 8],s[ 9]);
		/* iota: a[0,0] ^= round constant */
		p[0] ^= keccak_round_constants[23];
		
		h[ 0] = p[0];
		h[ 1] = p[1];
		h[ 2] = p[2];
		h[ 3] = p[3];
		h[ 4] = p[4];
		h[ 5] = p[5];
		h[ 6] = p[6];
		h[ 7] = p[7];
		
		p[0] += buffer[0];	p[1] += buffer[1];	p[2] += buffer[2];	p[3] += buffer[3];	p[4] += buffer[4];	p[5] += buffer[5];	p[6] += buffer[6];	p[7] += buffer[7];
		macro1(p);
		p[0] += buffer[8];	p[1] += buffer[9];	p[2] += buffer[10];	p[3] += buffer[11];	p[4] += buffer[12];	p[5] += buffer[13];	p[6] += buffer[14];	p[7] += buffer[15];
		macro2(p);
		p[0] += buffer[16];	p[1] += buffer[17];	p[2] += buffer[18];	p[3] += buffer[19];	p[4] += buffer[20];	p[5] += buffer[21];	p[6] += buffer[22];	p[7] += buffer[23];
		macro1(p);
		p[0] += buffer[24];	p[1] += buffer[25];	p[2] += buffer[26];	p[3] += buffer[27];	p[4] += buffer[28];	p[5] += buffer[29];	p[6] += buffer[30];	p[7] += buffer[31];
		macro2(p);
		p[0] += buffer[32];	p[1] += buffer[33];	p[2] += buffer[34];	p[3] += buffer[35];	p[4] += buffer[36];	p[5] += buffer[37];	p[6] += buffer[38];	p[7] += buffer[39];
		macro1(p);
		p[0] += buffer[40];	p[1] += buffer[41];	p[2] += buffer[42];	p[3] += buffer[43];	p[4] += buffer[44];	p[5] += buffer[45];	p[6] += buffer[46];	p[7] += buffer[47];
		macro2(p);
		p[0] += buffer[48];	p[1] += buffer[49];	p[2] += buffer[50];	p[3] += buffer[51];	p[4] += buffer[52];	p[5] += buffer[53];	p[6] += buffer[54];	p[7] += buffer[55];
		macro1(p);
		p[0] += buffer[56];	p[1] += buffer[57];	p[2] += buffer[58];	p[3] += buffer[59];	p[4] += buffer[60];	p[5] += buffer[61];	p[6] += buffer[62];	p[7] += buffer[63];
		macro2(p);
		p[0] += buffer[64];	p[1] += buffer[65];	p[2] += buffer[66];	p[3] += buffer[67];	p[4] += buffer[68];	p[5] += buffer[69];	p[6] += buffer[70];	p[7] += buffer[71];
		macro1(p);
		p[0] += buffer[72];	p[1] += buffer[73];	p[2] += buffer[74];	p[3] += buffer[75];	p[4] += buffer[76];	p[5] += buffer[77];	p[6] += buffer[78];	p[7] += buffer[79];
		macro2(p);
		p[0] += buffer[80];	p[1] += buffer[81];	p[2] += buffer[82];	p[3] += buffer[83];	p[4] += buffer[84];	p[5] += buffer[85];	p[6] += buffer[86];	p[7] += buffer[87];
		macro1(p);
		p[0] += buffer[88];	p[1] += buffer[89];	p[2] += buffer[90];	p[3] += buffer[91];	p[4] += buffer[92];	p[5] += buffer[93];	p[6] += buffer[94];	p[7] += buffer[95];
		macro2(p);
		p[0] += buffer[96];	p[1] += buffer[97];	p[2] += buffer[98];	p[3] += buffer[99];	p[4] += buffer[100];	p[5] += buffer[101];	p[6] += buffer[102];	p[7] += buffer[103];
		macro1(p);
		p[0] += buffer[104];	p[1] += buffer[105];	p[2] += buffer[106];	p[3] += buffer[107];	p[4] += buffer[108];	p[5] += buffer[109];	p[6] += buffer[110];	p[7] += buffer[111];
		macro2(p);
		p[0] += buffer[112];	p[1] += buffer[113];	p[2] += buffer[114];	p[3] += buffer[115];	p[4] += buffer[116];	p[5] += buffer[117];	p[6] += buffer[118];	p[7] += buffer[119];
		macro1(p);
		p[0] += buffer[120];	p[1] += buffer[121];	p[2] += buffer[122];	p[3] += buffer[123];	p[4] += buffer[124];	p[5] += buffer[125];	p[6] += buffer[126];	p[7] += buffer[127];
		macro2(p);
		p[0] += buffer[128];	p[1] += buffer[129];	p[2] += buffer[130];	p[3] += buffer[131];	p[4] += buffer[132];	p[5] += buffer[133];	p[6] += buffer[134];	p[7] += buffer[135];
		macro1(p);
		p[0] += buffer[136];	p[1] += buffer[137];	p[2] += buffer[138];	p[3] += buffer[139];	p[4] += buffer[140];	p[5] += buffer[141];	p[6] += buffer[142];	p[7] += buffer[143];
		macro2(p);
		p[0] += buffer[144];	p[1] += buffer[145];	p[2] += buffer[146];	p[3] += buffer[147];	p[4] += buffer[148];	p[5] += buffer[149];	p[6] += buffer[150];	p[7] += buffer[151];

		#define h0 p[0]
		#define h1 p[1]
		#define h2 p[2]
		#define h3 p[3]
		#define h4 p[4]
		#define h5 p[5]
		#define h6 p[6]
		#define h7 p[7]

		h0 ^= h[0];	h1 ^= h[1];	h2 ^= h[2];	h3 ^= h[3];	h4 ^= h[4];	h5 ^= h[5];	h6 ^= h[6];	h7 ^= h[7];

		uint2 skein_h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ vectorize(0x1BD11BDAA9FC1A22);

		uint2 hash64[8];

		hash64[5] = h5 + 8U;

		hash64[0] = h0 + h1;
		hash64[1] = ROL2(h1, 46) ^ hash64[0];
		hash64[2] = h2 + h3;
		hash64[3] = ROL2(h3, 36) ^ hash64[2];
		hash64[4] = h4 + hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] = h6 + h7 + t1;
		hash64[7] = ROL2(h7, 37) ^ hash64[6];
		hash64[2]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0]+= hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6]+= hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4]+= hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		
		hash64[0]+= h1;		hash64[1]+= h2;
		hash64[2]+= h3;		hash64[3]+= h4;
		hash64[4]+= h5;		hash64[5]+= h6 + t1;
		hash64[6]+= h7 + t2;	hash64[7]+= skein_h8 + 1U;
		macro2(hash64);
		hash64[0]+= h2;		hash64[1]+= h3;
		hash64[2]+= h4;		hash64[3]+= h5;
		hash64[4]+= h6;		hash64[5]+= h7 + t2;
		hash64[6]+= skein_h8+t0;hash64[7]+= h0 + 2U;
		macro1(hash64);
		hash64[0]+= h3;		hash64[1]+= h4;
		hash64[2]+= h5;		hash64[3]+= h6;
		hash64[4]+= h7;		hash64[5]+= skein_h8 + t0;
		hash64[6]+= h0 + t1;	hash64[7]+= h1 + 3U;
		macro2(hash64);
		hash64[0]+= h4;		hash64[1]+= h5;
		hash64[2]+= h6;		hash64[3]+= h7;
		hash64[4]+= skein_h8;	hash64[5]+= h0 + t1;
		hash64[6]+= h1 + t2;	hash64[7]+= h2 + 4U;
		macro1(hash64);
		hash64[0]+= h5;		hash64[1]+= h6;
		hash64[2]+= h7;		hash64[3]+= skein_h8;
		hash64[4]+= h0;		hash64[5]+= h1 + t2;
		hash64[6]+= h2 + t0;	hash64[7]+= h3 + 5U;
		macro2(hash64);
		hash64[0]+= h6;		hash64[1]+= h7;
		hash64[2]+= skein_h8;	hash64[3]+= h0;
		hash64[4]+= h1;		hash64[5]+= h2 + t0;
		hash64[6]+= h3 + t1;	hash64[7]+= h4 + 6U;
		macro1(hash64);
		hash64[0]+= h7;		hash64[1]+= skein_h8;
		hash64[2]+= h0;		hash64[3]+= h1;
		hash64[4]+= h2;		hash64[5]+= h3 + t1;
		hash64[6]+= h4 + t2;	hash64[7]+= h5 + 7U;
		macro2(hash64);
		hash64[0]+= skein_h8;	hash64[1]+= h0;
		hash64[2]+= h1;		hash64[3]+= h2;
		hash64[4]+= h3;		hash64[5]+= h4 + t2;
		hash64[6]+= h5 + t0;	hash64[7]+= h6 + 8U;
		macro1(hash64);
		hash64[0]+= h0;		hash64[1]+= h1;
		hash64[2]+= h2;		hash64[3]+= h3;
		hash64[4]+= h4;		hash64[5]+= h5 + t0;
		hash64[6]+= h6 + t1;	hash64[7]+= h7 + 9U;
		macro2(hash64);
		hash64[0]+= h1;		hash64[1]+= h2;
		hash64[2]+= h3;		hash64[3]+= h4;
		hash64[4]+= h5;		hash64[5]+= h6 + t1;
		hash64[6]+= h7 + t2;	hash64[7]+= skein_h8 + 10U;
		macro1(hash64);
		hash64[0]+= h2;		hash64[1]+= h3;
		hash64[2]+= h4;		hash64[3]+= h5;
		hash64[4]+= h6;		hash64[5]+= h7 + t2;
		hash64[6]+= skein_h8+t0;hash64[7]+= h0 + 11U;
		macro2(hash64);
		hash64[0]+= h3;		hash64[1]+= h4;
		hash64[2]+= h5;		hash64[3]+= h6;
		hash64[4]+= h7;		hash64[5]+= skein_h8 + t0;
		hash64[6]+= h0 + t1;	hash64[7]+= h1 + 12U;
		macro1(hash64);
		hash64[0]+= h4;		hash64[1]+= h5;
		hash64[2]+= h6;		hash64[3]+= h7;
		hash64[4]+= skein_h8;	hash64[5]+= h0 + t1;
		hash64[6]+= h1 + t2;	hash64[7]+= h2 + 13U;
		macro2(hash64);
		hash64[0]+= h5;		hash64[1]+= h6;
		hash64[2]+= h7;		hash64[3]+= skein_h8;
		hash64[4]+= h0;		hash64[5]+= h1 + t2;
		hash64[6]+= h2 + t0;	hash64[7]+= h3 + 14U;
		macro1(hash64);
		hash64[0]+= h6;		hash64[1]+= h7;
		hash64[2]+= skein_h8;	hash64[3]+= h0;
		hash64[4]+= h1;		hash64[5]+= h2 + t0;
		hash64[6]+= h3 + t1;	hash64[7]+= h4 + 15U;
		macro2(hash64);
		hash64[0]+= h7;		hash64[1]+= skein_h8;
		hash64[2]+= h0;		hash64[3]+= h1;
		hash64[4]+= h2;		hash64[5]+= h3 + t1;
		hash64[6]+= h4 + t2;	hash64[7]+= h5 + 16U;
		macro1(hash64);
		hash64[0]+= skein_h8;	hash64[1]+= h0;
		hash64[2]+= h1;		hash64[3]+= h2;
		hash64[4]+= h3;		hash64[5]+= h4 + t2;
		hash64[6]+= h5 + t0;	hash64[7]+= h6 + 17U;
		macro2(hash64);
		hash64[0]+= h0;		hash64[1]+= h1;
		hash64[2]+= h2;		hash64[3]+= h3;
		hash64[4]+= h4;		hash64[5]+= h5 + t0;
		hash64[6]+= h6 + t1;	hash64[7]+= h7 + 18U;
		
		phash[0] = *(uint2x4*)&hash64[0];
		phash[1] = *(uint2x4*)&hash64[4];

		#undef h0
		#undef h1
		#undef h2
		#undef h3
		#undef h4
		#undef h5
		#undef h6
		#undef h7
	}
}

__host__ void quark_keccak_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	// berechne wie viele Thread Blocks wir brauchen
	const uint32_t dev_id = device_map[thr_id];
	const uint32_t tpb = (device_sm[dev_id] > 500) ? TPB52 : TPB50;
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);

	quark_keccakskein512_gpu_hash_64 << <grid, block >> >(threads, (uint2 *)d_hash, d_nonceVector);
}


__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52,3)
#else
__launch_bounds__(TPB50,7)
#endif
void quark_keccakskein512_gpu_hash_64_final(uint32_t threads, uint2 *g_hash,const uint32_t * g_nonceVector, uint32_t *resNonce,const uint64_t highTarget)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint2 t[5], u[5], v, w;
	uint2 s[25];
	
	if (thread < threads)
	{
		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];

		uint2x4* phash = (uint2x4 *)&g_hash[hashPosition * 8];

		*(uint2x4*)&s[ 0] = __ldg4(&phash[ 0]);
		*(uint2x4*)&s[ 4] = __ldg4(&phash[ 1]);
		
		s[8] = make_uint2(1,0x80000000);

		/*theta*/
		t[ 0] = s[ 0]^s[ 5];
		t[ 1] = s[ 1]^s[ 6];
		t[ 2] = s[ 2]^s[ 7];
		t[ 3] = s[ 3]^s[ 8];
		t[ 4] = s[4];
		
		/*theta*/
		#pragma unroll 5
		for(int j=0;j<5;j++){
			u[ j] = ROL2(t[ j], 1);
		}
		
		s[ 4] = xor3x(s[ 4], t[3], u[ 0]);
		s[24] = s[19] = s[14] = s[ 9] = t[ 3] ^ u[ 0];

		s[ 0] = xor3x(s[ 0], t[4], u[ 1]);
		s[ 5] = xor3x(s[ 5], t[4], u[ 1]);
		s[20] = s[15] = s[10] = t[4] ^ u[ 1];

		s[ 1] = xor3x(s[ 1], t[0], u[ 2]);
		s[ 6] = xor3x(s[ 6], t[0], u[ 2]);
		s[21] = s[16] = s[11] = t[0] ^ u[ 2];
		
		s[ 2] = xor3x(s[ 2], t[1], u[ 3]);
		s[ 7] = xor3x(s[ 7], t[1], u[ 3]);
		s[22] = s[17] = s[12] = t[1] ^ u[ 3];
		
		s[ 3] = xor3x(s[ 3], t[2], u[ 4]);s[ 8] = xor3x(s[ 8], t[2], u[ 4]);
		s[23] = s[18] = s[13] = t[2] ^ u[ 4];
		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1]  = ROL2(s[6], 44);
		s[6]  = ROL2(s[9], 20);
		s[9]  = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2]  = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL8(s[19]);
		s[19] = ROR8(s[23]);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4]  = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8]  = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5]  = ROL2(s[3], 28);
		s[3]  = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7]  = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);
		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		#pragma unroll 5
		for(int j=0;j<25;j+=5){
			v=s[j];w=s[j + 1];s[j] = chi(v,w,s[j+2]);s[j+1] = chi(w,s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
		}
		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[ 0];

		for (int i = 1; i < 23; i++) {
			/*theta*/
			#pragma unroll
			for(int j=0;j<5;j++){
				t[ j] = vectorize(xor5(devectorize(s[ j]),devectorize(s[j+5]),devectorize(s[j+10]),devectorize(s[j+15]),devectorize(s[j+20])));
//				t[ j] = s[ j] ^ s[j+5] ^ s[j+10] ^ s[j+15] ^ s[j+20];
			}

			/*theta*/
			#pragma unroll
			for(int j=0;j<5;j++){
				u[ j] = ROL2(t[ j], 1);
			}
			s[ 4] = xor3x(s[ 4], t[3], u[ 0]);s[ 9] = xor3x(s[ 9], t[3], u[ 0]);s[14] = xor3x(s[14], t[3], u[ 0]);s[19] = xor3x(s[19], t[3], u[ 0]);s[24] = xor3x(s[24], t[3], u[ 0]);
			s[ 0] = xor3x(s[ 0], t[4], u[ 1]);s[ 5] = xor3x(s[ 5], t[4], u[ 1]);s[10] = xor3x(s[10], t[4], u[ 1]);s[15] = xor3x(s[15], t[4], u[ 1]);s[20] = xor3x(s[20], t[4], u[ 1]);
			s[ 1] = xor3x(s[ 1], t[0], u[ 2]);s[ 6] = xor3x(s[ 6], t[0], u[ 2]);s[11] = xor3x(s[11], t[0], u[ 2]);s[16] = xor3x(s[16], t[0], u[ 2]);s[21] = xor3x(s[21], t[0], u[ 2]);
			s[ 2] = xor3x(s[ 2], t[1], u[ 3]);s[ 7] = xor3x(s[ 7], t[1], u[ 3]);s[12] = xor3x(s[12], t[1], u[ 3]);s[17] = xor3x(s[17], t[1], u[ 3]);s[22] = xor3x(s[22], t[1], u[ 3]);
			s[ 3] = xor3x(s[ 3], t[2], u[ 4]);s[ 8] = xor3x(s[ 8], t[2], u[ 4]);s[13] = xor3x(s[13], t[2], u[ 4]);s[18] = xor3x(s[18], t[2], u[ 4]);s[23] = xor3x(s[23], t[2], u[ 4]);
			
/*			s[ 4] = s[ 4]^t[3]^u[ 0];s[ 9] = s[ 9]^t[3]^u[ 0];s[14] = s[14]^t[3]^u[ 0];s[19] = s[19]^t[3]^u[ 0];s[24] = s[24]^t[3]^u[ 0];
			s[ 0] = s[ 0]^t[4]^u[ 1];s[ 5] = s[ 5]^t[4]^u[ 1];s[10] = s[10]^t[4]^u[ 1];s[15] = s[15]^t[4]^u[ 1];s[20] = s[20]^t[4]^u[ 1];
			s[ 1] = s[ 1]^t[0]^u[ 2];s[ 6] = s[ 6]^t[0]^u[ 2];s[11] = s[11]^t[0]^u[ 2];s[16] = s[16]^t[0]^u[ 2];s[21] = s[21]^t[0]^u[ 2];
			s[ 2] = s[ 2]^t[1]^u[ 3];s[ 7] = s[ 7]^t[1]^u[ 3];s[12] = s[12]^t[1]^u[ 3];s[17] = s[17]^t[1]^u[ 3];s[22] = s[22]^t[1]^u[ 3];
			s[ 3] = s[ 3]^t[2]^u[ 4];s[ 8] = s[ 8]^t[2]^u[ 4];s[13] = s[13]^t[2]^u[ 4];s[18] = s[18]^t[2]^u[ 4];s[23] = s[23]^t[2]^u[ 4];*/

			/* rho pi: b[..] = rotl(a[..], ..) */
			v = s[1];
			s[1]  = ROL2(s[6], 44);
			s[6]  = ROL2(s[9], 20);
			s[9]  = ROL2(s[22], 61);
			s[22] = ROL2(s[14], 39);
			s[14] = ROL2(s[20], 18);
			s[20] = ROL2(s[2], 62);
			s[2]  = ROL2(s[12], 43);
			s[12] = ROL2(s[13], 25);
			s[13] = ROL8(s[19]);
			s[19] = ROR8(s[23]);
			s[23] = ROL2(s[15], 41);
			s[15] = ROL2(s[4], 27);
			s[4]  = ROL2(s[24], 14);
			s[24] = ROL2(s[21], 2);
			s[21] = ROL2(s[8], 55);
			s[8]  = ROL2(s[16], 45);
			s[16] = ROL2(s[5], 36);
			s[5]  = ROL2(s[3], 28);
			s[3]  = ROL2(s[18], 21);
			s[18] = ROL2(s[17], 15);
			s[17] = ROL2(s[11], 10);
			s[11] = ROL2(s[7], 6);
			s[7]  = ROL2(s[10], 3);
			s[10] = ROL2(v, 1);

			/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
			#pragma unroll
			for(int j=0;j<25;j+=5){
				v=s[j];w=s[j + 1];s[j] = chi(v,w,s[j+2]);s[j+1] = chi(w,s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
			}

			/* iota: a[0,0] ^= round constant */
			s[0] ^= keccak_round_constants[i];
		}
		/*theta*/
		#pragma unroll
		for(int j=0;j<5;j++){
			t[ j] = xor3x(xor3x(s[j+0],s[j+5],s[j+10]),s[j+15],s[j+20]);
		}
		/*theta*/
		#pragma unroll
		for(int j=0;j<5;j++){
			u[ j] = ROL2(t[ j], 1);
		}
		s[ 9] = xor3x(s[ 9], t[3], u[ 0]);
		s[24] = xor3x(s[24], t[3], u[ 0]);
		s[ 0] = xor3x(s[ 0], t[4], u[ 1]);
		s[10] = xor3x(s[10], t[4], u[ 1]);
		s[ 6] = xor3x(s[ 6], t[0], u[ 2]);
		s[16] = xor3x(s[16], t[0], u[ 2]);
		s[12] = xor3x(s[12], t[1], u[ 3]);
		s[22] = xor3x(s[22], t[1], u[ 3]);
		s[ 3] = xor3x(s[ 3], t[2], u[ 4]);
		s[18] = xor3x(s[18], t[2], u[ 4]);
		/* rho pi: b[..] = rotl(a[..], ..) */
		s[ 1]  = ROL2(s[ 6], 44);
		s[ 2]  = ROL2(s[12], 43);
		s[ 5]  = ROL2(s[ 3], 28);
		s[ 7]  = ROL2(s[10], 3);
		s[ 3]  = ROL2(s[18], 21);
		s[ 4]  = ROL2(s[24], 14);
		s[ 6]  = ROL2(s[ 9], 20);
		s[ 8]  = ROL2(s[16], 45);
		s[ 9]  = ROL2(s[22], 61);

		uint2 p[8],h[9];
		
		uint32_t t0;
		uint2 t1,t2;
		t0 = 8;
		t1 = vectorize(0xFF00000000000000);
		t2 = t1+t0;
				
		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		p[ 0] = chi(s[ 0],s[ 1],s[ 2]);
		p[ 1] = chi(s[ 1],s[ 2],s[ 3]);
		p[ 2] = chi(s[ 2],s[ 3],s[ 4]);
		p[ 3] = chi(s[ 3],s[ 4],s[ 0]);
		p[ 4] = chi(s[ 4],s[ 0],s[ 1]);
		p[ 5] = chi(s[ 5],s[ 6],s[ 7]);
		p[ 6] = chi(s[ 6],s[ 7],s[ 8]);
		p[ 7] = chi(s[ 7],s[ 8],s[ 9]);
		/* iota: a[0,0] ^= round constant */
		p[0] ^= keccak_round_constants[23];
		
		h[ 0] = p[0];
		h[ 1] = p[1];
		h[ 2] = p[2];
		h[ 3] = p[3];
		h[ 4] = p[4];
		h[ 5] = p[5];
		h[ 6] = p[6];
		h[ 7] = p[7];
		
		p[0] += buffer[0];	p[1] += buffer[1];	p[2] += buffer[2];	p[3] += buffer[3];	p[4] += buffer[4];	p[5] += buffer[5];	p[6] += buffer[6];	p[7] += buffer[7];
		macro1(p);
		p[0] += buffer[8];	p[1] += buffer[9];	p[2] += buffer[10];	p[3] += buffer[11];	p[4] += buffer[12];	p[5] += buffer[13];	p[6] += buffer[14];	p[7] += buffer[15];
		macro2(p);
		p[0] += buffer[16];	p[1] += buffer[17];	p[2] += buffer[18];	p[3] += buffer[19];	p[4] += buffer[20];	p[5] += buffer[21];	p[6] += buffer[22];	p[7] += buffer[23];
		macro1(p);
		p[0] += buffer[24];	p[1] += buffer[25];	p[2] += buffer[26];	p[3] += buffer[27];	p[4] += buffer[28];	p[5] += buffer[29];	p[6] += buffer[30];	p[7] += buffer[31];
		macro2(p);
		p[0] += buffer[32];	p[1] += buffer[33];	p[2] += buffer[34];	p[3] += buffer[35];	p[4] += buffer[36];	p[5] += buffer[37];	p[6] += buffer[38];	p[7] += buffer[39];
		macro1(p);
		p[0] += buffer[40];	p[1] += buffer[41];	p[2] += buffer[42];	p[3] += buffer[43];	p[4] += buffer[44];	p[5] += buffer[45];	p[6] += buffer[46];	p[7] += buffer[47];
		macro2(p);
		p[0] += buffer[48];	p[1] += buffer[49];	p[2] += buffer[50];	p[3] += buffer[51];	p[4] += buffer[52];	p[5] += buffer[53];	p[6] += buffer[54];	p[7] += buffer[55];
		macro1(p);
		p[0] += buffer[56];	p[1] += buffer[57];	p[2] += buffer[58];	p[3] += buffer[59];	p[4] += buffer[60];	p[5] += buffer[61];	p[6] += buffer[62];	p[7] += buffer[63];
		macro2(p);
		p[0] += buffer[64];	p[1] += buffer[65];	p[2] += buffer[66];	p[3] += buffer[67];	p[4] += buffer[68];	p[5] += buffer[69];	p[6] += buffer[70];	p[7] += buffer[71];
		macro1(p);
		p[0] += buffer[72];	p[1] += buffer[73];	p[2] += buffer[74];	p[3] += buffer[75];	p[4] += buffer[76];	p[5] += buffer[77];	p[6] += buffer[78];	p[7] += buffer[79];
		macro2(p);
		p[0] += buffer[80];	p[1] += buffer[81];	p[2] += buffer[82];	p[3] += buffer[83];	p[4] += buffer[84];	p[5] += buffer[85];	p[6] += buffer[86];	p[7] += buffer[87];
		macro1(p);
		p[0] += buffer[88];	p[1] += buffer[89];	p[2] += buffer[90];	p[3] += buffer[91];	p[4] += buffer[92];	p[5] += buffer[93];	p[6] += buffer[94];	p[7] += buffer[95];
		macro2(p);
		p[0] += buffer[96];	p[1] += buffer[97];	p[2] += buffer[98];	p[3] += buffer[99];	p[4] += buffer[100];	p[5] += buffer[101];	p[6] += buffer[102];	p[7] += buffer[103];
		macro1(p);
		p[0] += buffer[104];	p[1] += buffer[105];	p[2] += buffer[106];	p[3] += buffer[107];	p[4] += buffer[108];	p[5] += buffer[109];	p[6] += buffer[110];	p[7] += buffer[111];
		macro2(p);
		p[0] += buffer[112];	p[1] += buffer[113];	p[2] += buffer[114];	p[3] += buffer[115];	p[4] += buffer[116];	p[5] += buffer[117];	p[6] += buffer[118];	p[7] += buffer[119];
		macro1(p);
		p[0] += buffer[120];	p[1] += buffer[121];	p[2] += buffer[122];	p[3] += buffer[123];	p[4] += buffer[124];	p[5] += buffer[125];	p[6] += buffer[126];	p[7] += buffer[127];
		macro2(p);
		p[0] += buffer[128];	p[1] += buffer[129];	p[2] += buffer[130];	p[3] += buffer[131];	p[4] += buffer[132];	p[5] += buffer[133];	p[6] += buffer[134];	p[7] += buffer[135];
		macro1(p);
		p[0] += buffer[136];	p[1] += buffer[137];	p[2] += buffer[138];	p[3] += buffer[139];	p[4] += buffer[140];	p[5] += buffer[141];	p[6] += buffer[142];	p[7] += buffer[143];
		macro2(p);
		p[0] += buffer[144];	p[1] += buffer[145];	p[2] += buffer[146];	p[3] += buffer[147];	p[4] += buffer[148];	p[5] += buffer[149];	p[6] += buffer[150];	p[7] += buffer[151];

		#define h0 p[0]
		#define h1 p[1]
		#define h2 p[2]
		#define h3 p[3]
		#define h4 p[4]
		#define h5 p[5]
		#define h6 p[6]
		#define h7 p[7]

		h0 ^= h[0];	h1 ^= h[1];	h2 ^= h[2];	h3 ^= h[3];	h4 ^= h[4];	h5 ^= h[5];	h6 ^= h[6];	h7 ^= h[7];

		uint2 skein_h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ vectorize(0x1BD11BDAA9FC1A22);

		uint2 hash64[8];

		hash64[5] = h5 + 8U;

		hash64[0] = h0 + h1;
		hash64[1] = ROL2(h1, 46) ^ hash64[0];
		hash64[2] = h2 + h3;
		hash64[3] = ROL2(h3, 36) ^ hash64[2];
		hash64[4] = h4 + hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] = h6 + h7 + t1;
		hash64[7] = ROL2(h7, 37) ^ hash64[6];
		hash64[2]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0]+= hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6]+= hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6]+= hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0]+= hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2]+= hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4]+= hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		
		hash64[0]+= h1;		hash64[1]+= h2;
		hash64[2]+= h3;		hash64[3]+= h4;
		hash64[4]+= h5;		hash64[5]+= h6 + t1;
		hash64[6]+= h7 + t2;	hash64[7]+= skein_h8 + 1U;
		macro2(hash64);
		hash64[0]+= h2;		hash64[1]+= h3;
		hash64[2]+= h4;		hash64[3]+= h5;
		hash64[4]+= h6;		hash64[5]+= h7 + t2;
		hash64[6]+= skein_h8+t0;hash64[7]+= h0 + 2U;
		macro1(hash64);
		hash64[0]+= h3;		hash64[1]+= h4;
		hash64[2]+= h5;		hash64[3]+= h6;
		hash64[4]+= h7;		hash64[5]+= skein_h8 + t0;
		hash64[6]+= h0 + t1;	hash64[7]+= h1 + 3U;
		macro2(hash64);
		hash64[0]+= h4;		hash64[1]+= h5;
		hash64[2]+= h6;		hash64[3]+= h7;
		hash64[4]+= skein_h8;	hash64[5]+= h0 + t1;
		hash64[6]+= h1 + t2;	hash64[7]+= h2 + 4U;
		macro1(hash64);
		hash64[0]+= h5;		hash64[1]+= h6;
		hash64[2]+= h7;		hash64[3]+= skein_h8;
		hash64[4]+= h0;		hash64[5]+= h1 + t2;
		hash64[6]+= h2 + t0;	hash64[7]+= h3 + 5U;
		macro2(hash64);
		hash64[0]+= h6;		hash64[1]+= h7;
		hash64[2]+= skein_h8;	hash64[3]+= h0;
		hash64[4]+= h1;		hash64[5]+= h2 + t0;
		hash64[6]+= h3 + t1;	hash64[7]+= h4 + 6U;
		macro1(hash64);
		hash64[0]+= h7;		hash64[1]+= skein_h8;
		hash64[2]+= h0;		hash64[3]+= h1;
		hash64[4]+= h2;		hash64[5]+= h3 + t1;
		hash64[6]+= h4 + t2;	hash64[7]+= h5 + 7U;
		macro2(hash64);
		hash64[0]+= skein_h8;	hash64[1]+= h0;
		hash64[2]+= h1;		hash64[3]+= h2;
		hash64[4]+= h3;		hash64[5]+= h4 + t2;
		hash64[6]+= h5 + t0;	hash64[7]+= h6 + 8U;
		macro1(hash64);
		hash64[0]+= h0;		hash64[1]+= h1;
		hash64[2]+= h2;		hash64[3]+= h3;
		hash64[4]+= h4;		hash64[5]+= h5 + t0;
		hash64[6]+= h6 + t1;	hash64[7]+= h7 + 9U;
		macro2(hash64);
		hash64[0]+= h1;		hash64[1]+= h2;
		hash64[2]+= h3;		hash64[3]+= h4;
		hash64[4]+= h5;		hash64[5]+= h6 + t1;
		hash64[6]+= h7 + t2;	hash64[7]+= skein_h8 + 10U;
		macro1(hash64);
		hash64[0]+= h2;		hash64[1]+= h3;
		hash64[2]+= h4;		hash64[3]+= h5;
		hash64[4]+= h6;		hash64[5]+= h7 + t2;
		hash64[6]+= skein_h8+t0;hash64[7]+= h0 + 11U;
		macro2(hash64);
		hash64[0]+= h3;		hash64[1]+= h4;
		hash64[2]+= h5;		hash64[3]+= h6;
		hash64[4]+= h7;		hash64[5]+= skein_h8 + t0;
		hash64[6]+= h0 + t1;	hash64[7]+= h1 + 12U;
		macro1(hash64);
		hash64[0]+= h4;		hash64[1]+= h5;
		hash64[2]+= h6;		hash64[3]+= h7;
		hash64[4]+= skein_h8;	hash64[5]+= h0 + t1;
		hash64[6]+= h1 + t2;	hash64[7]+= h2 + 13U;
		macro2(hash64);
		hash64[0]+= h5;		hash64[1]+= h6;
		hash64[2]+= h7;		hash64[3]+= skein_h8;
		hash64[4]+= h0;		hash64[5]+= h1 + t2;
		hash64[6]+= h2 + t0;	hash64[7]+= h3 + 14U;
		macro1(hash64);
		hash64[0]+= h6;		hash64[1]+= h7;
		hash64[2]+= skein_h8;	hash64[3]+= h0;
		hash64[4]+= h1;		hash64[5]+= h2 + t0;
		hash64[6]+= h3 + t1;	hash64[7]+= h4 + 15U;
		macro2(hash64);
		hash64[0]+= h7;		hash64[1]+= skein_h8;
		hash64[2]+= h0;		hash64[3]+= h1;
		hash64[4]+= h2;		hash64[5]+= h3 + t1;
		hash64[6]+= h4 + t2;	hash64[7]+= h5 + 16U;
		macro1(hash64);
		hash64[0]+= skein_h8;	hash64[1]+= h0;
		hash64[2]+= h1;		hash64[3]+= h2;
		hash64[4]+= h3;		hash64[5]+= h4 + t2;
		hash64[6]+= h5 + t0;	hash64[7]+= h6 + 17U;
//		macro2(hash64);
		hash64[0] += hash64[1];
		hash64[2] += hash64[3];
		hash64[4] += hash64[5];
		hash64[6] += hash64[7];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];

		hash64[2] += hash64[1];
		
		hash64[4] += hash64[7];
		
		hash64[6] += hash64[5];
		
		hash64[0] += hash64[3];
		
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];

		hash64[4] += hash64[1];
		hash64[6] += hash64[3];
		
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];

		if(devectorize(hash64[3]+h3)<=highTarget){
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;		
		}	
//		phash[0] = *(uint2x4*)&hash64[0];
//		phash[1] = *(uint2x4*)&hash64[4];

		#undef h0
		#undef h1
		#undef h2
		#undef h3
		#undef h4
		#undef h5
		#undef h6
		#undef h7
	}
}

__host__ void quark_keccak_skein512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash,uint32_t *d_resNonce,const uint64_t highTarget){

	// berechne wie viele Thread Blocks wir brauchen
	const uint32_t dev_id = device_map[thr_id];
	const uint32_t tpb = (device_sm[dev_id] > 500) ? TPB52 : TPB50;
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);

	quark_keccakskein512_gpu_hash_64_final <<<grid, block >>>(threads, (uint2 *)d_hash, d_nonceVector, d_resNonce, highTarget);
}
