/* Based on SP's work
 * 
 * Provos Alexis - 2016
 */

#include "miner.h"
#include "cuda_vectors.h"
#include "skein/skein_header.h"

#define TPB52 512
#define TPB50 512

/* ************************ */
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
__launch_bounds__(TPB52, 3)
#else
__launch_bounds__(TPB50, 3)
#endif
void quark_skein512_gpu_hash_64(const uint32_t threads,uint64_t* g_hash, const uint32_t* g_nonceVector){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		// Skein
		uint2 p[8], h[9];

		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread];

		uint64_t *Hash = &g_hash[hashPosition<<3];

		uint2x4 *phash = (uint2x4*)Hash;
		*(uint2x4*)&p[0] = __ldg4(&phash[0]);
		*(uint2x4*)&p[4] = __ldg4(&phash[1]);
		
		h[0] = p[0];	h[1] = p[1];	h[2] = p[2];	h[3] = p[3];
		h[4] = p[4];	h[5] = p[5];	h[6] = p[6];	h[7] = p[7];

		p[0] += buffer[0];	p[1] += buffer[1];	p[2] += buffer[2];	p[3] += buffer[3];	p[4] += buffer[4];	p[5] += buffer[5];	p[6] += buffer[6];	p[7] += buffer[7];
		TFBIGMIX8e();
		p[0] += buffer[8];	p[1] += buffer[9];	p[2] += buffer[10];	p[3] += buffer[11];	p[4] += buffer[12];	p[5] += buffer[13];	p[6] += buffer[14];	p[7] += buffer[15];
		TFBIGMIX8o();
		p[0] += buffer[16];	p[1] += buffer[17];	p[2] += buffer[18];	p[3] += buffer[19];	p[4] += buffer[20];	p[5] += buffer[21];	p[6] += buffer[22];	p[7] += buffer[23];
		TFBIGMIX8e();
		p[0] += buffer[24];	p[1] += buffer[25];	p[2] += buffer[26];	p[3] += buffer[27];	p[4] += buffer[28];	p[5] += buffer[29];	p[6] += buffer[30];	p[7] += buffer[31];
		TFBIGMIX8o();
		p[0] += buffer[32];	p[1] += buffer[33];	p[2] += buffer[34];	p[3] += buffer[35];	p[4] += buffer[36];	p[5] += buffer[37];	p[6] += buffer[38];	p[7] += buffer[39];
		TFBIGMIX8e();
		p[0] += buffer[40];	p[1] += buffer[41];	p[2] += buffer[42];	p[3] += buffer[43];	p[4] += buffer[44];	p[5] += buffer[45];	p[6] += buffer[46];	p[7] += buffer[47];
		TFBIGMIX8o();
		p[0] += buffer[48];	p[1] += buffer[49];	p[2] += buffer[50];	p[3] += buffer[51];	p[4] += buffer[52];	p[5] += buffer[53];	p[6] += buffer[54];	p[7] += buffer[55];
		TFBIGMIX8e();
		p[0] += buffer[56];	p[1] += buffer[57];	p[2] += buffer[58];	p[3] += buffer[59];	p[4] += buffer[60];	p[5] += buffer[61];	p[6] += buffer[62];	p[7] += buffer[63];
		TFBIGMIX8o();
		p[0] += buffer[64];	p[1] += buffer[65];	p[2] += buffer[66];	p[3] += buffer[67];	p[4] += buffer[68];	p[5] += buffer[69];	p[6] += buffer[70];	p[7] += buffer[71];
		TFBIGMIX8e();
		p[0] += buffer[72];	p[1] += buffer[73];	p[2] += buffer[74];	p[3] += buffer[75];	p[4] += buffer[76];	p[5] += buffer[77];	p[6] += buffer[78];	p[7] += buffer[79];
		TFBIGMIX8o();
		p[0] += buffer[80];	p[1] += buffer[81];	p[2] += buffer[82];	p[3] += buffer[83];	p[4] += buffer[84];	p[5] += buffer[85];	p[6] += buffer[86];	p[7] += buffer[87];
		TFBIGMIX8e();
		p[0] += buffer[88];	p[1] += buffer[89];	p[2] += buffer[90];	p[3] += buffer[91];	p[4] += buffer[92];	p[5] += buffer[93];	p[6] += buffer[94];	p[7] += buffer[95];
		TFBIGMIX8o();
		p[0] += buffer[96];	p[1] += buffer[97];	p[2] += buffer[98];	p[3] += buffer[99];	p[4] += buffer[100];	p[5] += buffer[101];	p[6] += buffer[102];	p[7] += buffer[103];
		TFBIGMIX8e();
		p[0] += buffer[104];	p[1] += buffer[105];	p[2] += buffer[106];	p[3] += buffer[107];	p[4] += buffer[108];	p[5] += buffer[109];	p[6] += buffer[110];	p[7] += buffer[111];
		TFBIGMIX8o();
		p[0] += buffer[112];	p[1] += buffer[113];	p[2] += buffer[114];	p[3] += buffer[115];	p[4] += buffer[116];	p[5] += buffer[117];	p[6] += buffer[118];	p[7] += buffer[119];
		TFBIGMIX8e();
		p[0] += buffer[120];	p[1] += buffer[121];	p[2] += buffer[122];	p[3] += buffer[123];	p[4] += buffer[124];	p[5] += buffer[125];	p[6] += buffer[126];	p[7] += buffer[127];
		TFBIGMIX8o();
		p[0] += buffer[128];	p[1] += buffer[129];	p[2] += buffer[130];	p[3] += buffer[131];	p[4] += buffer[132];	p[5] += buffer[133];	p[6] += buffer[134];	p[7] += buffer[135];
		TFBIGMIX8e();
		p[0] += buffer[136];	p[1] += buffer[137];	p[2] += buffer[138];	p[3] += buffer[139];	p[4] += buffer[140];	p[5] += buffer[141];	p[6] += buffer[142];	p[7] += buffer[143];
		TFBIGMIX8o();
		p[0] += buffer[144];	p[1] += buffer[145];	p[2] += buffer[146];	p[3] += buffer[147];	p[4] += buffer[148];	p[5] += buffer[149];	p[6] += buffer[150];	p[7] += buffer[151];

		h[0]^= p[0];
		h[1]^= p[1];
		h[2]^= p[2];
		h[3]^= p[3];
		h[4]^= p[4];
		h[5]^= p[5];
		h[6]^= p[6];
		h[7]^= p[7];
		
		h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ vectorize(0x1BD11BDAA9FC1A22);

		uint32_t t0;
		uint2 t1,t2;
		t0 = 8;
		t1 = vectorize(0xFF00000000000000);
		t2 = t1+t0;

		p[5] = h[5] + 8U;

		p[0] = h[0] + h[1];
		p[1] = ROL2(h[1], 46) ^ p[0];
		p[2] = h[2] + h[3];
		p[3] = ROL2(h[3], 36) ^ p[2];
		p[4] = h[4] + p[5];
		p[5] = ROL2(p[5], 19) ^ p[4];
		p[6] = (h[6] + h[7] + t1);
		p[7] = ROL2(h[7], 37) ^ p[6];
		p[2]+= p[1];
		p[1] = ROL2(p[1], 33) ^ p[2];
		p[4]+= p[7];
		p[7] = ROL2(p[7], 27) ^ p[4];
		p[6]+= p[5];
		p[5] = ROL2(p[5], 14) ^ p[6];
		p[0]+= p[3];
		p[3] = ROL2(p[3], 42) ^ p[0];
		p[4]+= p[1];
		p[1] = ROL2(p[1], 17) ^ p[4];
		p[6]+= p[3];
		p[3] = ROL2(p[3], 49) ^ p[6];
		p[0]+= p[5];
		p[5] = ROL2(p[5], 36) ^ p[0];
		p[2]+= p[7];
		p[7] = ROL2(p[7], 39) ^ p[2];
		p[6]+= p[1];
		p[1] = ROL2(p[1], 44) ^ p[6];
		p[0]+= p[7];
		p[7] = ROL2(p[7], 9) ^ p[0];
		p[2]+= p[5];
		p[5] = ROL2(p[5], 54) ^ p[2];
		p[4]+= p[3];
		p[3] = ROR8(p[3]) ^ p[4];
		
		p[0]+= h[1];		p[1]+= h[2];
		p[2]+= h[3];		p[3]+= h[4];
		p[4]+= h[5];		p[5]+= h[6] + t1;
		p[6]+= h[7] + t2;	p[7]+= h[8] + 1U;
		TFBIGMIX8o();
		p[0]+= h[2];		p[1]+= h[3];
		p[2]+= h[4];		p[3]+= h[5];
		p[4]+= h[6];		p[5]+= h[7] + t2;
		p[6]+= h[8] + t0;	p[7]+= h[0] + 2U;
		TFBIGMIX8e();
		p[0]+= h[3];		p[1]+= h[4];
		p[2]+= h[5];		p[3]+= h[6];
		p[4]+= h[7];		p[5]+= h[8] + t0;
		p[6]+= h[0] + t1;	p[7]+= h[1] + 3U;
		TFBIGMIX8o();
		p[0]+= h[4];		p[1]+= h[5];
		p[2]+= h[6];		p[3]+= h[7];
		p[4]+= h[8];		p[5]+= h[0] + t1;
		p[6]+= h[1] + t2;	p[7]+= h[2] + 4U;
		TFBIGMIX8e();
		p[0]+= h[5];		p[1]+= h[6];
		p[2]+= h[7];		p[3]+= h[8];
		p[4]+= h[0];		p[5]+= h[1] + t2;
		p[6]+= h[2] + t0;	p[7]+= h[3] + 5U;
		TFBIGMIX8o();
		p[0]+= h[6];		p[1]+= h[7];
		p[2]+= h[8];		p[3]+= h[0];
		p[4]+= h[1];		p[5]+= h[2] + t0;
		p[6]+= h[3] + t1;	p[7]+= h[4] + 6U;
		TFBIGMIX8e();
		p[0]+= h[7];		p[1]+= h[8];
		p[2]+= h[0];		p[3]+= h[1];
		p[4]+= h[2];		p[5]+= h[3] + t1;
		p[6]+= h[4] + t2;	p[7]+= h[5] + 7U;
		TFBIGMIX8o();
		p[0]+= h[8];		p[1]+= h[0];
		p[2]+= h[1];		p[3]+= h[2];
		p[4]+= h[3];		p[5]+= h[4] + t2;
		p[6]+= h[5] + t0;	p[7]+= h[6] + 8U;
		TFBIGMIX8e();
		p[0]+= h[0];		p[1]+= h[1];
		p[2]+= h[2];		p[3]+= h[3];
		p[4]+= h[4];		p[5]+= h[5] + t0;
		p[6]+= h[6] + t1;	p[7]+= h[7] + 9U;
		TFBIGMIX8o();
		p[0] = p[0] + h[1];	p[1] = p[1] + h[2];
		p[2] = p[2] + h[3];	p[3] = p[3] + h[4];
		p[4] = p[4] + h[5];	p[5] = p[5] + h[6] + t1;
		p[6] = p[6] + h[7] + t2;p[7] = p[7] + h[8] + 10U;
		TFBIGMIX8e();
		p[0] = p[0] + h[2];	p[1] = p[1] + h[3];
		p[2] = p[2] + h[4];	p[3] = p[3] + h[5];
		p[4] = p[4] + h[6];	p[5] = p[5] + h[7] + t2;
		p[6] = p[6] + h[8] + t0;p[7] = p[7] + h[0] + 11U;
		TFBIGMIX8o();
		p[0] = p[0] + h[3];	p[1] = p[1] + h[4];
		p[2] = p[2] + h[5];	p[3] = p[3] + h[6];
		p[4] = p[4] + h[7];	p[5] = p[5] + h[8] + t0;
		p[6] = p[6] + h[0] + t1;p[7] = p[7] + h[1] + 12U;
		TFBIGMIX8e();
		p[0] = p[0] + h[4];	p[1] = p[1] + h[5];
		p[2] = p[2] + h[6];	p[3] = p[3] + h[7];
		p[4] = p[4] + h[8];	p[5] = p[5] + h[0] + t1;
		p[6] = p[6] + h[1] + t2;p[7] = p[7] + h[2] + 13U;
		TFBIGMIX8o();
		p[0] = p[0] + h[5];	p[1] = p[1] + h[6];
		p[2] = p[2] + h[7];	p[3] = p[3] + h[8];
		p[4] = p[4] + h[0];	p[5] = p[5] + h[1] + t2;
		p[6] = p[6] + h[2] + t0;p[7] = p[7] + h[3] + 14U;
		TFBIGMIX8e();
		p[0] = p[0] + h[6];	p[1] = p[1] + h[7];
		p[2] = p[2] + h[8];	p[3] = p[3] + h[0];
		p[4] = p[4] + h[1];	p[5] = p[5] + h[2] + t0;
		p[6] = p[6] + h[3] + t1;p[7] = p[7] + h[4] + 15U;
		TFBIGMIX8o();
		p[0] = p[0] + h[7];	p[1] = p[1] + h[8];
		p[2] = p[2] + h[0];	p[3] = p[3] + h[1];
		p[4] = p[4] + h[2];	p[5] = p[5] + h[3] + t1;
		p[6] = p[6] + h[4] + t2;p[7] = p[7] + h[5] + 16U;
		TFBIGMIX8e();
		p[0] = p[0] + h[8];	p[1] = p[1] + h[0];
		p[2] = p[2] + h[1];	p[3] = p[3] + h[2];
		p[4] = p[4] + h[3];	p[5] = p[5] + h[4] + t2;
		p[6] = p[6] + h[5] + t0;p[7] = p[7] + h[6] + 17U;
		TFBIGMIX8o();
		p[0] = p[0] + h[0];	p[1] = p[1] + h[1];
		p[2] = p[2] + h[2];	p[3] = p[3] + h[3];
		p[4] = p[4] + h[4];	p[5] = p[5] + h[5] + t0;
		p[6] = p[6] + h[6] + t1;p[7] = p[7] + h[7] + 18U;
		
		phash = (uint2x4*)p;
		uint2x4 *outpt = (uint2x4*)Hash;
		outpt[0] = phash[0];
		outpt[1] = phash[1];

	}
}

__host__
void quark_skein512_cpu_hash_64(int thr_id,uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	uint32_t tpb = TPB52;
	int dev_id = device_map[thr_id];
	
	if (device_sm[dev_id] <= 500) tpb = TPB50;
	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);
	quark_skein512_gpu_hash_64 << <grid, block >> >(threads, (uint64_t*)d_hash, d_nonceVector);

}

static __constant__ uint2 c_buffer[120]; // padded message (80 bytes + 72 bytes midstate + align)

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52, 3)
#else
__launch_bounds__(TPB50, 5)
#endif
void skein512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *output64){

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint2 h[9];
		uint2 t0, t1, t2;

		uint32_t nonce = cuda_swab32(startNounce + thread);
		uint2 nonce2 = make_uint2(c_buffer[0].x, nonce);

		uint2 p[8];
		p[1] = nonce2;

		h[0] = c_buffer[ 1];
		h[1] = c_buffer[ 2];
		h[2] = c_buffer[ 3];
		h[3] = c_buffer[ 4];
		h[4] = c_buffer[ 5];
		h[5] = c_buffer[ 6];
		h[6] = c_buffer[ 7];
		h[7] = c_buffer[ 8];
		h[8] = c_buffer[ 9];

		t0 = vectorize(0x50ull);
		t1 = vectorize(0xB000000000000000ull);
		t2 = t0^t1;
		
		p[ 1]=nonce2 + h[1];	p[ 0]= c_buffer[10] + p[ 1];
		p[ 2]=c_buffer[11];
		p[ 3]=c_buffer[12];
		p[ 4]=c_buffer[13];
		p[ 5]=c_buffer[14];
		p[ 6]=c_buffer[15];
		p[ 7]=c_buffer[16];
		
//		TFBIGMIX8e();
		p[1] = ROL2(p[1], 46) ^ p[0];
		p[2] += p[1];
		p[0] += p[3];
		p[1] = ROL2(p[1], 33) ^ p[2];
		p[3] = c_buffer[17] ^ p[0];
		p[4] += p[1];
		p[6] += p[3];
		p[0] += p[5];
		p[2] += p[7];
		p[1] = ROL2(p[1], 17) ^ p[4];
		p[3] = ROL2(p[3], 49) ^ p[6];
		p[5] = c_buffer[18] ^ p[0];
		p[7] = c_buffer[19] ^ p[2];
		p[6] += p[1];
		p[0] += p[7];
		p[2] += p[5];
		p[4] += p[3];
		p[1] = ROL2(p[1], 44) ^ p[6];
		p[7] = ROL2(p[7], 9) ^ p[0];
		p[5] = ROL2(p[5], 54) ^ p[2];
		p[3] = ROR8(p[3]) ^ p[4];
				
		p[ 0]+=h[1];	p[ 1]+=h[2];	p[ 2]+=h[3];	p[ 3]+=h[4];	p[ 4]+=h[5];	p[ 5]+=c_buffer[20];	p[ 7]+=c_buffer[21];	p[ 6]+=c_buffer[22];
		TFBIGMIX8o();
		p[ 0]+=h[2];	p[ 1]+=h[3];	p[ 2]+=h[4];	p[ 3]+=h[5];	p[ 4]+=h[6];	p[ 5]+=c_buffer[22];	p[ 7]+=c_buffer[23];	p[ 6]+=c_buffer[24];
		TFBIGMIX8e();
		p[ 0]+=h[3];	p[ 1]+=h[4];	p[ 2]+=h[5];	p[ 3]+=h[6];	p[ 4]+=h[7];	p[ 5]+=c_buffer[24];	p[ 7]+=c_buffer[25];	p[ 6]+=c_buffer[26];
		TFBIGMIX8o();
		p[ 0]+=h[4];	p[ 1]+=h[5];	p[ 2]+=h[6];	p[ 3]+=h[7];	p[ 4]+=h[8];	p[ 5]+=c_buffer[26];	p[ 7]+=c_buffer[27];	p[ 6]+=c_buffer[28];
		TFBIGMIX8e();
		p[ 0]+=h[5];	p[ 1]+=h[6];	p[ 2]+=h[7];	p[ 3]+=h[8];	p[ 4]+=h[0];	p[ 5]+=c_buffer[28];	p[ 7]+=c_buffer[29];	p[ 6]+=c_buffer[30];
		TFBIGMIX8o();
		p[ 0]+=h[6];	p[ 1]+=h[7];	p[ 2]+=h[8];	p[ 3]+=h[0];	p[ 4]+=h[1];	p[ 5]+=c_buffer[30];	p[ 7]+=c_buffer[31];	p[ 6]+=c_buffer[32];
		TFBIGMIX8e();
		p[ 0]+=h[7];	p[ 1]+=h[8];	p[ 2]+=h[0];	p[ 3]+=h[1];	p[ 4]+=h[2];	p[ 5]+=c_buffer[32];	p[ 7]+=c_buffer[33];	p[ 6]+=c_buffer[34];
		TFBIGMIX8o();
		p[ 0]+=h[8];	p[ 1]+=h[0];	p[ 2]+=h[1];	p[ 3]+=h[2];	p[ 4]+=h[3];	p[ 5]+=c_buffer[34];	p[ 7]+=c_buffer[35];	p[ 6]+=c_buffer[36];
		TFBIGMIX8e();
		p[ 0]+=h[0];	p[ 1]+=h[1];	p[ 2]+=h[2];	p[ 3]+=h[3];	p[ 4]+=h[4];	p[ 5]+=c_buffer[36];	p[ 7]+=c_buffer[37];	p[ 6]+=c_buffer[38];
		TFBIGMIX8o();
		p[ 0]+=h[1];	p[ 1]+=h[2];	p[ 2]+=h[3];	p[ 3]+=h[4];	p[ 4]+=h[5];	p[ 5]+=c_buffer[38];	p[ 7]+=c_buffer[39];	p[ 6]+=c_buffer[40];
		TFBIGMIX8e();
		p[ 0]+=h[2];	p[ 1]+=h[3];	p[ 2]+=h[4];	p[ 3]+=h[5];	p[ 4]+=h[6];	p[ 5]+=c_buffer[40];	p[ 7]+=c_buffer[41];	p[ 6]+=c_buffer[42];
		TFBIGMIX8o();
		p[ 0]+=h[3];	p[ 1]+=h[4];	p[ 2]+=h[5];	p[ 3]+=h[6];	p[ 4]+=h[7];	p[ 5]+=c_buffer[42];	p[ 7]+=c_buffer[43];	p[ 6]+=c_buffer[44];
		TFBIGMIX8e();
		p[ 0]+=h[4];	p[ 1]+=h[5];	p[ 2]+=h[6];	p[ 3]+=h[7];	p[ 4]+=h[8];	p[ 5]+=c_buffer[44];	p[ 7]+=c_buffer[45];	p[ 6]+=c_buffer[46];
		TFBIGMIX8o();
		p[ 0]+=h[5];	p[ 1]+=h[6];	p[ 2]+=h[7];	p[ 3]+=h[8];	p[ 4]+=h[0];	p[ 5]+=c_buffer[46];	p[ 7]+=c_buffer[47];	p[ 6]+=c_buffer[48];
		TFBIGMIX8e();
		p[ 0]+=h[6];	p[ 1]+=h[7];	p[ 2]+=h[8];	p[ 3]+=h[0];	p[ 4]+=h[1];	p[ 5]+=c_buffer[48];	p[ 7]+=c_buffer[49];	p[ 6]+=c_buffer[50];
		TFBIGMIX8o();
		p[ 0]+=h[7];	p[ 1]+=h[8];	p[ 2]+=h[0];	p[ 3]+=h[1];	p[ 4]+=h[2];	p[ 5]+=c_buffer[50];	p[ 7]+=c_buffer[51];	p[ 6]+=c_buffer[52];
		TFBIGMIX8e();
		p[ 0]+=h[8];	p[ 1]+=h[0];	p[ 2]+=h[1];	p[ 3]+=h[2];	p[ 4]+=h[3];	p[ 5]+=c_buffer[52];	p[ 7]+=c_buffer[53];	p[ 6]+=c_buffer[54];
		TFBIGMIX8o();
		p[ 0]+=h[0];	p[ 1]+=h[1];	p[ 2]+=h[2];	p[ 3]+=h[3];	p[ 4]+=h[4];	p[ 5]+=c_buffer[54];	p[ 7]+=c_buffer[55];	p[ 6]+=c_buffer[56];
		
		p[0]^= c_buffer[57];
		p[1]^= nonce2;

		t0 = vectorize(8); // extra
		t1 = vectorize(0xFF00000000000000ull); // etype
		t2 = t0^t1;

		h[0] = p[ 0];
		h[1] = p[ 1];
		h[2] = p[ 2];
		h[3] = p[ 3];
		h[4] = p[ 4];
		h[5] = p[ 5];
		h[6] = p[ 6];
		h[7] = p[ 7];

		h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ vectorize(0x1BD11BDAA9FC1A22);
		p[ 0] = p[ 1] = p[ 2] = p[ 3] = p[ 4] =p[ 5] =p[ 6] = p[ 7] = vectorize(0);

		#define h0 h[0]
		#define h1 h[1]
		#define h2 h[2]
		#define h3 h[3]
		#define h4 h[4]
		#define h5 h[5]
		#define h6 h[6]
		#define h7 h[7]
		#define h8 h[8]

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint64_t *outpHash = &output64[thread<<3];
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			outpHash[i] = devectorize(p[i]);
	}

}

__host__
void skein512_cpu_setBlock_80(void *pdata)
{
	uint64_t message[20];
	memcpy(&message[0], pdata, 80);

	uint64_t p[8];
	uint64_t h[9];
	uint64_t t0, t1, t2;

	h[0] = 0x4903ADFF749C51CEull;
	h[1] = 0x0D95DE399746DF03ull;
	h[2] = 0x8FD1934127C79BCEull;
	h[3] = 0x9A255629FF352CB1ull;
	h[4] = 0x5DB62599DF6CA7B0ull;
	h[5] = 0xEABE394CA9D5C3F4ull;
	h[6] = 0x991112C71A75B523ull;
	h[7] = 0xAE18A40B660FCC33ull;
	// h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ SPH_C64(0x1BD11BDAA9FC1A22);
	h[8] = 0xcab2076d98173ec4ULL;

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

	message[10] = message[0] ^ p[0];
	message[11] = message[1] ^ p[1];
	message[12] = message[2] ^ p[2];
	message[13] = message[3] ^ p[3];
	message[14] = message[4] ^ p[4];
	message[15] = message[5] ^ p[5];
	message[16] = message[6] ^ p[6];
	message[17] = message[7] ^ p[7];

	message[18] = t2;
	
	uint64_t buffer[128];
	
//	buffer[ 0] = message[ 8];
	buffer[ 0] = message[ 9];
	h[0] = buffer[ 1] = message[10];
	h[1] = buffer[ 2] = message[11];
	h[2] = buffer[ 3] = message[12];
	h[3] = buffer[ 4] = message[13];
	h[4] = buffer[ 5] = message[14];
	h[5] = buffer[ 6] = message[15];
	h[6] = buffer[ 7] = message[16];
	h[7] = buffer[ 8] = message[17];
	h[8] = buffer[ 9] = h[0]^h[1]^h[2]^h[3]^h[4]^h[5]^h[6]^h[7]^0x1BD11BDAA9FC1A22ULL;
	
	t0 = 0x50ull;
	t1 = 0xB000000000000000ull;
	t2 = t0^t1;
	
	p[ 0] = message[ 8] + h[0];	p[ 2] = h[2];		p[ 3] = h[3];	p[ 4] = h[4];
	p[ 5] = h[5] + t0;		p[ 6] = h[6] + t1;	p[ 7] = h[7];

	p[2] += p[3];	p[4] += p[5];	p[6] += p[7];

	p[3] = ROTL64(p[3], 36) ^ p[2];	p[5] = ROTL64(p[5], 19) ^ p[4];	p[7] = ROTL64(p[7], 37) ^ p[6];

	p[4] += p[7];	p[6] += p[5];

	p[7] = ROTL64(p[7], 27) ^ p[4];
	p[5] = ROTL64(p[5], 14) ^ p[6];

	buffer[10] = p[ 0];
	buffer[11] = p[ 2];
	buffer[12] = p[ 3];
	buffer[13] = p[ 4];
	buffer[14] = p[ 5];
	buffer[15] = p[ 6];
	buffer[16] = p[ 7];
	buffer[17] = ROTL64(p[3], 42);
	buffer[18] = ROTL64(p[5], 36);
	buffer[19] = ROTL64(p[7], 39);
	
	buffer[20]=h[6]+t1;
	buffer[21]=h[8]+1;
	buffer[22]=h[7]+t2;
	buffer[23]=h[0]+2;
	buffer[24]=h[8]+t0;
	buffer[25]=h[1]+3;
	buffer[26]=h[0]+t1;
	buffer[27]=h[2]+4;
	buffer[28]=h[1]+t2;
	buffer[29]=h[3]+5;
	buffer[30]=h[2]+t0;
	buffer[31] = h[4]+6;
	buffer[32] = h[3]+t1;
	buffer[33] = h[5]+7;
	buffer[34] = h[4]+t2;
	buffer[35] = h[6]+8;
	buffer[36] = h[5]+t0;
	buffer[37] = h[7]+9;
	buffer[38] = h[6]+t1;
	buffer[39] = h[8]+10;
	buffer[40] = h[7]+t2;
	buffer[41] = h[0]+11;
	buffer[42] = h[8]+t0;
	buffer[43] = h[1]+12;
	buffer[44] = h[0]+t1;
	buffer[45] = h[2]+13;
	buffer[46] = h[1]+t2;
	buffer[47] = h[3]+14;
	buffer[48] = h[2]+t0;
	buffer[49] = h[4]+15;
	buffer[50] = h[3]+t1;
	buffer[51] = h[5]+16;
	buffer[52] = h[4]+t2;
	buffer[53] = h[6]+17;
	buffer[54] = h[5]+t0;
	buffer[55] = h[7]+18;
	buffer[56] = h[6]+t1;
		
	buffer[57] = message[ 8];

	cudaMemcpyToSymbol(c_buffer, buffer, sizeof(buffer), 0, cudaMemcpyHostToDevice);

	CUDA_SAFE_CALL(cudaGetLastError());
}

__host__
void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *g_hash)
{
	uint32_t tpb = TPB52;
	int dev_id = device_map[thr_id];
	
	if (device_sm[dev_id] <= 500) tpb = TPB50;
	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);

	uint64_t *d_hash = (uint64_t*) g_hash;
	
	// hash function is cut in 2 parts to reduce kernel size
	skein512_gpu_hash_80 <<< grid, block >>> (threads, startNounce, d_hash);
}
