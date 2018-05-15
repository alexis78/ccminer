/*
	Based upon Tanguy Pruvot's repo
	Provos Alexis - 2016
*/

#include <memory.h>
#include "cuda_helper.h"
#include "miner.h"

#define TPB 512

#define TFBIGMIX8e(){\
		p0+=p1;p2+=p3;p4+=p5;p6+=p7;p1=ROTL64(p1,46) ^ p0;p3=ROTL64(p3,36) ^ p2;p5=ROTL64(p5,19) ^ p4;p7=ROTL64(p7,37) ^ p6;\
		p2+=p1;p4+=p7;p6+=p5;p0+=p3;p1=ROTL64(p1,33) ^ p2;p7=ROTL64(p7,27) ^ p4;p5=ROTL64(p5,14) ^ p6;p3=ROTL64(p3,42) ^ p0;\
		p4+=p1;p6+=p3;p0+=p5;p2+=p7;p1=ROTL64(p1,17) ^ p4;p3=ROTL64(p3,49) ^ p6;p5=ROTL64(p5,36) ^ p0;p7=ROTL64(p7,39) ^ p2;\
		p6+=p1;p0+=p7;p2+=p5;p4+=p3;p1=ROTL64(p1,44) ^ p6;p7=ROTL64(p7, 9) ^ p0;p5=ROTL64(p5,54) ^ p2;p3=ROTR64(p3, 8) ^ p4;\
}
#define TFBIGMIX8o(){\
		p0+=p1;p2+=p3;p4+=p5;p6+=p7;p1=ROTL64(p1,39) ^ p0;p3=ROTL64(p3,30) ^ p2;p5=ROTL64(p5,34) ^ p4;p7=ROTL64(p7,24) ^ p6;\
		p2+=p1;p4+=p7;p6+=p5;p0+=p3;p1=ROTL64(p1,13) ^ p2;p7=ROTL64(p7,50) ^ p4;p5=ROTL64(p5,10) ^ p6;p3=ROTL64(p3,17) ^ p0;\
		p4+=p1;p6+=p3;p0+=p5;p2+=p7;p1=ROTL64(p1,25) ^ p4;p3=ROTL64(p3,29) ^ p6;p5=ROTL64(p5,39) ^ p0;p7=ROTL64(p7,43) ^ p2;\
		p6+=p1;p0+=p7;p2+=p5;p4+=p3;p1=ROTL64(p1, 8) ^ p6;p7=ROTL64(p7,35) ^ p0;p5=ROTR64(p5, 8) ^ p2;p3=ROTL64(p3,22) ^ p4;\
}

__constant__ uint64_t c_sk_buf[64];

__constant__ uint64_t c_t2[ 3] = { 0x08, 0xff00000000000000, 0xff00000000000008};
__constant__ uint32_t c_add[18] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};

#define skein_ks_parity64 0x1BD11BDAA9FC1A22ull
//#include <stdio.h>
__global__  __launch_bounds__(512, 2)
void skein256_gpu_hash_32(uint32_t threads, uint64_t *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint64_t dt0 = __ldg(&outputHash[thread]);
		const uint64_t dt1 = __ldg(&outputHash[threads   + thread]);
		const uint64_t dt2 = __ldg(&outputHash[threads*2 + thread]);
		const uint64_t dt3 = __ldg(&outputHash[threads*3 + thread]);

		uint64_t h[ 9] = {
			0xCCD044A12FDB3E13, 0xE83590301A79A9EB,	0x55AEA0614F816E6F, 0x2A2767A4AE9B94DB,
			0xEC06025E74DD7683, 0xE7A436CDC4746251,	0xC36FBAF9393AD185, 0x3EEDBA1833EDFC13,
			0xb69d3cfcc73a4e2a, // skein_ks_parity64 ^ h[0..7]
		};

		int i=0;
		
		uint64_t p0 = c_sk_buf[i++] + dt0 + dt1;
		uint64_t p1 = c_sk_buf[i++] + dt1;
		uint64_t p2 = c_sk_buf[i++] + dt2 + dt3;
		uint64_t p3 = c_sk_buf[i++] + dt3;
		uint64_t p4 = c_sk_buf[i++];
		uint64_t p5 = c_sk_buf[i++];
		uint64_t p6 = c_sk_buf[i++];
		uint64_t p7 = c_sk_buf[i++];

//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 1);
//		TFBIGMIX8e();
		p1=ROTL64(p1,46) ^ p0;
		p3=ROTL64(p3,36) ^ p2;
		p2+=p1;
		p0+=p3;
		p1=ROTL64(p1,33) ^ p2;
		p3=ROTL64(p3,42) ^ p0;
		p4+=p1;
		p6+=p3;
		p0+=p5;
		p2+=p7;
		p1=ROTL64(p1,17) ^ p4;
		p3=ROTL64(p3,49) ^ p6;
		p5=c_sk_buf[i++] ^ p0;
		p7=c_sk_buf[i++] ^ p2;
		p6+=p1;
		p0+=p7;
		p2+=p5;
		p4+=p3;
		p1=ROTL64(p1,44) ^ p6;
		p7=ROTL64(p7, 9) ^ p0;
		p5=ROTL64(p5,54) ^ p2;
		p3=ROTL64(p3,56) ^ p4;

		p0+=h[ 1];		p1+=h[ 2];
		p2+=h[ 3];		p3+=h[ 4];
		p4+=h[ 5];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

		TFBIGMIX8o();

		p0+=h[ 2];		p1+=h[ 3];
		p2+=h[ 4];		p3+=h[ 5];
		p4+=h[ 6];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 3);
		TFBIGMIX8e();

		p0+=h[ 3];		p1+=h[ 4];
		p2+=h[ 5];		p3+=h[ 6];
		p4+=h[ 7];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

		TFBIGMIX8o();

		p0+=h[ 4];		p1+=h[ 5];
		p2+=h[ 6];		p3+=h[ 7];
		p4+=h[ 8];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 5);
		TFBIGMIX8e();

		p0+=h[ 5];		p1+=h[ 6];
		p2+=h[ 7];		p3+=h[ 8];
		p4+=h[ 0];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];
		
		TFBIGMIX8o();

		p0+=h[ 6];		p1+=h[ 7];
		p2+=h[ 8];		p3+=h[ 0];
		p4+=h[ 1];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 7);
		TFBIGMIX8e();

		p0+=h[ 7];		p1+=h[ 8];
		p2+=h[ 0];		p3+=h[ 1];
		p4+=h[ 2];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

		TFBIGMIX8o();

		p0+=h[ 8];		p1+=h[ 0];
		p2+=h[ 1];		p3+=h[ 2];
		p4+=h[ 3];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 9);
		TFBIGMIX8e();

		p0+=h[ 0];		p1+=h[ 1];
		p2+=h[ 2];		p3+=h[ 3];
		p4+=h[ 4];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

		TFBIGMIX8o();

		p0+=h[ 1];		p1+=h[ 2];
		p2+=h[ 3];		p3+=h[ 4];
		p4+=h[ 5];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7,11);
		TFBIGMIX8e();

		p0+=h[ 2];		p1+=h[ 3];
		p2+=h[ 4];		p3+=h[ 5];
		p4+=h[ 6];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

		TFBIGMIX8o();

		p0+=h[ 3];		p1+=h[ 4];
		p2+=h[ 5];		p3+=h[ 6];
		p4+=h[ 7];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7,13);
		TFBIGMIX8e();

		p0+=h[ 4];		p1+=h[ 5];
		p2+=h[ 6];		p3+=h[ 7];
		p4+=h[ 8];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

		TFBIGMIX8o();

		p0+=h[ 5];		p1+=h[ 6];
		p2+=h[ 7];		p3+=h[ 8];
		p4+=h[ 0];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7,15);
		TFBIGMIX8e();

		p0+=h[ 6];		p1+=h[ 7];
		p2+=h[ 8];		p3+=h[ 0];
		p4+=h[ 1];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

		TFBIGMIX8o();

		p0+=h[ 7];		p1+=h[ 8];
		p2+=h[ 0];		p3+=h[ 1];
		p4+=h[ 2];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7,17);
		TFBIGMIX8e();

		p0+=h[ 8];		p1+=h[ 0];
		p2+=h[ 1];		p3+=h[ 2];
		p4+=h[ 3];		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];	p6+=c_sk_buf[i];

		TFBIGMIX8o();
		p4+=h[ 4];
		p5+=c_sk_buf[i++];
		p7+=c_sk_buf[i++];
		p6+=c_sk_buf[i];
				
		p0 = (p0+h[ 0]) ^ dt0;
		p1 = (p1+h[ 1]) ^ dt1;
		p2 = (p2+h[ 2]) ^ dt2;
		p3 = (p3+h[ 3]) ^ dt3;

		h[0] = p0;
		h[1] = p1;
		h[2] = p2;
		h[3] = p3;
		h[4] = p4;
		h[5] = p5;
		h[6] = p6;
		h[7] = p7;
		h[8] = h[ 0] ^ h[ 1] ^ h[ 2] ^ h[ 3] ^ h[ 4] ^ h[ 5] ^ h[ 6] ^ h[ 7] ^ skein_ks_parity64;

		p5+=c_t2[0];  //p5 already equal h[5]
		p6+=c_t2[1];

//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 1);
		TFBIGMIX8e();

		p0+=h[ 1];		p1+=h[ 2];
		p2+=h[ 3];		p3+=h[ 4];
		p4+=h[ 5];		p5+=h[ 6] + c_t2[ 1];
		p6+=h[ 7] + c_t2[ 2];	p7+=h[ 8] + c_add[ 0];
	
		TFBIGMIX8o();

		p0+=h[ 2];		p1+=h[ 3];
		p2+=h[ 4];		p3+=h[ 5];
		p4+=h[ 6];		p5+=h[ 7] + c_t2[ 2];
		p6+=h[ 8] + c_t2[ 0];	p7+=h[ 0] + c_add[ 1];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 3);
		TFBIGMIX8e();

		p0+=h[ 3];		p1+=h[ 4];
		p2+=h[ 5];		p3+=h[ 6];
		p4+=h[ 7];		p5+=h[ 8] + c_t2[ 0];
		p6+=h[ 0] + c_t2[ 1];	p7+=h[ 1] + c_add[ 2];
	
		TFBIGMIX8o();

		p0+=h[ 4];		p1+=h[ 5];
		p2+=h[ 6];		p3+=h[ 7];
		p4+=h[ 8];		p5+=h[ 0] + c_t2[ 1];
		p6+=h[ 1] + c_t2[ 2];	p7+=h[ 2] + c_add[ 3];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 5);
		TFBIGMIX8e();

		p0+=h[ 5];		p1+=h[ 6];
		p2+=h[ 7];		p3+=h[ 8];
		p4+=h[ 0];		p5+=h[ 1] + c_t2[ 2];
		p6+=h[ 2] + c_t2[ 0];	p7+=h[ 3] + c_add[ 4];
	
		TFBIGMIX8o();

		p0+=h[ 6];		p1+=h[ 7];
		p2+=h[ 8];		p3+=h[ 0];
		p4+=h[ 1];		p5+=h[ 2] + c_t2[ 0];
		p6+=h[ 3] + c_t2[ 1];	p7+=h[ 4] + c_add[ 5];

//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 7);
		TFBIGMIX8e();

		p0+=h[ 7];		p1+=h[ 8];
		p2+=h[ 0];		p3+=h[ 1];
		p4+=h[ 2];		p5+=h[ 3] + c_t2[ 1];
		p6+=h[ 4] + c_t2[ 2];	p7+=h[ 5] + c_add[ 6];
	
		TFBIGMIX8o();

		p0+=h[ 8];		p1+=h[ 0];
		p2+=h[ 1];		p3+=h[ 2];
		p4+=h[ 3];		p5+=h[ 4] + c_t2[ 2];
		p6+=h[ 5] + c_t2[ 0];	p7+=h[ 6] + c_add[ 7];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 9);
		TFBIGMIX8e();

		p0+=h[ 0];		p1+=h[ 1];
		p2+=h[ 2];		p3+=h[ 3];
		p4+=h[ 4];		p5+=h[ 5] + c_t2[ 0];
		p6+=h[ 6] + c_t2[ 1];	p7+=h[ 7] + c_add[ 8];
	
		TFBIGMIX8o();

		p0+=h[ 1];		p1+=h[ 2];
		p2+=h[ 3];		p3+=h[ 4];
		p4+=h[ 5];		p5+=h[ 6] + c_t2[ 1];
		p6+=h[ 7] + c_t2[ 2];	p7+=h[ 8] + c_add[ 9];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7,11);
		TFBIGMIX8e();

		p0+=h[ 2];		p1+=h[ 3];
		p2+=h[ 4];		p3+=h[ 5];
		p4+=h[ 6];		p5+=h[ 7] + c_t2[ 2];
		p6+=h[ 8] + c_t2[ 0];	p7+=h[ 0] + c_add[10];
	
		TFBIGMIX8o();

		p0+=h[ 3];		p1+=h[ 4];
		p2+=h[ 5];		p3+=h[ 6];
		p4+=h[ 7];		p5+=h[ 8] + c_t2[ 0];
		p6+=h[ 0] + c_t2[ 1];	p7+=h[ 1] + c_add[11];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7,13);
		TFBIGMIX8e();

		p0+=h[ 4];		p1+=h[ 5];
		p2+=h[ 6];		p3+=h[ 7];
		p4+=h[ 8];		p5+=h[ 0] + c_t2[ 1];
		p6+=h[ 1] + c_t2[ 2];	p7+=h[ 2] + c_add[12];
	
		TFBIGMIX8o();

		p0+=h[ 5];		p1+=h[ 6];
		p2+=h[ 7];		p3+=h[ 8];
		p4+=h[ 0];		p5+=h[ 1] + c_t2[ 2];
		p6+=h[ 2] + c_t2[ 0];	p7+=h[ 3] + c_add[13];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7,15);
		TFBIGMIX8e();

		p0+=h[ 6];		p1+=h[ 7];
		p2+=h[ 8];		p3+=h[ 0];
		p4+=h[ 1];		p5+=h[ 2] + c_t2[ 0];
		p6+=h[ 3] + c_t2[ 1];	p7+=h[ 4] + c_add[14];
	
		TFBIGMIX8o();

		p0+=h[ 7];		p1+=h[ 8];
		p2+=h[ 0];		p3+=h[ 1];
		p4+=h[ 2];		p5+=h[ 3] + c_t2[ 1];
		p6+=h[ 4] + c_t2[ 2];	p7+=h[ 5] + c_add[15];
		
//		Round_8_512v30(h, t, p0, p1, p2, p3, p4, p5, p6, p7,17);	
		TFBIGMIX8e();

		p0+=h[ 8];		p1+=h[ 0];
		p2+=h[ 1];		p3+=h[ 2];
		p4+=h[ 3];		p5+=h[ 4] + c_t2[ 2];
		p6+=h[ 5] + c_t2[ 0];	p7+=h[ 6] + c_add[16];
	
		TFBIGMIX8o();

		p0+=h[ 0];		p1+=h[ 1];
		p2+=h[ 2];		p3+=h[ 3];
		p4+=h[ 4];		p5+=h[ 5] + c_t2[ 0];
		p6+=h[ 6] + c_t2[ 1];	p7+=h[ 7] + c_add[17];
		
		outputHash[thread] = p0;
		outputHash[threads   + thread] = p1;
		outputHash[threads*2 + thread] = p2;
		outputHash[threads*3 + thread] = p3;
	} //thread
}
__host__
void skein256_cpu_init(int thr_id){

	uint64_t h[ 9] = {
			0xCCD044A12FDB3E13, 0xE83590301A79A9EB,	0x55AEA0614F816E6F, 0x2A2767A4AE9B94DB,
			0xEC06025E74DD7683, 0xE7A436CDC4746251,	0xC36FBAF9393AD185, 0x3EEDBA1833EDFC13,
			0xb69d3cfcc73a4e2a, // skein_ks_parity64 ^ h[0..7]
		};
	
	uint64_t t[3] = {0x20, 0xf000000000000000, 0xf000000000000020};
	uint64_t dt0,dt1,dt2,dt3;
	
	dt0=dt1=dt2=dt3=0;
		
	uint64_t sk_buf[64];
	int i=0;
	
	uint64_t p0 = h[0] + dt0;
	uint64_t p1 = h[1] + dt1;
	uint64_t p2 = h[2] + dt2;
	uint64_t p3 = h[3] + dt3;
	uint64_t p4 = h[4];
	uint64_t p5 = h[5] + t[0];
	uint64_t p6 = h[6] + t[1];
	uint64_t p7 = h[7];
	
	p0+=p1;
	p2+=p3;
	p4+=p5;
	p6+=p7;
	
	p5=ROTL64(p5,19) ^ p4;
	p7=ROTL64(p7,37) ^ p6;

	p4+=p7;
	p6+=p5;	

	p7=ROTL64(p7,27) ^ p4;
	p5=ROTL64(p5,14) ^ p6;
	
	sk_buf[i++] = p0;
	sk_buf[i++] = p1;
	sk_buf[i++] = p2;
	sk_buf[i++] = p3;
	sk_buf[i++] = p4;
	sk_buf[i++] = p5;
	sk_buf[i++] = p6;
	sk_buf[i++] = p7;
	sk_buf[i++] = ROTL64(p5,36);
	sk_buf[i++] = ROTL64(p7,39);
	sk_buf[i++] = h[ 6] + t[1];
	sk_buf[i++] = h[ 8] + 1;
	sk_buf[i++] = h[ 7] + t[2];
	sk_buf[i++] = h[ 0] + 2;
	sk_buf[i++] = h[ 8] + t[ 0];
	sk_buf[i++] = h[ 1] + 3;
	sk_buf[i++] = h[ 0] + t[ 1];
	sk_buf[i++] = h[ 2] + 4;
	sk_buf[i++] = h[ 1] + t[ 2];
	sk_buf[i++] = h[ 3] + 5;
	sk_buf[i++] = h[ 2] + t[ 0];
	sk_buf[i++] = h[ 4] + 6;
	sk_buf[i++] = h[ 3] + t[ 1];
	sk_buf[i++] = h[ 5] + 7;
	sk_buf[i++] = h[ 4] + t[ 2];
	sk_buf[i++] = h[ 6] + 8;
	sk_buf[i++] = h[ 5] + t[ 0];
	sk_buf[i++] = h[ 7] + 9;
	sk_buf[i++] = h[ 6] + t[ 1];
	sk_buf[i++] = h[ 8] + 10;
	sk_buf[i++] = h[ 7] + t[ 2];
	sk_buf[i++] = h[ 0] + 11;
	sk_buf[i++] = h[ 8] + t[ 0];
	sk_buf[i++] = h[ 1] + 12;
	sk_buf[i++] = h[ 0] + t[ 1];
	sk_buf[i++] = h[ 2] + 13;
	sk_buf[i++] = h[ 1] + t[ 2];
	sk_buf[i++] = h[ 3] + 14;
	sk_buf[i++] = h[ 2] + t[ 0];
	sk_buf[i++] = h[ 4] + 15;
	sk_buf[i++] = h[ 3] + t[ 1];
	sk_buf[i++] = h[ 5] + 16;
	sk_buf[i++] = h[ 4] + t[ 2];
	sk_buf[i++] = h[ 6] + 17;
	sk_buf[i++] = h[ 5] + t[ 0];	
	sk_buf[i++] = h[ 7] + 18;
	sk_buf[i++] = h[ 6] + t[ 1];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sk_buf, sk_buf, sizeof(sk_buf), 0, cudaMemcpyHostToDevice));
}

__host__
void skein256_cpu_hash_32(const uint32_t threads, uint2 *d_hash)
{
	dim3 grid((threads + TPB - 1) / TPB);
	dim3 block(TPB);
	skein256_gpu_hash_32<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

