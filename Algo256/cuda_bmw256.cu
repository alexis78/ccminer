#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "cuda_vectors.h"

static uint32_t *d_gnounce[MAX_GPUS];
static uint32_t *d_GNonce[MAX_GPUS];

#define shl(x, n) (x << n)
#define shr(x, n) (x >> n)

#define ss0(x) (shr(x, 1)^ shl(x, 3) ^ ROTL32(x,  4) ^ ROTL32(x, 19))
#define ss1(x) (shr(x, 1)^ shl(x, 2) ^ ROL8(x)       ^ ROTL32(x, 23))
#define ss2(x) (shr(x, 2)^ shl(x, 1) ^ ROTL32(x, 12) ^ ROTL32(x, 25))
#define ss3(x) (shr(x, 2)^ shl(x, 2) ^ ROTL32(x, 15) ^ ROTL32(x, 29))
#define ss4(x) (shr(x, 1) ^ x)
#define ss5(x) (shr(x, 2) ^ x)

#define rs1(x) ROTL32(x,  3)
#define rs2(x) ROTL32(x,  7)
#define rs3(x) ROTL32(x, 13)
#define rs4(x) ROL16(x)
#define rs5(x) ROTL32(x, 19)
#define rs6(x) ROTL32(x, 23)
#define rs7(x) ROTL32(x, 27)

#define TPB 1024
#define NBN 2

__global__ __launch_bounds__(TPB,1)
void bmw256_gpu_hash_32(uint32_t threads, uint2 *g_hash, uint32_t *const __restrict__ nonceVector, const uint2 target){

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){
		uint32_t M32[16] = { 0 };

		*(uint2*)&M32[ 0] = __ldg(&g_hash[thread]);
		*(uint2*)&M32[ 2] = __ldg(&g_hash[thread + 1 * threads]);
		*(uint2*)&M32[ 4] = __ldg(&g_hash[thread + 2 * threads]);
		*(uint2*)&M32[ 6] = __ldg(&g_hash[thread + 3 * threads]);

		M32[ 8]=0x80;
		M32[14]=0x100;
		
//		Compression256(message);
		uint32_t Q[32], XL32, XH32;

		const uint32_t H[16] = {
			0x40414243, 0x44454647, 0x48494A4B, 0x4C4D4E4F,	0x50515253, 0x54555657, 0x58595A5B, 0x5C5D5E5F,
			0x60616263, 0x64656667, 0x68696A6B, 0x6C6D6E6F,	0x70717273, 0x74757677, 0x78797A7B, 0x7C7D7E7F
		};
		uint32_t tmp[16];
		
		*(uint16*)&tmp[ 0] = *(uint16*)&M32[ 0] ^ *(uint16*)&H[ 0];
		
		Q[ 0] = tmp[ 5] - tmp[ 7] + tmp[10] + tmp[13] + tmp[14];	Q[ 1] = tmp[ 6] - tmp[ 8] + tmp[11] + tmp[14] - tmp[15];
		Q[ 2] = tmp[ 0] + tmp[ 7] + tmp[ 9] - tmp[12] + tmp[15];	Q[ 3] = tmp[ 0] - tmp[ 1] + tmp[ 8] - tmp[10] + tmp[13];
		Q[ 4] = tmp[ 1] + tmp[ 2] + tmp[ 9] - tmp[11] - tmp[14];	Q[ 5] = tmp[ 3] - tmp[ 2] + tmp[10] - tmp[12] + tmp[15];
		Q[ 6] = tmp[ 4] - tmp[ 0] - tmp[ 3] - tmp[11] + tmp[13];	Q[ 7] = tmp[ 1] - tmp[ 4] - tmp[ 5] - tmp[12] - tmp[14];
		Q[ 8] = tmp[ 2] - tmp[ 5] - tmp[ 6] + tmp[13] - tmp[15];	Q[ 9] = tmp[ 0] - tmp[ 3] + tmp[ 6] - tmp[ 7] + tmp[14];
		Q[10] = tmp[ 8] - tmp[ 1] - tmp[ 4] - tmp[ 7] + tmp[15];	Q[11] = tmp[ 8] - tmp[ 0] - tmp[ 2] - tmp[ 5] + tmp[ 9];
		Q[12] = tmp[ 1] + tmp[ 3] - tmp[ 6] - tmp[ 9] + tmp[10];	Q[13] = tmp[ 2] + tmp[ 4] + tmp[ 7] + tmp[10] + tmp[11];
		Q[14] = tmp[ 3] - tmp[ 5] + tmp[ 8] - tmp[11] - tmp[12];	Q[15] = tmp[12] - tmp[ 4] - tmp[ 6] - tmp[ 9] + tmp[13];

		/*  Diffuse the differences in every word in a bijective manner with ssi, and then add the values of the previous double pipe. */
		Q[ 0] = ss0(Q[ 0]) + H[ 1];	Q[ 1] = ss1(Q[ 1]) + H[ 2];	Q[ 2] = ss2(Q[ 2]) + H[ 3];	Q[ 3] = ss3(Q[ 3]) + H[ 4];
		Q[ 4] = ss4(Q[ 4]) + H[ 5];	Q[ 5] = ss0(Q[ 5]) + H[ 6];	Q[ 6] = ss1(Q[ 6]) + H[ 7];	Q[ 7] = ss2(Q[ 7]) + H[ 8];
		Q[ 8] = ss3(Q[ 8]) + H[ 9];	Q[ 9] = ss4(Q[ 9]) + H[10];	Q[10] = ss0(Q[10]) + H[11];	Q[11] = ss1(Q[11]) + H[12];
		Q[12] = ss2(Q[12]) + H[13];	Q[13] = ss3(Q[13]) + H[14];	Q[14] = ss4(Q[14]) + H[15];	Q[15] = ss0(Q[15]) + H[ 0];

		/* This is the Message expansion or f_1 in the documentation. It has 16 rounds. Blue Midnight Wish has two tunable security parameters. */
		/* The parameters are named EXPAND_1_ROUNDS and EXPAND_2_ROUNDS. The following relation for these parameters should is satisfied:	*/
		/* EXPAND_1_ROUNDS + EXPAND_2_ROUNDS = 16                           									*/
		
		tmp[ 0] = ROTL32(M32[ 0], 1);	tmp[ 1] = ROTL32(M32[ 1], 2);	tmp[ 2] = ROTL32(M32[ 2], 3);	tmp[ 3] = ROTL32(M32[ 3], 4);
		tmp[ 4] = ROTL32(M32[ 4], 5);	tmp[ 5] = ROTL32(M32[ 5], 6);	tmp[ 6] = ROTL32(M32[ 6], 7);	tmp[ 7] = ROL8(M32[ 7]);
		tmp[ 8] = ROTL32(M32[ 8], 9);	
										tmp[14] = ROTL32(M32[14],15);
		
		uint32_t tmp2[ 2];
		
		Q[16] = ss1(Q[ 0]) + ss2(Q[ 1]) + ss3(Q[ 2]) + ss0(Q[ 3]) + ss1(Q[ 4]) + ss2(Q[ 5]) + ss3(Q[ 6]) + ss0(Q[ 7])
		      + ss1(Q[ 8]) + ss2(Q[ 9]) + ss3(Q[10]) + ss0(Q[11]) + ss1(Q[12]) + ss2(Q[13]) + ss3(Q[14]) + ss0(Q[15]) + ((shl(0x05555555,4) + tmp[ 0] + tmp[ 3]) ^ H[ 7]);
		Q[17] = ss1(Q[ 1]) + ss2(Q[ 2]) + ss3(Q[ 3]) + ss0(Q[ 4]) + ss1(Q[ 5]) + ss2(Q[ 6]) + ss3(Q[ 7]) + ss0(Q[ 8])
		      + ss1(Q[ 9]) + ss2(Q[10]) + ss3(Q[11]) + ss0(Q[12]) + ss1(Q[13]) + ss2(Q[14]) + ss3(Q[15]) + ss0(Q[16]) + ((17U*(0x05555555) + tmp[ 1] + tmp[ 4]) ^ H[ 8]);

		tmp2[ 0] = Q[ 2] + Q[ 4] + Q[ 6] + Q[ 8] + Q[10] + Q[12] + Q[14];
		tmp2[ 1] = Q[ 3] + Q[ 5] + Q[ 7] + Q[ 9] + Q[11] + Q[13] + Q[15];
		
		Q[18] = rs1(Q[ 3]) + rs2(Q[ 5]) + rs3(Q[ 7]) + rs4(Q[ 9]) + rs5(Q[11]) + rs6(Q[13]) + rs7(Q[15]) + ss4(Q[16]) + ss5(Q[17]) + tmp2[ 0] +((18U*(0x05555555) + tmp[ 2] + tmp[ 5]) ^ H[ 9]);
		Q[19] = rs1(Q[ 4]) + rs2(Q[ 6]) + rs3(Q[ 8]) + rs4(Q[10]) + rs5(Q[12]) + rs6(Q[14]) + rs7(Q[16]) + ss4(Q[17]) + ss5(Q[18]) + tmp2[ 1] +((19U*(0x05555555) + tmp[ 3] + tmp[ 6]) ^ H[10]);

		tmp2[ 0]+= Q[16] - Q[ 2];
		tmp2[ 1]+= Q[17] - Q[ 3];
		
		Q[20] = rs1(Q[ 5])+rs2(Q[ 7])+rs3(Q[ 9])+rs4(Q[11])+rs5(Q[13])+rs6(Q[15])+rs7(Q[17])+ss4(Q[18])+ss5(Q[19])+tmp2[ 0]+((20U*(0x05555555) + tmp[ 4] + tmp[ 7] - tmp[14]) ^ H[11]);
		Q[21] = rs1(Q[ 6])+rs2(Q[ 8])+rs3(Q[10])+rs4(Q[12])+rs5(Q[14])+rs6(Q[16])+rs7(Q[18])+ss4(Q[19])+ss5(Q[20])+tmp2[ 1]+((21U*(0x05555555) + tmp[ 5] + tmp[ 8]) ^ H[12]);

		tmp2[ 0]+= Q[18] - Q[ 4];
		tmp2[ 1]+= Q[19] - Q[ 5];
		
		Q[22] = rs1(Q[ 7])+rs2(Q[ 9])+rs3(Q[11])+rs4(Q[13])+rs5(Q[15])+rs6(Q[17])+rs7(Q[19])+ss4(Q[20])+ss5(Q[21])+tmp2[ 0]+((22U*(0x05555555) + tmp[ 6] - tmp[ 0]) ^ H[13]);
		Q[23] = rs1(Q[ 8])+rs2(Q[10])+rs3(Q[12])+rs4(Q[14])+rs5(Q[16])+rs6(Q[18])+rs7(Q[20])+ss4(Q[21])+ss5(Q[22])+tmp2[ 1]+((23U*(0x05555555) + tmp[ 7] - tmp[ 1]) ^ H[14]);

		tmp2[ 0]+= Q[20] - Q[ 6];
		tmp2[ 1]+= Q[21] - Q[ 7];

		Q[24] = rs1(Q[ 9])+rs2(Q[11])+rs3(Q[13])+rs4(Q[15])+rs5(Q[17])+rs6(Q[19])+rs7(Q[21])+ss4(Q[22])+ss5(Q[23])+tmp2[ 0]+((24U*(0x05555555) + tmp[ 8] - tmp[ 2]) ^ H[15]);
		Q[25] = rs1(Q[10])+rs2(Q[12])+rs3(Q[14])+rs4(Q[16])+rs5(Q[18])+rs6(Q[20])+rs7(Q[22])+ss4(Q[23])+ss5(Q[24])+tmp2[ 1]+((25U*(0x05555555) - tmp[ 3]) ^ H[ 0]);
		
		tmp2[ 0]+= Q[22] - Q[ 8];
		tmp2[ 1]+= Q[23] - Q[ 9];
		
		Q[26] = rs1(Q[11])+rs2(Q[13])+rs3(Q[15])+rs4(Q[17])+rs5(Q[19])+rs6(Q[21])+rs7(Q[23])+ss4(Q[24])+ss5(Q[25])+tmp2[ 0]+((26U*(0x05555555) - tmp[ 4]) ^ H[ 1]);
		Q[27] = rs1(Q[12])+rs2(Q[14])+rs3(Q[16])+rs4(Q[18])+rs5(Q[20])+rs6(Q[22])+rs7(Q[24])+ss4(Q[25])+ss5(Q[26])+tmp2[ 1]+((27U*(0x05555555) + tmp[14] - tmp[ 5]) ^ H[ 2]);

		tmp2[ 0]+= Q[24] - Q[10];
		tmp2[ 1]+= Q[25] - Q[11];

		Q[28] = rs1(Q[13])+rs2(Q[15])+rs3(Q[17])+rs4(Q[19])+rs5(Q[21])+rs6(Q[23])+rs7(Q[25])+ss4(Q[26])+ss5(Q[27])+tmp2[ 0]+((28U*(0x05555555) - tmp[ 6]) ^ H[ 3]);
		Q[29] = rs1(Q[14])+rs2(Q[16])+rs3(Q[18])+rs4(Q[20])+rs5(Q[22])+rs6(Q[24])+rs7(Q[26])+ss4(Q[27])+ss5(Q[28])+tmp2[ 1]+((29U*(0x05555555) + tmp[ 0] - tmp[ 7]) ^ H[ 4]);

		tmp2[ 0]+= Q[26] - Q[12];
		tmp2[ 1]+= Q[27] - Q[13];

		Q[30] = rs1(Q[15])+rs2(Q[17])+rs3(Q[19])+rs4(Q[21])+rs5(Q[23])+rs6(Q[25])+rs7(Q[27])+ss4(Q[28])+ss5(Q[29])+tmp2[ 0]+((30U*(0x05555555) + tmp[14] + tmp[ 1] - tmp[ 8]) ^ H[ 5]);
		Q[31] = rs1(Q[16])+rs2(Q[18])+rs3(Q[20])+rs4(Q[22])+rs5(Q[24])+rs6(Q[26])+rs7(Q[28])+ss4(Q[29])+ss5(Q[30])+tmp2[ 1]+((31U*(0x05555555) + tmp[ 2]) ^ H[ 6]);
		
		/* Blue Midnight Wish has two temporary cummulative variables that accumulate via XORing 16 new variables that are produced in the Message Expansion part. */
		XL32 =  Q[16] ^ 	  xor3x(Q[17], Q[18], xor3x(Q[19], Q[20], xor3x(Q[21], Q[22], Q[23])));
		XH32 = xor3x(XL32, Q[24], xor3x(Q[25], Q[26], xor3x(Q[27], Q[28], xor3x(Q[29], Q[30], Q[31]))));

		/*  This part is the function f_2 - in the documentation            */
		/*  Compute the double chaining pipe for the next message block.    */
		M32[0] = xor3x(shl(XH32, 5), shr(Q[16], 5), M32[ 0]) + xor3x(XL32, Q[24], Q[ 0]);
		M32[1] = xor3x(shr(XH32, 7), shl(Q[17], 8), M32[ 1]) + xor3x(XL32, Q[25], Q[ 1]);
		M32[2] = xor3x(shr(XH32, 5), shl(Q[18], 5), M32[ 2]) + xor3x(XL32, Q[26], Q[ 2]);
		M32[3] = xor3x(shr(XH32, 1), shl(Q[19], 5), M32[ 3]) + xor3x(XL32, Q[27], Q[ 3]);
		M32[4] = xor3x(shr(XH32, 3), Q[20] 	  , M32[ 4]) + xor3x(XL32, Q[28], Q[ 4]);
		M32[5] = xor3x(shl(XH32, 6), shr(Q[21], 6), M32[ 5]) + xor3x(XL32, Q[29], Q[ 5]);
		M32[6] = xor3x(shr(XH32, 4), shl(Q[22], 6), M32[ 6]) + xor3x(XL32, Q[30], Q[ 6]);
		M32[7] = xor3x(shr(XH32,11), shl(Q[23], 2), M32[ 7]) + xor3x(XL32, Q[31], Q[ 7]);

		M32[ 8] = ROTL32(M32[ 4], 9) + xor3x(XH32, Q[24], M32[ 8]) + xor3x(shl(XL32, 8), Q[23], Q[ 8]);
		M32[ 9] = ROTL32(M32[ 5],10) + xor3x(XH32, Q[25], M32[ 9]) + xor3x(shr(XL32, 6), Q[16], Q[ 9]);
		M32[10] = ROTL32(M32[ 6],11) + xor3x(XH32, Q[26], M32[10]) + xor3x(shl(XL32, 6), Q[17], Q[10]);
		M32[11] = ROTL32(M32[ 7],12) + xor3x(XH32, Q[27], M32[11]) + xor3x(shl(XL32, 4), Q[18], Q[11]);
		M32[12] = ROTL32(M32[ 0],13) + xor3x(XH32, Q[28], M32[12]) + xor3x(shr(XL32, 3), Q[19], Q[12]);
		M32[13] = ROTL32(M32[ 1],14) + xor3x(XH32, Q[29], M32[13]) + xor3x(shr(XL32, 4), Q[20], Q[13]);
		M32[14] = ROTL32(M32[ 2],15) + xor3x(XH32, Q[30], M32[14]) + xor3x(shr(XL32, 7), Q[21], Q[14]);
		M32[15] = ROL16(M32[ 3])     + xor3x(XH32, Q[31], M32[15]) + xor3x(shr(XL32, 2), Q[22], Q[15]);

//		Compression256_2(M32);
		const uint32_t H2[16] = {
			0xaaaaaaa0, 0xaaaaaaa1, 0xaaaaaaa2, 0xaaaaaaa3,	0xaaaaaaa4, 0xaaaaaaa5, 0xaaaaaaa6, 0xaaaaaaa7,
			0xaaaaaaa8, 0xaaaaaaa9, 0xaaaaaaaa, 0xaaaaaaab,	0xaaaaaaac, 0xaaaaaaad, 0xaaaaaaae, 0xaaaaaaaf
		};

		*(uint16*)&tmp[ 0] = *(uint16*)&M32[ 0] ^ *(uint16*)&H2[ 0];
		
		Q[ 0] = tmp[ 5] - tmp[ 7] + tmp[10] + tmp[13] + tmp[14];	Q[ 1] = tmp[ 6] - tmp[ 8] + tmp[11] + tmp[14] - tmp[15];
		Q[ 2] = tmp[ 0] + tmp[ 7] + tmp[ 9] - tmp[12] + tmp[15];	Q[ 3] = tmp[ 0] - tmp[ 1] + tmp[ 8] - tmp[10] + tmp[13];
		Q[ 4] = tmp[ 1] + tmp[ 2] + tmp[ 9] - tmp[11] - tmp[14];	Q[ 5] = tmp[ 3] - tmp[ 2] + tmp[10] - tmp[12] + tmp[15];
		Q[ 6] = tmp[ 4] - tmp[ 0] - tmp[ 3] - tmp[11] + tmp[13];	Q[ 7] = tmp[ 1] - tmp[ 4] - tmp[ 5] - tmp[12] - tmp[14];
		Q[ 8] = tmp[ 2] - tmp[ 5] - tmp[ 6] + tmp[13] - tmp[15];	Q[ 9] = tmp[ 0] - tmp[ 3] + tmp[ 6] - tmp[ 7] + tmp[14];
		Q[10] = tmp[ 8] - tmp[ 1] - tmp[ 4] - tmp[ 7] + tmp[15];	Q[11] = tmp[ 8] - tmp[ 0] - tmp[ 2] - tmp[ 5] + tmp[ 9];
		Q[12] = tmp[ 1] + tmp[ 3] - tmp[ 6] - tmp[ 9] + tmp[10];	Q[13] = tmp[ 2] + tmp[ 4] + tmp[ 7] + tmp[10] + tmp[11];
		Q[14] = tmp[ 3] - tmp[ 5] + tmp[ 8] - tmp[11] - tmp[12];	Q[15] = tmp[12] - tmp[ 4] - tmp[ 6] - tmp[ 9] + tmp[13];

		/*  Diffuse the differences in every word in a bijective manner with ssi, and then add the values of the previous double pipe. */
		Q[ 0] = ss0(Q[ 0]) + H2[ 1];	Q[ 1] = ss1(Q[ 1]) + H2[ 2];	Q[ 2] = ss2(Q[ 2]) + H2[ 3];	Q[ 3] = ss3(Q[ 3]) + H2[ 4];
		Q[ 4] = ss4(Q[ 4]) + H2[ 5];	Q[ 5] = ss0(Q[ 5]) + H2[ 6];	Q[ 6] = ss1(Q[ 6]) + H2[ 7];	Q[ 7] = ss2(Q[ 7]) + H2[ 8];
		Q[ 8] = ss3(Q[ 8]) + H2[ 9];	Q[ 9] = ss4(Q[ 9]) + H2[10];	Q[10] = ss0(Q[10]) + H2[11];	Q[11] = ss1(Q[11]) + H2[12];
		Q[12] = ss2(Q[12]) + H2[13];	Q[13] = ss3(Q[13]) + H2[14];	Q[14] = ss4(Q[14]) + H2[15];	Q[15] = ss0(Q[15]) + H2[ 0];

		/* This is the Message expansion or f_1 in the documentation. It has 16 rounds. Blue Midnight Wish has two tunable security parameters. */
		/* The parameters are named EXPAND_1_ROUNDS and EXPAND_2_ROUNDS. The following relation for these parameters should is satisfied:	*/
		/* EXPAND_1_ROUNDS + EXPAND_2_ROUNDS = 16                           									*/
		tmp[ 0] = ROTL32(M32[ 0], 1);	tmp[ 1] = ROTL32(M32[ 1], 2);	tmp[ 2] = ROTL32(M32[ 2], 3);	tmp[ 3] = ROTL32(M32[ 3], 4);
		tmp[ 4] = ROTL32(M32[ 4], 5);	tmp[ 5] = ROTL32(M32[ 5], 6);	tmp[ 6] = ROTL32(M32[ 6], 7);	tmp[ 7] = ROL8(M32[ 7]);
		tmp[ 8] = ROTL32(M32[ 8], 9);	tmp[ 9] = ROTL32(M32[ 9],10);	tmp[10] = ROTL32(M32[10],11);	tmp[11] = ROTL32(M32[11],12);
		tmp[12] = ROTL32(M32[12],13);	tmp[13] = ROTL32(M32[13],14);	tmp[14] = ROTL32(M32[14],15);	tmp[15] = ROL16(M32[15]);
		
		Q[16] = ss1(Q[ 0]) + ss2(Q[ 1]) + ss3(Q[ 2]) + ss0(Q[ 3]) + ss1(Q[ 4]) + ss2(Q[ 5]) + ss3(Q[ 6]) + ss0(Q[ 7])
		      + ss1(Q[ 8]) + ss2(Q[ 9]) + ss3(Q[10]) + ss0(Q[11]) + ss1(Q[12]) + ss2(Q[13]) + ss3(Q[14]) + ss0(Q[15])
		      + ((shl(0x05555555,4) + tmp[ 0] + tmp[ 3] - tmp[10]) ^ H2[ 7]);
		Q[17] = ss1(Q[ 1]) + ss2(Q[ 2]) + ss3(Q[ 3]) + ss0(Q[ 4]) + ss1(Q[ 5]) + ss2(Q[ 6]) + ss3(Q[ 7]) + ss0(Q[ 8])
		      + ss1(Q[ 9]) + ss2(Q[10]) + ss3(Q[11]) + ss0(Q[12]) + ss1(Q[13]) + ss2(Q[14]) + ss3(Q[15]) + ss0(Q[16])
		      + ((17U*(0x05555555) + tmp[ 1] + tmp[ 4] - tmp[11]) ^ H2[ 8]);

		tmp2[ 0] = Q[ 2] + Q[ 4] + Q[ 6] + Q[ 8] + Q[10] + Q[12] + Q[14];
		tmp2[ 1] = Q[ 3] + Q[ 5] + Q[ 7] + Q[ 9] + Q[11] + Q[13] + Q[15];

		Q[18] = rs1(Q[ 3])+rs2(Q[ 5])+rs3(Q[ 7])+rs4(Q[ 9])+rs5(Q[11])+rs6(Q[13])+rs7(Q[15])+ss4(Q[16])+ss5(Q[17])+tmp2[ 0]+((18U*(0x05555555) + tmp[ 2] + tmp[ 5] - tmp[12]) ^ H2[ 9]);
		Q[19] = rs1(Q[ 4])+rs2(Q[ 6])+rs3(Q[ 8])+rs4(Q[10])+rs5(Q[12])+rs6(Q[14])+rs7(Q[16])+ss4(Q[17])+ss5(Q[18])+tmp2[ 1]+((19U*(0x05555555) + tmp[ 3] + tmp[ 6] - tmp[13]) ^ H2[10]);
		
		tmp2[ 0]+= Q[16] - Q[ 2];
		tmp2[ 1]+= Q[17] - Q[ 3];
		
		Q[20] = rs1(Q[ 5])+rs2(Q[ 7])+rs3(Q[ 9])+rs4(Q[11])+rs5(Q[13])+rs6(Q[15])+rs7(Q[17])+ss4(Q[18])+ss5(Q[19])+tmp2[ 0]+((20U*(0x05555555) + tmp[ 4] + tmp[ 7] - tmp[14]) ^ H2[11]);
		Q[21] = rs1(Q[ 6])+rs2(Q[ 8])+rs3(Q[10])+rs4(Q[12])+rs5(Q[14])+rs6(Q[16])+rs7(Q[18])+ss4(Q[19])+ss5(Q[20])+tmp2[ 1]+((21U*(0x05555555) + tmp[ 5] + tmp[ 8] - tmp[15]) ^ H2[12]);
		
		tmp2[ 0]+= Q[18] - Q[ 4];
		tmp2[ 1]+= Q[19] - Q[ 5];
		
		Q[22] = rs1(Q[ 7])+rs2(Q[ 9])+rs3(Q[11])+rs4(Q[13])+rs5(Q[15])+rs6(Q[17])+rs7(Q[19])+ss4(Q[20])+ss5(Q[21])+tmp2[ 0]+((22U*(0x05555555) + tmp[ 6] + tmp[ 9] - tmp[ 0]) ^ H2[13]);
		Q[23] = rs1(Q[ 8])+rs2(Q[10])+rs3(Q[12])+rs4(Q[14])+rs5(Q[16])+rs6(Q[18])+rs7(Q[20])+ss4(Q[21])+ss5(Q[22])+tmp2[ 1]+((23U*(0x05555555) + tmp[ 7] + tmp[10] - tmp[ 1]) ^ H2[14]);
		
		tmp2[ 0]+= Q[20] - Q[ 6];
		tmp2[ 1]+= Q[21] - Q[ 7];
		
		Q[24] = rs1(Q[ 9])+rs2(Q[11])+rs3(Q[13])+rs4(Q[15])+rs5(Q[17])+rs6(Q[19])+rs7(Q[21])+ss4(Q[22])+ss5(Q[23])+tmp2[ 0]+((24U*(0x05555555) + tmp[ 8] + tmp[11] - tmp[ 2]) ^ H2[15]);
		Q[25] = rs1(Q[10])+rs2(Q[12])+rs3(Q[14])+rs4(Q[16])+rs5(Q[18])+rs6(Q[20])+rs7(Q[22])+ss4(Q[23])+ss5(Q[24])+tmp2[ 1]+((25U*(0x05555555) + tmp[ 9] + tmp[12] - tmp[ 3]) ^ H2[ 0]);
		
		tmp2[ 0]+= Q[22] - Q[ 8];
		tmp2[ 1]+= Q[23] - Q[ 9];
		
		Q[26] = rs1(Q[11])+rs2(Q[13])+rs3(Q[15])+rs4(Q[17])+rs5(Q[19])+rs6(Q[21])+rs7(Q[23])+ss4(Q[24])+ss5(Q[25])+tmp2[ 0]+((26U*(0x05555555) + tmp[10] + tmp[13] - tmp[ 4]) ^ H2[ 1]);
		Q[27] = rs1(Q[12])+rs2(Q[14])+rs3(Q[16])+rs4(Q[18])+rs5(Q[20])+rs6(Q[22])+rs7(Q[24])+ss4(Q[25])+ss5(Q[26])+tmp2[ 1]+((27U*(0x05555555) + tmp[11] + tmp[14] - tmp[ 5]) ^ H2[ 2]);
		
		tmp2[ 0]+= Q[24] - Q[10];
		tmp2[ 1]+= Q[25] - Q[11];
		
		Q[28] = rs1(Q[13])+rs2(Q[15])+rs3(Q[17])+rs4(Q[19])+rs5(Q[21])+rs6(Q[23])+rs7(Q[25])+ss4(Q[26])+ss5(Q[27])+tmp2[ 0]+((28U*(0x05555555) + tmp[12] + tmp[15] - tmp[ 6]) ^ H2[ 3]);
		Q[29] = rs1(Q[14])+rs2(Q[16])+rs3(Q[18])+rs4(Q[20])+rs5(Q[22])+rs6(Q[24])+rs7(Q[26])+ss4(Q[27])+ss5(Q[28])+tmp2[ 1]+((29U*(0x05555555) + tmp[13] + tmp[ 0] - tmp[ 7]) ^ H2[ 4]);
		
		tmp2[ 0]+= Q[26] - Q[12];
		tmp2[ 1]+= Q[27] - Q[13];
		
		Q[30] = rs1(Q[15])+rs2(Q[17])+rs3(Q[19])+rs4(Q[21])+rs5(Q[23])+rs6(Q[25])+rs7(Q[27])+ss4(Q[28])+ss5(Q[29])+tmp2[ 0]+((30U*(0x05555555) + tmp[14] + tmp[ 1] - tmp[ 8]) ^ H2[ 5]);
		Q[31] = rs1(Q[16])+rs2(Q[18])+rs3(Q[20])+rs4(Q[22])+rs5(Q[24])+rs6(Q[26])+rs7(Q[28])+ss4(Q[29])+ss5(Q[30])+tmp2[ 1]+((31U*(0x05555555) + tmp[15] + tmp[ 2] - tmp[ 9]) ^ H2[ 6]);

		/* Blue Midnight Wish has two temporary cummulative variables that accumulate via XORing */
		/* 16 new variables that are produced in the Message Expansion part.                    */
		XL32 = Q[16] ^ Q[17] ^ Q[18] ^ Q[19] ^ Q[20] ^ Q[21] ^ Q[22] ^ Q[23];
		XH32 = XL32  ^ Q[24] ^ Q[25] ^ Q[26] ^ Q[27] ^ Q[28] ^ Q[29] ^ Q[30] ^ Q[31];

		M32[ 3] = (M32[ 3] ^ shl(Q[19], 5) ^ shr(XH32, 1)) + (Q[27] ^ Q[ 3] ^ XL32);
		M32[15] = ROL16(M32[ 3]) + (Q[31] ^ M32[15] ^ XH32) + (Q[22] ^ Q[15] ^ shr(XL32, 2));
		
		if (M32[15] <= target.y){
			M32[ 2] = xor3x(shr(XH32, 5), shl(Q[18], 5), M32[ 2]) + xor3x(XL32, Q[26], Q[ 2]);
			M32[14] = ROTL32(M32[ 2], 15)		  + xor3x(XH32, Q[30], M32[14]) + xor3x(shr(XL32, 7), Q[21], Q[14]);
			if (M32[14] <= target.x){
				uint32_t tmp = atomicExch(&nonceVector[0], thread);
				if (tmp != 0)
					nonceVector[1] = tmp;
			}
		}
	}
}

__host__
void bmw256_cpu_hash_32(int thr_id, uint32_t threads, uint2 *g_hash, uint32_t *resultnonces, const uint2 target)
{
	const dim3 grid((threads + TPB - 1) / TPB);
	const dim3 block(TPB);

	bmw256_gpu_hash_32<<<grid, block>>>(threads, g_hash, d_GNonce[thr_id], target);
//	cudaThreadSynchronize();
	cudaMemcpy(d_gnounce[thr_id], d_GNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	resultnonces[0] = *(d_gnounce[thr_id]);
	resultnonces[1] = *(d_gnounce[thr_id] + 1);
}


__host__
void bmw256_cpu_init(int thr_id)
{
	cudaMalloc(&d_GNonce[thr_id], 2 * sizeof(uint32_t));
	cudaMallocHost(&d_gnounce[thr_id], 2 * sizeof(uint32_t));
}

__host__
void bmw_set_output(int thr_id)
{
	cudaMemset(d_GNonce[thr_id], 0, 2 * sizeof(uint32_t));
}

__host__
void bmw256_cpu_free(int thr_id)
{
	cudaFree(d_GNonce[thr_id]);
	cudaFreeHost(d_gnounce[thr_id]);
}
