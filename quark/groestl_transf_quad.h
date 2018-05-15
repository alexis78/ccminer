#define merge8(z,x,y,b)\
	z=__byte_perm(x, y, b); \

#define SWAP8(x,y)\
	x=__byte_perm(x, y, 0x5410); \
	y=__byte_perm(y, x, 0x5410);

#define SWAP4(x,y)\
	t = 0xf0f0f0f0 & (x ^ (y<<4)); \
	x^= t; \
	y^= t>>4;

#define SWAP2(x,y)\
	t = 0xcccccccc & (x ^ (y<<2)); \
	x^= t; \
	y^= t>>2;

#define SWAP1(x,y)\
	t = 0xaaaaaaaa & (x ^ (y<<1)); \
	x^= t; \
	y^= t>>1;

__device__ 
static void to_bitslice_quad(uint32_t *const __restrict__ input, uint32_t *const __restrict__ output){

	uint32_t other[8];
	uint32_t t;

	uint32_t perm = (threadIdx.x & 1) ? 0x7362 : 0x5140;
	const uint32_t n = threadIdx.x & 3;
		
		#pragma unroll 8
		for (uint32_t i = 0; i < 8; i++)
		{
			input[i] = __shfl(input[i], (n ^ (3 * ((n == 1) || (n == 2)))&3), 4);
			other[i] = __shfl(input[i], (threadIdx.x + 1) & 3, 4);
			input[i] = __shfl(input[i], threadIdx.x & 2, 4);
			other[i] = __shfl(other[i], threadIdx.x & 2, 4);
		}

		merge8(output[0], input[0], input[4], perm);
		merge8(output[1], other[0], other[4], perm);
		merge8(output[2], input[1], input[5], perm);
		merge8(output[3], other[1], other[5], perm);
		merge8(output[4], input[2], input[6], perm);
		merge8(output[5], other[2], other[6], perm);
		merge8(output[6], input[3], input[7], perm);
		merge8(output[7], other[3], other[7], perm);

		SWAP1(output[0], output[1]);
		SWAP1(output[2], output[3]);
		SWAP1(output[4], output[5]);
		SWAP1(output[6], output[7]);

		SWAP2(output[0], output[2]);
		SWAP2(output[1], output[3]);
		SWAP2(output[4], output[6]);
		SWAP2(output[5], output[7]);

		SWAP4(output[0], output[4]);
		SWAP4(output[1], output[5]);
		SWAP4(output[2], output[6]);
		SWAP4(output[3], output[7]);
}

__device__ 
static void from_bitslice_quad52(const uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{
	uint32_t t;
	const uint32_t perm = 0x7531;//(threadIdx.x & 1) ? 0x3175 : 0x7531;

	output[0] = __byte_perm(input[0], input[4], perm);
	output[2] = __byte_perm(input[1], input[5], perm);
	output[8] = __byte_perm(input[2], input[6], perm);
	output[10] = __byte_perm(input[3], input[7], perm);

	SWAP1(output[0], output[2]);
	SWAP1(output[8], output[10]);
	SWAP2(output[0], output[8]);
	SWAP2(output[2], output[10]);

	t = __byte_perm(output[0], output[8], 0x5410);
	output[8] = __byte_perm(output[0], output[8], 0x7632);
	output[0] = t;

	t = __byte_perm(output[2], output[10], 0x5410);
	output[10] = __byte_perm(output[2], output[10], 0x7632);
	output[2] = t;

	SWAP4(output[0], output[8]);
	SWAP4(output[2], output[10]);

	if (threadIdx.x & 1)
	{
		output[4] = __byte_perm(output[0], 0, 0x3232);
		output[0] = __byte_perm(output[0], 0, 0x1032);

		output[6] = __byte_perm(output[2], 0, 0x3232);
		output[2] = __byte_perm(output[2], 0, 0x1032);

		output[12] = __byte_perm(output[8], 0, 0x3232);
		output[8] = __byte_perm(output[8], 0, 0x1032);

		output[14] = __byte_perm(output[10], 0, 0x3232);
		output[10] = __byte_perm(output[10], 0, 0x1032);
	}
	else
	{
		output[4] = output[0];
		output[6] = output[2];
		output[12] = output[8];
		output[14] = output[10];
	}
}
