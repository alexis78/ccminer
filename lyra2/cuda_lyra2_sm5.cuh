/**
* Lyra2 (v1) cuda implementation based on djm34 work - SM 5/5.2
* tpruvot@github 2015
* 
* Provos Alexis - 2016
*/

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#undef __CUDA_ARCH__
#define __CUDA_ARCH__ 500
#endif

#if __CUDA_ARCH__ == 500

__device__ uint2x4* DMatrix;

static __device__ __forceinline__
void Gfunc(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; d ^= a; d = SWAPUINT2(d);
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);
}

static __device__ __forceinline__
void round_lyra(uint2x4* s)
{
	Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
}

static __device__ __forceinline__
void reduceDuplex(uint2x4 state[4],const uint32_t thread)
{
	uint2x4 state1[3];

	const uint32_t ps1 = (256 * thread);
	const uint32_t ps2 = (memshift * 7 + memshift * 8 + 256 * thread);

	#pragma unroll 4
	for (int i = 0; i < 8; i++){
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 - i*memshift;

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix+s1)[j]);
			
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		#pragma unroll
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state1[j];
	}
}

static __device__ __forceinline__
void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2x4 state[4],const uint32_t thread)
{
	uint2x4 state1[3], state2[3];

	const uint32_t ps1 = (             memshift*8 * rowIn    + 256 * thread);
	const uint32_t ps2 = (             memshift*8 * rowInOut + 256 * thread);
	const uint32_t ps3 = (memshift*7 + memshift*8 * rowOut   + 256 * thread);

	#pragma nounroll
	for (int i = 0; i < 8; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;
		const uint32_t s3 = ps3 - i*memshift;
		
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j]= __ldg4(&(DMatrix + s1)[j]);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j]= __ldg4(&(DMatrix + s2)[j]);

		#pragma unroll
		for (int j = 0; j < 3; j++) {
			uint2x4 tmp = state1[j] + state2[j];
			state[j] ^= tmp;
		}

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++) {
			(DMatrix + s3)[j] = state1[j]^state[ j];
		}

		((uint2*)state2)[0] ^= ((uint2*)state)[11];

		#pragma unroll
		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j+1] ^= ((uint2*)state)[j];

		#pragma unroll
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state2[j];
	}
}

static __device__ __forceinline__
void reduceDuplexRowt(const int rowIn, const int rowInOut, const int rowOut, uint2x4* state, const uint32_t thread)
{
	const uint32_t ps1 = (memshift * 8 * rowIn    + 256 * thread);
	const uint32_t ps2 = (memshift * 8 * rowInOut + 256 * thread);
	const uint32_t ps3 = (memshift * 8 * rowOut   + 256 * thread);

	#pragma unroll 1
	for (int i = 0; i < 8; i++)
	{
		uint2x4 state1[3], state2[3];

		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;
		const uint32_t s3 = ps3 + i*memshift;
		
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] += state2[j];

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		((uint2*)state2)[0] ^= ((uint2*)state)[11];

		#pragma unroll
		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

		if (rowInOut != rowOut) {
			#pragma unroll
			for (int j = 0; j < 3; j++){
				state1[j]=state[j]^(__ldg4(&(DMatrix + s3)[j]));
			}
			#pragma unroll
			for (int j = 0; j < 3; j++){
				(DMatrix + s3)[j]=state1[j];
			}
		}else{
			#pragma unroll
			for (int j = 0; j < 3; j++)
				state2[j] ^= state[j];
		}
		#pragma unroll
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state2[j];		
	}
}

__global__ __launch_bounds__(TPB50, 1)
void lyra2_gpu_hash_32(uint32_t threads, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint2x4 state[4];

		((uint2*)state)[0] = __ldg(&g_hash[thread]);
		((uint2*)state)[1] = __ldg(&g_hash[thread + threads]);
		((uint2*)state)[2] = __ldg(&g_hash[thread + threads*2]);
		((uint2*)state)[3] = __ldg(&g_hash[thread + threads*3]);

		state[1] = state[0];
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<24; i++)
			round_lyra(state); //because 12 is not enough 

		const uint32_t ps1 = (memshift * 7  + 256 * thread);
		
		for (int i = 0; i < 8; i++)
		{
			const uint32_t s1 = ps1 - memshift * i;

			for (int j = 0; j < 3; j++)
				(DMatrix + s1)[j] = (state)[j];
			round_lyra(state);
		}

		reduceDuplex(state, thread);

		reduceDuplexRowSetup(1, 0, 2, state,  thread);
		reduceDuplexRowSetup(2, 1, 3, state,  thread);

		for(uint32_t i=3;i<7;i++){
			reduceDuplexRowSetup(i, 8%(i+1), i+1, state,  thread);		
		}

		uint32_t rowa;
		int prev=7;
		#pragma unroll 1
		for(int i=0;i<23;i+=3){
			rowa = ((uint2*)state)[0].x & 7;
			reduceDuplexRowt(prev, rowa, i&7, state, thread);
			prev = i&7;
		}
		
		const uint32_t shift = (memshift * 8 * rowa + 256 * thread);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + shift)[j]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

		g_hash[thread]             = ((uint2*)state)[0];
		g_hash[thread + threads]   = ((uint2*)state)[1];
		g_hash[thread + threads*2] = ((uint2*)state)[2];
		g_hash[thread + threads*3] = ((uint2*)state)[3];
	}
}
#else
/* for unsupported SM arch */
__global__ void lyra2_gpu_hash_32(uint32_t threads, uint2 *g_hash) {}
#endif
