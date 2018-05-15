/*
//	Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice
	Based upon Christians, Tanguy Pruvot's and SP's work
		
	Provos Alexis - 2016
*/

#include "cuda_vectors.h"
#include "cuda_helper.h"

#define TPB52 512
#define TPB50 512
#define THF 4

#include "quark/groestl_functions_quad.h"
#include "quark/groestl_transf_quad.h"

__constant__ const uint32_t msg[2][4] = {
						{0x00000080,0,0,0},
						{0,0,0,0x01000000}
					};

#if __CUDA_ARCH__ > 500
__global__ __launch_bounds__(TPB52, 2)
#else
__global__ __launch_bounds__(TPB50, 2)
#endif
void quark_groestl512_gpu_hash_64_quad(uint32_t threads, uint32_t* g_hash, uint32_t* g_nonceVector){
	uint32_t msgBitsliced[8];
	uint32_t state[8];
	uint32_t output[16];
	// durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
	if (thread < threads){
	        // GROESTL
		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : __ldg(&g_nonceVector[thread]);

	        uint32_t *inpHash = &g_hash[hashPosition<<4];

	        const uint32_t thr = threadIdx.x & (THF-1);

		uint32_t message[8] = {
			#if __CUDA_ARCH__ > 500
			__ldg(&inpHash[thr]), __ldg(&inpHash[(THF)+thr]), __ldg(&inpHash[(2 * THF) + thr]), __ldg(&inpHash[(3 * THF) + thr]),msg[0][thr], 0, 0, msg[1][thr]
			#else
			inpHash[thr], inpHash[(THF)+thr], inpHash[(2 * THF) + thr], inpHash[(3 * THF) + thr], msg[0][thr], 0, 0, msg[1][thr]
			#endif
		};

		to_bitslice_quad(message, msgBitsliced);

	        groestl512_progressMessage_quad(state, msgBitsliced,thr);

		from_bitslice_quad52(state, output);

#if __CUDA_ARCH__ <= 500
		output[0] = __byte_perm(output[0], __shfl(output[0], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[2] = __byte_perm(output[2], __shfl(output[2], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[4] = __byte_perm(output[4], __shfl(output[4], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[6] = __byte_perm(output[6], __shfl(output[6], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[8] = __byte_perm(output[8], __shfl(output[8], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[10] = __byte_perm(output[10], __shfl(output[10], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[12] = __byte_perm(output[12], __shfl(output[12], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[14] = __byte_perm(output[14], __shfl(output[14], (threadIdx.x + 1) & 3, 4), 0x7632);
	
		if (thr == 0 || thr == 2){
			output[0 + 1] = __shfl(output[0], (threadIdx.x + 2) & 3, 4);
			output[2 + 1] = __shfl(output[2], (threadIdx.x + 2) & 3, 4);
			output[4 + 1] = __shfl(output[4], (threadIdx.x + 2) & 3, 4);
			output[6 + 1] = __shfl(output[6], (threadIdx.x + 2) & 3, 4);
			output[8 + 1] = __shfl(output[8], (threadIdx.x + 2) & 3, 4);
			output[10 + 1] = __shfl(output[10], (threadIdx.x + 2) & 3, 4);
			output[12 + 1] = __shfl(output[12], (threadIdx.x + 2) & 3, 4);
			output[14 + 1] = __shfl(output[14], (threadIdx.x + 2) & 3, 4);		
			if(thr==0){
				*(uint2x4*)&inpHash[0] = *(uint2x4*)&output[0];
				*(uint2x4*)&inpHash[8] = *(uint2x4*)&output[8];
			}
		}
#else
		output[0] = __byte_perm(output[0], __shfl(output[0], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[0 + 1] = __shfl(output[0], (threadIdx.x + 2) & 3, 4);

		output[2] = __byte_perm(output[2], __shfl(output[2], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[2 + 1] = __shfl(output[2], (threadIdx.x + 2) & 3, 4);
		
		output[4] = __byte_perm(output[4], __shfl(output[4], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[4 + 1] = __shfl(output[4], (threadIdx.x + 2) & 3, 4);
		
		output[6] = __byte_perm(output[6], __shfl(output[6], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[6 + 1] = __shfl(output[6], (threadIdx.x + 2) & 3, 4);
		
		output[8] = __byte_perm(output[8], __shfl(output[8], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[8 + 1] = __shfl(output[8], (threadIdx.x + 2) & 3, 4);

		output[10] = __byte_perm(output[10], __shfl(output[10], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[10 + 1] = __shfl(output[10], (threadIdx.x + 2) & 3, 4);
		
		output[12] = __byte_perm(output[12], __shfl(output[12], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[12 + 1] = __shfl(output[12], (threadIdx.x + 2) & 3, 4);
		
		output[14] = __byte_perm(output[14], __shfl(output[14], (threadIdx.x + 1) & 3, 4), 0x7632);
		output[14 + 1] = __shfl(output[14], (threadIdx.x + 2) & 3, 4);

		if(thr==0){
			*(uint2x4*)&inpHash[0] = *(uint2x4*)&output[0];
			*(uint2x4*)&inpHash[8] = *(uint2x4*)&output[8];
		}
#endif
	}
}

__host__
void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash){

	const int dev_id = device_map[thr_id];
	// Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
	// mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
	// berechne wie viele Thread Blocks wir brauchen
	uint32_t tpb = (device_sm[dev_id] <= 500) ? TPB50 : TPB52;

	dim3 grid((THF*threads + tpb-1)/tpb);
	dim3 block(tpb);
	quark_groestl512_gpu_hash_64_quad<<<grid, block>>>(threads, d_hash, d_nonceVector);
}
