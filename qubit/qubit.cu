/*
 * qubit algorithm
 *
 */
extern "C" {
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}

#include "miner.h"

#include "cuda_helper.h"
#include "x11/cuda_x11.h"

#define NBN 2

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

static uint32_t *d_hash[MAX_GPUS];

extern void qubit_luffa512_cpu_setBlock_80(void *pdata);
extern void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern "C" void qubithash(void *state, const void *input)
{
	uint8_t _ALIGN(128) hash[64];

	// luffa1-cubehash2-shavite3-simd4-echo5

	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, input, 80);
	sph_luffa512_close(&ctx_luffa, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512 (&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512 (&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512 (&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512 (&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_qubit(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){

	int dev_id = device_map[thr_id];

	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t intensity  = (device_sm[dev_id] <= 500) ? 20 : 21;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity); // 256*256*8
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x007f;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		
		x11_simd_echo_512_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
		if(h_resNonce[thr_id] == NULL){
			gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
			exit(EXIT_FAILURE);
		}

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	qubit_luffa512_cpu_setBlock_80((void*)endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
	do {
		// Hash with CUDA
		qubit_luffa512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		x11_cubehash_shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_simd_echo512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], *(uint64_t*)&ptarget[6]);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0] != UINT32_MAX){
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][0]);
			qubithash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				work_set_target_ratio(work, vhash64);
				pdata[19] = startNounce + h_resNonce[thr_id][0];
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
//					if (!opt_quiet)
//						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce %08x ",h_resNonce[thr_id][1]);
				
					be32enc(&endiandata[19], startNounce+h_resNonce[thr_id][1]);
					qubithash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0])
						work_set_target_ratio(work, vhash64);
					pdata[21] = startNounce+h_resNonce[thr_id][1];
					res++;
				}
				return res;
			}
			else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));				
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && (uint64_t)max_nonce > (uint64_t)throughput + pdata[19]);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_qubit(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	cudaFree(d_hash[thr_id]);

	x11_simd_echo_512_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
