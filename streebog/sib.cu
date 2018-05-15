extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_luffa.h"
#include "sph/sph_echo.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_streebog.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

#include <stdio.h>
#include <memory.h>

#define NBN 2

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern void streebog_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash,uint32_t* d_resNonce);
extern void streebog_set_target(const uint32_t* ptarget);

extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
//extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads,uint32_t *d_hash);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);

extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads,uint32_t *d_hash);
extern void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

// Sibcoin CPU Hash
extern "C" void polyhash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_skein512_context ctx_skein;
	sph_shabal512_context    ctx_shabal;	
	sph_echo512_context ctx_echo;
	sph_luffa512_context ctx_luffa;
	sph_fugue512_context ctx_fugue;
	sph_gost512_context ctx_gost;


        sph_skein512_init(&ctx_skein);
        sph_skein512 (&ctx_skein, input, 80);
        sph_skein512_close(&ctx_skein, (void*) hash);

        sph_shabal512_init(&ctx_shabal);
        sph_shabal512 (&ctx_shabal, (const void*) hash, 64);
        sph_shabal512_close(&ctx_shabal, (void*) hash);

        sph_echo512_init(&ctx_echo);
        sph_echo512 (&ctx_echo, (const void*) hash, 64);
        sph_echo512_close(&ctx_echo, (void*) hash);

        sph_luffa512_init(&ctx_luffa);
        sph_luffa512 (&ctx_luffa, (const void*) hash, 64);
        sph_luffa512_close (&ctx_luffa, (void*) hash);

        sph_fugue512_init(&ctx_fugue);
        sph_fugue512(&ctx_fugue, (const void*) hash, 64);
        sph_fugue512_close(&ctx_fugue, (void*) hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*) hash, 64);
	sph_gost512_close(&ctx_gost, (void*) hash);

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "poly"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_poly(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{

	int dev_id = device_map[thr_id];
	
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t default_throughput;
	if(device_sm[dev_id]<=500) default_throughput = 1<<20;
	else if(device_sm[dev_id]<=520) default_throughput = 1<<21;
	else if(device_sm[dev_id]>520) default_throughput = 1<<22;
	
	if((strstr(device_name[dev_id], "3GB")))default_throughput = 1<<21;
	
	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput); // 19=256*256*8;
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

        if (opt_benchmark)
                ptarget[7] = 0xf;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}

		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
		if(h_resNonce[thr_id] == NULL){
			gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
			exit(EXIT_FAILURE);
		}
		x13_fugue512_cpu_init(thr_id, throughput);	
		init[thr_id] = true;
	}
	
	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

        skein512_cpu_setBlock_80(endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
       	streebog_set_target(ptarget);

	do {
		// Hash with CUDA

		skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);

		x11_luffa512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id]);
		streebog_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id]);

		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0] != UINT32_MAX){
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][0]);
			polyhash(vhash64, endiandata);

			if (vhash64[7] <= Htarg) {
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				work_set_target_ratio(work, vhash64);
				pdata[19] = startNounce + h_resNonce[thr_id][0];
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					pdata[21] = startNounce+h_resNonce[thr_id][1];
//					if(!opt_quiet)
//						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", pdata[21]);
					be32enc(&endiandata[19], startNounce+h_resNonce[thr_id][1]);
					polyhash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0]){
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19],pdata[21]);
					}

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
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > (uint64_t)throughput + pdata[19]));

	*hashes_done = pdata[19] - first_nonce;
	
	return 0;
}

// cleanup
extern "C" void free_poly(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	cudaFree(d_hash[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
