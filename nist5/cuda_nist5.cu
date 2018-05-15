extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}

#include "miner.h"

#include "cuda_helper.h"

#include "quark/cuda_quark.h"


static uint32_t *d_hash[MAX_GPUS];
#define NBN 2

/* 8 adapters max */
static uint32_t	*d_resNonce[MAX_GPUS];
static uint32_t	*h_resNonce[MAX_GPUS];

// Original nist5hash Funktion aus einem miner Quelltext
extern "C" void nist5hash(void *state, const void *input)
{
    sph_blake512_context ctx_blake;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
    
    uint8_t hash[64];

    sph_blake512_init(&ctx_blake);
    sph_blake512 (&ctx_blake, input, 80);
    sph_blake512_close(&ctx_blake, (void*) hash);
    
    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
    sph_groestl512_close(&ctx_groestl, (void*) hash);

    sph_jh512_init(&ctx_jh);
    sph_jh512 (&ctx_jh, (const void*) hash, 64);
    sph_jh512_close(&ctx_jh, (void*) hash);

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
    sph_keccak512_close(&ctx_keccak, (void*) hash);

    sph_skein512_init(&ctx_skein);
    sph_skein512 (&ctx_skein, (const void*) hash, 64);
    sph_skein512_close(&ctx_skein, (void*) hash);

    memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_nist5(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done){

	int dev_id = device_map[thr_id];

	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	uint32_t throughput =  cuda_default_throughput(thr_id, 1 << 21); // 256*256*16
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00FF;

	const uint64_t highTarget = *(uint64_t*)&ptarget[6];
	
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

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));				
	int rc = 0;
	do {
		// Hash with CUDA
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		quark_groestl512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		quark_keccak_skein512_cpu_hash_64_final(thr_id, throughput, NULL, d_hash[thr_id],d_resNonce[thr_id],highTarget);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0] != UINT32_MAX){
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNounce+h_resNonce[thr_id][0]);
			nist5hash(vhash64, endiandata);
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				rc = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				work_set_target_ratio(work, vhash64);
				pdata[19] =startNounce + h_resNonce[thr_id][0];
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					be32enc(&endiandata[19], startNounce+h_resNonce[thr_id][1]);
					nist5hash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0])
						work_set_target_ratio(work, vhash64);
					pdata[21] = startNounce+h_resNonce[thr_id][1];
					rc=2;
				}
				return rc;
			}
			else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
			}
		}
		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && (uint64_t)max_nonce > (uint64_t)throughput + (uint64_t)pdata[19]);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// ressources cleanup
extern "C" void free_nist5(int thr_id){
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	cudaFree(d_hash[thr_id]);
	
//	blake512_cpu_free(thr_id);
	
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
