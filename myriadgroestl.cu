#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>

#include "sph/sph_groestl.h"

#include "miner.h"
#include "cuda_helper.h"

#define NBN 2
//#define NPT 2

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

void myriadgroestl_cpu_init(int thr_id, uint32_t threads);
void myriadgroestl_cpu_free(int thr_id);
void myriadgroestl_cpu_setBlock(int thr_id, void *data);
void myriadgroestl_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_resNounce, const uint64_t target);

void myriadhash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_groestl512_context ctx_groestl;
	SHA256_CTX sha256;

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, input, 80);
	sph_groestl512_close(&ctx_groestl, hash);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256,(unsigned char *)hash, 64);
	SHA256_Final((unsigned char *)hash, &sha256);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_myriad(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[32];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t first_nonce = pdata[19];
	int dev_id = device_map[thr_id];
	int intensity = 23;//(device_sm[dev_id] >= 600) ? 20 : 18;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x00000000;

	// init
	if(!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		myriadgroestl_cpu_init(thr_id, throughput);
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)));
		
		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	myriadgroestl_cpu_setBlock(thr_id, endiandata);
	
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
	int rc = 0;
	do {
		myriadgroestl_cpu_hash(thr_id, throughput, pdata[19], d_resNonce[thr_id], *(uint64_t*)&ptarget[6]);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);
		
		if (h_resNonce[thr_id][0] != UINT32_MAX){
			uint32_t _ALIGN(64) vhash64[8];
			endiandata[19] = swab32(h_resNonce[thr_id][0]);
			myriadhash(vhash64, endiandata);
			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				*hashes_done = pdata[19] - first_nonce + throughput + 1;
				rc = 1;
				work_set_target_ratio(work, vhash64);
				pdata[19] = h_resNonce[thr_id][0];
				work->nonces[0] = pdata[19];
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
//					if(!opt_quiet)
//						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", swab32(h_resNonce[thr_id][1]));
					endiandata[19] = swab32(h_resNonce[thr_id][1]);
					myriadhash(vhash64, endiandata);
					pdata[21] = h_resNonce[thr_id][1];
					work->nonces[1] = pdata[21];
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0]){
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19],pdata[21]);
						xchg(work->nonces[ 0],work->nonces[ 1]);
					}
					rc=2;
				}
				return rc;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));				
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && (uint64_t)max_nonce > (uint64_t)throughput + (uint64_t)pdata[19]);

	*hashes_done = pdata[19] - first_nonce + 1;

	return rc;
}

// cleanup
void free_myriad(int thr_id){

	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	myriadgroestl_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
