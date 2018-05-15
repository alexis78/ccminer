
extern "C"
{
#include "sph/yescrypt.h"
}

#include "cuda_helper.h"
#include "miner.h"

#define NBN 2

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern void yescrypt_setBlockTarget(uint32_t * data, const void *ptarget);
extern void yescrypt_cpu_init(int thr_id, int threads);
extern void yescrypt_free(int thr_id);
extern void yescrypt_cpu_hash_k4(int thr_id, int threads, uint32_t startNounce, uint32_t* resNonce);
  

static bool init[MAX_GPUS] = { 0 };

int scanhash_yescrypt(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){

	int dev_id = device_map[thr_id];
	
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	int coef = 16;
	if (device_sm[device_map[thr_id]] == 500) coef = 6;
	if (device_sm[device_map[thr_id]] == 350) coef = 2;

	const int throughput = 64*coef;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]); 
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}
		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		
		yescrypt_cpu_init(thr_id, throughput);

		cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t));
		cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t));

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	yescrypt_setBlockTarget(pdata,ptarget);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
	int rc = 0;
	do {
		yescrypt_cpu_hash_k4(thr_id, throughput, pdata[19], d_resNonce[thr_id]);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if(h_resNonce[thr_id][0] != UINT32_MAX){
			uint32_t vhash64[8];
			be32enc(&endiandata[19], h_resNonce[thr_id][0]);
			yescrypt_hash((uchar*) endiandata, (uchar*)vhash64);

			if ( (vhash64[7] <= ptarget[7]) && fulltest(vhash64, ptarget)) {
				*hashes_done = pdata[19] - first_nonce + throughput;
				pdata[19] = h_resNonce[thr_id][0];
				rc = 1;

				if(h_resNonce[thr_id][1] != UINT32_MAX){
//					if(!opt_quiet)
//						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", h_resNonce[thr_id][1]);
					endiandata[19] = h_resNonce[thr_id][1];
					yescrypt_hash((uchar*)endiandata, (uchar*)vhash64);
					pdata[21] = h_resNonce[thr_id][1];
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio){
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19],pdata[21]);
					}
					rc = 2;
				}
				return rc;
			} else {
				gpulog(LOG_INFO,dev_id, "Result for nonce $%08X does not validate on CPU!", h_resNonce[thr_id]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
			}
		}
		pdata[19] += throughput;
	}while(!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));
	*hashes_done = pdata[19] - first_nonce + 1;
	return rc;
}

// cleanup
void free_yescrypt(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	yescrypt_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
