#include <cuda_runtime.h>
#include <string.h>

#include <miner.h>
#include "cuda_helper.h"

#include "neoscrypt.h"

#define NBN 2

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern void neoscrypt_setBlockTarget(uint32_t* pdata);
extern void neoscrypt_cpu_init(int thr_id, uint32_t threads);
extern void neoscrypt_free(int thr_id);
extern void neoscrypt_cpu_hash_k4(bool stratum, int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *result,const uint32_t target7);

static bool init[MAX_GPUS] = { 0 };

int scanhash_neoscrypt(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done){

	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	int intensity = 14;

	if (strstr(device_name[dev_id], "GTX 10")) intensity = 16;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	
	if (strstr(device_name[dev_id], "GTX 9")) throughput = 45312;
	
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);
	
	api_set_throughput(thr_id, throughput);

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if(!init[thr_id]){
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			cudaGetLastError(); // reset errors if device is not "reset"
		}
		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		
		neoscrypt_cpu_init(thr_id, throughput);
		
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
		if(h_resNonce[thr_id] == NULL){
			gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
			exit(EXIT_FAILURE);
		}
		
		CUDA_LOG_ERROR();
		init[thr_id] = true;
	}

	if (have_stratum) {
		for (int k = 0; k < 20; k++)
			be32enc(&endiandata[k], pdata[k]);
	} else {
		for (int k = 0; k < 20; k++)
			endiandata[k] = pdata[k];
	}

	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));

	neoscrypt_setBlockTarget(endiandata);

	int rc = 0;

	do{
		neoscrypt_cpu_hash_k4(have_stratum, thr_id, throughput, pdata[19], d_resNonce[thr_id], ptarget[7]);

		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if(h_resNonce[thr_id][0] != UINT32_MAX){
			uint32_t vhash[8];
			if(have_stratum)
				be32enc(&endiandata[19], h_resNonce[thr_id][0]);
			else
				endiandata[19] = h_resNonce[thr_id][0];
			neoscrypt((uchar*)vhash,(uchar*)endiandata, 0x80000620);
			if(vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)){
				*hashes_done = pdata[19] - first_nonce + throughput;
				work_set_target_ratio(work, vhash);
				pdata[19] = h_resNonce[thr_id][0];
				rc = 1;
				if(h_resNonce[thr_id][1] != UINT32_MAX){
//					if(!opt_quiet)
//						gpulog(LOG_BLUE,dev_id,"Found second nonce %08x !", h_resNonce[thr_id][1]);
					if(have_stratum)
						be32enc(&endiandata[19], h_resNonce[thr_id][1]);
					else
						endiandata[19] = h_resNonce[thr_id][1];
					neoscrypt((uchar*)vhash,(uchar*)endiandata, 0x80000620);
					pdata[21] = h_resNonce[thr_id][1];
					if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]){
						work_set_target_ratio(work, vhash);
						xchg(pdata[19],pdata[21]);
					}
					rc = 2;
				}
				return rc;
			}
			else{
				if(vhash[7] != ptarget[7])
					applog(LOG_WARNING, "GPU #%d: Nonce $%08X does not validate on CPU!", device_map[thr_id], h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
			}
		}
		pdata[19] += throughput;
	} while(!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));
	*hashes_done = pdata[19] - first_nonce;
	return rc;
}

// cleanup
void free_neoscrypt(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	neoscrypt_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
