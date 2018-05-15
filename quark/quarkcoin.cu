extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}

#include "miner.h"

#include "cuda_helper.h"
#include "cuda_quark.h"

#include <stdio.h>

static uint32_t *d_hash[MAX_GPUS];

// Speicher zur Generierung der Noncevektoren für die bedingten Hashes
static uint32_t *d_branch1Nonces[MAX_GPUS];
static uint32_t *d_branch2Nonces[MAX_GPUS];
static uint32_t *d_branch3Nonces[MAX_GPUS];

static uint32_t h_resNonce[MAX_GPUS][4];
static uint32_t *d_resNonce[MAX_GPUS];

// Original Quarkhash Funktion aus einem miner Quelltext
extern "C" void quarkhash(void *state, const void *input)
{
	unsigned char _ALIGN(128) hash[64];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;

	sph_blake512_init(&ctx_blake);
	sph_blake512 (&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, (void*) hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*) hash);

	if (hash[0] & 0x8)
	{
		sph_groestl512_init(&ctx_groestl);
		sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
		sph_groestl512_close(&ctx_groestl, (void*) hash);
	}
	else
	{
		sph_skein512_init(&ctx_skein);
		sph_skein512 (&ctx_skein, (const void*) hash, 64);
		sph_skein512_close(&ctx_skein, (void*) hash);
	}

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*) hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512 (&ctx_jh, (const void*) hash, 64);
	sph_jh512_close(&ctx_jh, (void*) hash);

	if (hash[0] & 0x8)
	{
		sph_blake512_init(&ctx_blake);
		sph_blake512 (&ctx_blake, (const void*) hash, 64);
		sph_blake512_close(&ctx_blake, (void*) hash);
	}
	else
	{
		sph_bmw512_init(&ctx_bmw);
		sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
		sph_bmw512_close(&ctx_bmw, (void*) hash);
	}

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512 (&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, (void*) hash);

	if (hash[0] & 0x8)
	{
		sph_keccak512_init(&ctx_keccak);
		sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
		sph_keccak512_close(&ctx_keccak, (void*) hash);
	}
	else
	{
		sph_jh512_init(&ctx_jh);
		sph_jh512 (&ctx_jh, (const void*) hash, 64);
		sph_jh512_close(&ctx_jh, (void*) hash);
	}

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_quark(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	const uint64_t highTarget = *(uint64_t*)&ptarget[6];

	int dev_id = device_map[thr_id];
	uint32_t default_throughput = 1U << 22;
	default_throughput+=3774720; //22.9
	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x00F;

	if (!init[thr_id])
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

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));

		quark_compactTest_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_branch1Nonces[thr_id], throughput*sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc(&d_branch2Nonces[thr_id], throughput*sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc(&d_branch3Nonces[thr_id], throughput*sizeof(uint32_t)));

		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 4 * sizeof(uint32_t)));

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	
	int rc = 0;
	cudaMemset(d_resNonce[thr_id], 0xFFFFFFFF, 4*sizeof(uint32_t));	
	do {
		uint32_t nrm1=0, nrm2=0, nrm3=0;

		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		quark_bmw512_cpu_hash_64_quark(thr_id, throughput, d_hash[thr_id]);
		
		quark_compactTest_single_false_cpu_hash_64(thr_id, throughput, d_hash[thr_id], NULL, d_branch3Nonces[thr_id], &nrm3);
		quark_skein512_cpu_hash_64(thr_id, nrm3, d_branch3Nonces[thr_id], d_hash[thr_id]);
		quark_groestl512_cpu_hash_64(thr_id, nrm3, d_branch3Nonces[thr_id], d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, nrm3, d_branch3Nonces[thr_id], d_hash[thr_id]);

		quark_compactTest_cpu_hash_64(thr_id, nrm3, d_hash[thr_id], d_branch3Nonces[thr_id], d_branch1Nonces[thr_id], &nrm1, d_branch2Nonces[thr_id], &nrm2);
		
		quark_blake512_cpu_hash_64(thr_id, nrm1, d_branch1Nonces[thr_id], d_hash[thr_id]);
		quark_bmw512_cpu_hash_64(thr_id, nrm2, d_branch2Nonces[thr_id], d_hash[thr_id]);
		quark_keccak_skein512_cpu_hash_64(thr_id, nrm3, d_branch3Nonces[thr_id], d_hash[thr_id]);

		quark_compactTest_cpu_hash_64(thr_id, nrm3, d_hash[thr_id], d_branch3Nonces[thr_id], d_branch1Nonces[thr_id], &nrm1, d_branch2Nonces[thr_id], &nrm2);

		quark_keccak512_cpu_hash_64_final(thr_id, nrm1, d_branch1Nonces[thr_id], d_hash[thr_id],highTarget,&d_resNonce[thr_id][0]);
		quark_jh512_cpu_hash_64_final(thr_id, nrm2, d_branch2Nonces[thr_id], d_hash[thr_id],highTarget,&d_resNonce[thr_id][2]);

		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], 4*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		int pos = -1;
		if (h_resNonce[thr_id][0] != UINT32_MAX)
			pos = 0;
		else if (h_resNonce[thr_id][2] != UINT32_MAX)
			pos = 2;
		
		if (pos != -1 ){
			const uint32_t startNounce = pdata[19];
			uint32_t vhash[8];
			be32enc(&endiandata[19], (startNounce+h_resNonce[thr_id][pos]));
			quarkhash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work_set_target_ratio(work, vhash);
				*hashes_done = pdata[19] + throughput - first_nonce;
				pdata[19] = (startNounce+h_resNonce[thr_id][pos]);
//				work->nonces[ 0] = pdata[19];
				rc = 1;
				//check for another nonce
				for(int i=pos+1;i<4;i++){
					if(h_resNonce[thr_id][i] != UINT32_MAX){
						be32enc(&endiandata[19], (startNounce+h_resNonce[thr_id][i]));
						quarkhash(vhash, endiandata);
						if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
//							work_set_target_ratio(work, vhash);
							pdata[21] = (startNounce+h_resNonce[thr_id][i]);
							if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]){
								work_set_target_ratio(work, vhash);
								xchg(pdata[19],pdata[21]);
							}
//							work->nonces[ 0] = pdata[21];
//							if(!opt_quiet)
//								applog(LOG_BLUE,"Found 2nd nonce: %08X",pdata[21]);
							rc++;
							return rc;
						}else{
							gpulog(LOG_WARNING, thr_id, "2nd nonce result for %08x does not validate on CPU!", h_resNonce[thr_id][i]);
							applog_hash((uchar*) vhash);
							applog_hash((uchar*) ptarget);
							break;
						}
					}
				}
				return rc;
			}
			else{
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][pos]);
				cudaMemset(d_resNonce[thr_id], 0xFFFFFFFF, 4*sizeof(uint32_t));	
				applog_hash((uchar*) vhash);
				applog_hash((uchar*) ptarget);
			}
		}
		pdata[19] += throughput;

	}while(!work_restart[thr_id].restart && ((uint64_t)max_nonce > (uint64_t)throughput + pdata[19]));

	*hashes_done = pdata[19] - first_nonce;

	return rc;
}

// cleanup
extern "C" void free_quark(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	//cudaFreeHost(h_resNonce[thr_id]);
	
	cudaFree(d_branch1Nonces[thr_id]);
	cudaFree(d_branch2Nonces[thr_id]);
	cudaFree(d_branch3Nonces[thr_id]);

	quark_compactTest_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
