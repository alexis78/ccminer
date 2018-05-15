/*
 * whirlpool routine (djm)
 */
extern "C"
{
#include "sph/sph_whirlpool.h"
}
#include "miner.h"
#include "cuda_helper.h"

#define NBN 2

// Memory for the hash functions
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void whirlpool512_setBlock_80(void *pdata, const void *ptarget);
extern void whirlpool512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_resNonce, const uint64_t target);

// CPU Hash function
extern "C" void wcoinhash(void *state, const void *input)
{
	sph_whirlpool_context ctx_whirlpool;

	unsigned char hash[128]; // uint32_t hashA[16], hashB[16];
	#define hashB hash+64

	memset(hash, 0, sizeof hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, input, 80);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hash, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hashB);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hashB, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hash, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_whirl(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{

	int dev_id = device_map[thr_id];

	uint32_t endiandata[20];
	uint32_t* pdata = work->data;
	uint32_t* ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	uint32_t default_throughput;
	if(device_sm[dev_id]<=500) default_throughput = 1<<22;
	else if(device_sm[dev_id]<=520) default_throughput = 1<<25;
	else if((device_sm[dev_id]<=520) && is_windows()) default_throughput = 1<<24;
	else if(device_sm[dev_id]>520) default_throughput = 1<<25;
	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput); // 19=256*256*8;
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		
		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		
		x15_whirlpool_cpu_init(thr_id, throughput, 1 /* old whirlpool */);
		
//		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
		if(h_resNonce[thr_id] == NULL){
			gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
			exit(EXIT_FAILURE);
		}
		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}

	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
	whirlpool512_setBlock_80((void*)endiandata, ptarget);
	do {
		whirlpool512_cpu_hash_80(thr_id, throughput, pdata[19], d_resNonce[thr_id], *(uint64_t*)&ptarget[ 6]);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0] != UINT32_MAX){
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][0]);
			wcoinhash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				work_set_target_ratio(work, vhash64);
				pdata[19] = startNounce + h_resNonce[thr_id][0];
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					pdata[21] = startNounce+h_resNonce[thr_id][1];
//					if(!opt_quiet)
//						gpulog(LOG_BLUE,dev_id,"Found 2nd nonce: %08x", pdata[21]);
					be32enc(&endiandata[19], pdata[21]);
					wcoinhash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0]){
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19],pdata[21]);
					}
					res++;
				}
				return res;					
			}else {
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
extern "C" void free_whirl(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	x15_whirlpool_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}

