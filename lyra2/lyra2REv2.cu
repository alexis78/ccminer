extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "lyra2/Lyra2.h"
}

#include "miner.h"
#include "cuda_helper.h"


static uint2 *d_hash[MAX_GPUS];
static uint2 *d_matrix[MAX_GPUS];

extern void blake256_14round_cpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint2* d_Hash);
extern void blake256_14round_cpu_setBlock_80(const uint32_t *pdata);

extern void keccak256_cpu_hash_32(const int thr_id,const uint32_t threads, uint2* d_hash);
extern void keccak256_cpu_init(int thr_id);
extern void keccak256_cpu_free(int thr_id);

extern void skein256_cpu_hash_32(const uint32_t threads, uint2 *d_hash);
extern void skein256_cpu_init(int thr_id);

extern void cubehash256_cpu_hash_32(const uint32_t threads, uint2* d_hash);

extern void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads,uint2* DMatrix, uint2 *d_Hash);
extern void lyra2v2_cpu_init(int thr_id, uint2 *hash);

extern void bmw256_setTarget(const void *ptarget);
extern void bmw256_cpu_init(int thr_id);
extern void bmw256_cpu_free(int thr_id);
extern void bmw_set_output(int thr_id);
extern void bmw256_cpu_hash_32(int thr_id, uint32_t threads,uint2 *g_hash, uint32_t *resultnonces, const uint2 target);

void lyra2v2_hash(void *state, const void *input)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context      ctx_blake;
	sph_keccak256_context     ctx_keccak;
	sph_skein256_context      ctx_skein;
	sph_bmw256_context        ctx_bmw;
	sph_cubehash256_context   ctx_cube;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashB, 32);
	sph_cubehash256_close(&ctx_cube, hashA);

	LYRA2(hashB, 32, hashA, 32, hashA, 32, 1, 4, 4);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashB, 32);
	sph_skein256_close(&ctx_skein, hashA);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 32);
	sph_cubehash256_close(&ctx_cube, hashB);

	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 32);
	sph_bmw256_close(&ctx_bmw, hashA);

	memcpy(state, hashA, 32);
}

#ifdef _DEBUG
#define TRACE(algo) { \
	if (max_nonce == 1 && pdata[19] <= 1) { \
		uint32_t* debugbuf = NULL; \
		cudaMallocHost(&debugbuf, 32); \
		cudaMemcpy(debugbuf, d_hash[thr_id], 32, cudaMemcpyDeviceToHost); \
		printf("lyra2 %s %08x %08x %08x %08x...%08x... \n", algo, swab32(debugbuf[0]), swab32(debugbuf[1]), \
			swab32(debugbuf[2]), swab32(debugbuf[3]), swab32(debugbuf[7])); \
		cudaFreeHost(debugbuf); \
	} \
}
#else
#define TRACE(algo) {}
#endif

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_lyra2v2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500) ? 22 : 20;
	uint32_t throughput = cuda_default_throughput(dev_id, 1UL << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x000f;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);			
			CUDA_LOG_ERROR();
		}

		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		
		cuda_get_arch(dev_id);
		skein256_cpu_init(thr_id);
//		keccak256_cpu_init(thr_id);
		bmw256_cpu_init(thr_id);

		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], 4 * 4 * sizeof(uint2) * throughput));
//		lyra2v2_cpu_init(thr_id, d_matrix[thr_id]);
		
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint32_t) * throughput));
//		api_set_throughput(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256_14round_cpu_setBlock_80(pdata);
	bmw_set_output(thr_id);
	do {
		uint32_t foundNonces[2] = { 0, 0 };

		blake256_14round_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
		keccak256_cpu_hash_32(thr_id,throughput, d_hash[thr_id]);
		cubehash256_cpu_hash_32(throughput, d_hash[thr_id]);
		lyra2v2_cpu_hash_32(thr_id, throughput,d_matrix[thr_id],d_hash[thr_id]);
		skein256_cpu_hash_32(throughput, d_hash[thr_id]);
		cubehash256_cpu_hash_32(throughput, d_hash[thr_id]);
		bmw256_cpu_hash_32(thr_id, throughput, d_hash[thr_id], foundNonces, *(uint2*)&ptarget[6]);

		if (foundNonces[0] != 0)
		{
			const uint32_t startNounce = pdata[19];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNounce + foundNonces[0]);
			lyra2v2_hash(vhash64, endiandata);
			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget))
			{
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				work_set_target_ratio(work, vhash64);
				pdata[19] = startNounce + foundNonces[0];
				// check if there was another one...
				if (foundNonces[1] != 0)
				{
					be32enc(&endiandata[19], (pdata[19]-foundNonces[ 0])+foundNonces[1]);
					lyra2v2_hash(vhash64, endiandata);
					pdata[21] = startNounce + foundNonces[1];
//					if(!opt_quiet)
//						applog(LOG_BLUE,"Found 2nd nonce: %08X",pdata[21]);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0]) {
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19], pdata[21]);
					}
					res=2;
				}
				return res;
			}
			else
			{
				if(vhash64[7]>ptarget[ 7])
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonces[0]);
				bmw_set_output(thr_id);
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && !abort_flag);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_lyra2v2(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);

	bmw256_cpu_free(thr_id);
//	keccak256_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
