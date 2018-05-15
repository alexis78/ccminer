extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "lyra2/Lyra2.h"
}

#include "miner.h"
#include "cuda_helper.h"

static uint2* d_hash[MAX_GPUS];
static uint2* d_matrix[MAX_GPUS];

extern void blake256_14round_cpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint2* d_Hash);
extern void blake256_14round_cpu_setBlock_80(const uint32_t *pdata);

extern void keccak256_cpu_hash_32(const int thr_id,const uint32_t threads, uint2* d_hash);
extern void keccak256_cpu_init(int thr_id);
extern void keccak256_cpu_free(int thr_id);

extern void skein256_cpu_init(int thr_id);
extern void skein256_cpu_hash_32(const uint32_t threads, uint2 *d_hash);

extern void lyra2_cpu_init(int thr_id, uint32_t threads, uint2* d_matrix);
extern void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint2* d_outputHash);

extern void groestl256_cpu_init(int thr_id, uint32_t threads);
extern void groestl256_cpu_free(int thr_id);
extern void groestl256_setTarget(const void *ptarget);
extern void groestl256_set_output(int thr_id);
extern void groestl256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint2 *d_Hash, uint32_t *resultnonces);


#ifdef _DEBUG
#define TRACE(algo) { \
	if (max_nonce == 1 && pdata[19] <= 1) { \
		uint32_t* debugbuf = NULL; \
		cudaMallocHost(&debugbuf, 8*sizeof(uint32_t)); \
		cudaMemcpy(debugbuf, d_hash[thr_id], 8*sizeof(uint32_t), cudaMemcpyDeviceToHost); \
		printf("lyra %s %08x %08x %08x %08x...\n", algo, swab32(debugbuf[0]), swab32(debugbuf[1]), \
			swab32(debugbuf[2]), swab32(debugbuf[3])); \
		cudaFreeHost(debugbuf); \
	} \
}
#else
#define TRACE(algo) {}
#endif

extern "C" void lyra2re_hash(void *state, const void *input)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context     ctx_blake;
	sph_keccak256_context    ctx_keccak;
	sph_skein256_context     ctx_skein;
	sph_groestl256_context   ctx_groestl;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashA, 32);
	sph_skein256_close(&ctx_skein, hashB);

	sph_groestl256_init(&ctx_groestl);
	sph_groestl256(&ctx_groestl, hashB, 32);
	sph_groestl256_close(&ctx_groestl, hashA);

	memcpy(state, hashA, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_lyra2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{

	const int dev_id = device_map[thr_id];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[dev_id] > 500 ) ? 19 : 17;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity); // 18=256*256*4;
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
			CUDA_LOG_ERROR();
		}

		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		
		keccak256_cpu_init(thr_id);
		skein256_cpu_init(thr_id);
		groestl256_cpu_init(thr_id, throughput);

		// DMatrix
//		size_t matrix_sz = sizeof(uint64_t) * 4 * 4;
//		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
//		lyra2_cpu_init(thr_id, throughput, d_matrix[thr_id]);
		size_t matrix_sz = device_sm[dev_id] > 500 ? sizeof(uint64_t) * 4 * 4 : 16 * 8 * 8 * sizeof(uint2);
		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
		lyra2_cpu_init(thr_id, throughput, d_matrix[thr_id]);
//		cudaMalloc(&d_matrix[thr_id], (size_t)16 * 8 * 8 * sizeof(uint2) * throughput);
//		lyra2_cpu_init(thr_id, throughput, d_matrix[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)32 * throughput));

		init[thr_id] = true;
	}

	uint32_t _ALIGN(128) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256_14round_cpu_setBlock_80(pdata);
	groestl256_setTarget(ptarget);
	groestl256_set_output(thr_id);
	do {
		uint32_t foundNonces[2];

		blake256_14round_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
		keccak256_cpu_hash_32(thr_id,throughput, d_hash[thr_id]);
		lyra2_cpu_hash_32(thr_id, throughput, d_hash[thr_id]);
		skein256_cpu_hash_32(throughput, d_hash[thr_id]);
		groestl256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], foundNonces);

		if (foundNonces[ 0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash64[8];

			be32enc(&endiandata[19], foundNonces[ 0]);
			lyra2re_hash(vhash64, endiandata);

			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				int res = 1;

				work_set_target_ratio(work, vhash64);
				*hashes_done = pdata[19] - first_nonce + throughput;
				pdata[19] = foundNonces[ 0];
				if (foundNonces[ 1] != UINT32_MAX)
				{
					be32enc(&endiandata[19], foundNonces[ 1]);
					lyra2re_hash(vhash64, endiandata);
//					if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
//						applog(LOG_NOTICE,"Legit extranonce");
						if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0])
							work_set_target_ratio(work, vhash64);
						pdata[21] = foundNonces[ 1];
						res++;
//					}
//					if(!opt_quiet)
//						applog(LOG_BLUE,"Found 2nd nonce: %08X",pdata[21]);
				}
				return res;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonces[ 0]);
				groestl256_set_output(thr_id);
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_lyra2(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);

	keccak256_cpu_free(thr_id);
	groestl256_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
