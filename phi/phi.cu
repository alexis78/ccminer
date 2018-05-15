//
//
//  PHI1612 algo
//  Skein + JH + CubeHash + Fugue + Gost + Echo
//
//  Implemented by anorganix @ bitcointalk
//  Feel free to send some satoshis to 1Bitcoin8tfbtGAQNFxDRUVUfFgFWKoWi9
//
//  Changes
//      - 01.10.2017
//          - initial release
//
//      - 07.10.2017
//          - speed increase due to using a faster GOST implementation
//          - changed default intensity for 10-series cards
//
//      - 06.01.2018
//          - imported Fugue algo from tpruvot@github to replace the one from alexis78@github
//          - speed increase due to some JH and Fugue tweaks
//
//

extern "C"
{
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_fugue.h"
#include "sph/sph_streebog.h"
#include "sph/sph_echo.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x13_fugue512_cpu_free(int thr_id);

extern void streebog_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);

#include <stdio.h>
#include <memory.h>

#define NBN 2

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern "C" void phihash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_skein512_context ctx_skein;
	sph_jh512_context ctx_jh;
	sph_cubehash512_context ctx_cubehash;
	sph_fugue512_context ctx_fugue;
	sph_gost512_context ctx_gost;
	sph_echo512_context ctx_echo;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*)hash, 64);
	sph_gost512_close(&ctx_gost, (void*)hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	memcpy(output, hash, 32);
}

#define _DEBUG_PREFIX "phi"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_phi(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	int dev_id = device_map[thr_id];

	uint32_t default_throughput = 1 << 20;

	if (device_sm[dev_id] <= 520)
	{
		default_throughput = 1 << 19;
	}

	if (device_sm[dev_id] <= 500)
	{
		default_throughput = 1 << 18;
	}

	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput);
	throughput &= 0xFFFFFF70;

	if (opt_benchmark)
	{
		ptarget[7] = 0xf;
	}

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		if (opt_cudaschedule == -1 && gpu_threads == 1)
		{
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u CUDA threads", throughput2intensity(throughput), throughput);

		x13_fugue512_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));

		h_resNonce[thr_id] = (uint32_t*)malloc(NBN * sizeof(uint32_t));

		if (h_resNonce[thr_id] == NULL)
		{
			gpulog(LOG_ERR, thr_id, "Host memory allocation failed");
			exit(EXIT_FAILURE);
		}

		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];

	for (int k = 0; k < 20; k++)
	{
		be32enc(&endiandata[k], pdata[k]);
	}

	skein512_cpu_setBlock_80(endiandata);
	cudaMemset(d_resNonce[thr_id], 0xff, NBN * sizeof(uint32_t));

	do
	{
		skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		quark_jh512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]);
		x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_fugue512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		streebog_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_echo512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], *(uint64_t*)&ptarget[6]);

		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNonce = pdata[19];

			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNonce + h_resNonce[thr_id][0]);
			phihash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;

				*hashes_done = pdata[19] - first_nonce + throughput;
				work_set_target_ratio(work, vhash64);
				pdata[19] = startNonce + h_resNonce[thr_id][0];

				if (h_resNonce[thr_id][1] != UINT32_MAX)
				{
					pdata[21] = startNonce + h_resNonce[thr_id][1];
					be32enc(&endiandata[19], startNonce + h_resNonce[thr_id][1]);
					phihash(vhash64, endiandata);

					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0])
					{
						work_set_target_ratio(work, vhash64);
						xchg(pdata[19], pdata[21]);
					}

					res++;
				}

				return res;
			}
			else
			{
				gpulog(LOG_WARNING, thr_id, "Result for %08x does not validate on CPU", h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN * sizeof(uint32_t));
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > (uint64_t)throughput + pdata[19]));

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_phi(int thr_id)
{
	if (!init[thr_id])
	{
		return;
	}

	cudaDeviceSynchronize();

	free(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	cudaFree(d_hash[thr_id]);

	x13_fugue512_cpu_free(thr_id);
	cuda_check_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
