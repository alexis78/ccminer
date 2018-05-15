/**
 * Blake2-B CUDA Implementation
 *
 * tpruvot@github July 2016
 *
 */

#include <miner.h>

#include <string.h>
#include <stdint.h>

#include <sph/blake2b.h>

#include <cuda_helper.h>
#include <cuda_vectors.h>

#define TPB 512
#define NBN 2

static uint32_t *d_resNonces[MAX_GPUS];
static uint32_t *h_resNonces[MAX_GPUS];

static __constant__ uint2 _ALIGN(16) c_data[10];
static __constant__ uint2 _ALIGN(16) c_v[16];

static __constant__ const uint32_t blake2b_sigma[12][16] = {
	{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } , { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  } ,
	{ 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  } , { 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  } ,
	{ 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 } , { 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  } ,
	{ 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 } , { 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 } ,
	{ 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  } , { 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  } ,
	{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } , { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  }
};

extern "C" void blake2b_hash(void *output, const void *input)
{
	uint8_t _ALIGN(64) hash[32];
	blake2b_ctx ctx;

	blake2b_init(&ctx, 32, NULL, 0);
	blake2b_update(&ctx, input, 80);
	blake2b_final(&ctx, hash);

	memcpy(output, hash, 32);
}

// ----------------------------------------------------------------

__device__ __forceinline__
static void G(const int r, const int i, uint2 &a, uint2 &b, uint2 &c, uint2 &d,const uint2 m[16])
{
	a = a + b + m[ blake2b_sigma[r][2*i] ];
	d = SWAPUINT2( d ^ a );
	c = c + d;
	b = ROR24( b ^ c );
	a = a + b + m[ blake2b_sigma[r][2*i+1] ];
	d = ROR16( d ^ a );
	c = c + d;
	b = ROR2( b ^ c, 63);
}

#define ROUND(r) \
	G(r, 0, v[0], v[4], v[ 8], v[12], m); \
	G(r, 1, v[1], v[5], v[ 9], v[13], m); \
	G(r, 2, v[2], v[6], v[10], v[14], m); \
	G(r, 3, v[3], v[7], v[11], v[15], m); \
	G(r, 4, v[0], v[5], v[10], v[15], m); \
	G(r, 5, v[1], v[6], v[11], v[12], m); \
	G(r, 6, v[2], v[7], v[ 8], v[13], m); \
	G(r, 7, v[3], v[4], v[ 9], v[14], m);

__global__ __launch_bounds__(512,1)
void blake2b_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint32_t target6)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	
	if(thread<threads){

		const uint32_t nonce = thread + startNonce;

		uint2 v[16];
		uint2 m[16];

		*(uint2x4*)&m[0] = *(uint2x4*)&c_data[0];
		*(uint2x4*)&m[4] = *(uint2x4*)&c_data[4];
		m[4].x = nonce;
		m[8] = c_data[8];
		m[9] = c_data[9];

		m[10] = m[11] = make_uint2(0,0);
		m[12] = m[13] = m[14] = m[15] = make_uint2(0,0);

		#pragma unroll 4
		for(uint32_t i=0;i<16;i+=4){
			*(uint2x4*)&v[i] = *(uint2x4*)&c_v[i];
		}

		v[ 2] = v[ 2] + m[4];
		v[14] = SWAPUINT2( v[14] ^ v[2] );
		v[10] = v[10] + v[14];
		v[ 6] = ROR24( v[ 6] ^ v[10] );
		v[ 2] = v[ 2] + v[ 6] + m[ 5];
		v[14] = ROR16( v[14] ^ v[ 2] );
		v[10] = v[10] + v[14];
		v[ 6] = ROR2( v[ 6] ^ v[10], 63);


		v[10] = v[10] + v[15];
		v[ 5] = ROR24( v[ 5] ^ v[10] );
		v[ 0] = v[ 0] + v[ 5];
		v[15] = ROR16(v[15] ^ v[0]);
		v[10] = v[10] + v[15];
		v[ 5] = ROR2( v[ 5] ^ v[10], 63);
		
		G(0, 5, v[1], v[6], v[11], v[12], m);
		G(0, 6, v[2], v[7], v[ 8], v[13], m);
		G(0, 7, v[3], v[4], v[ 9], v[14], m);
		ROUND( 1 );
		ROUND( 2 );
		ROUND( 3 );
		ROUND( 4 );
		ROUND( 5 );
		ROUND( 6 );
		ROUND( 7 );
		ROUND( 8 );
		ROUND( 9 );
		ROUND( 10 );
//		ROUND_F( 11 );
		G(11, 0, v[0], v[4], v[ 8], v[12], m);
		G(11, 1, v[1], v[5], v[ 9], v[13], m);
		G(11, 2, v[2], v[6], v[10], v[14], m);
		G(11, 3, v[3], v[7], v[11], v[15], m);
//		G(11, 4, v[0], v[5], v[10], v[15], m);
		v[ 0] = v[ 0] + v[ 5] + m[ 1];
		v[15] = SWAPUINT2( v[15] ^ v[0] );
		v[10] = v[10] + v[15];
		v[ 5] = ROR24( v[ 5] ^ v[10] );
		v[ 0] = v[ 0] + v[ 5];
//		G(11, 5, v[1], v[6], v[11], v[12], m);

//		H(11, 6, v[2], v[7], v[ 8], v[13], m);
		v[ 2] = v[ 2] + v[ 7] + m[blake2b_sigma[11][12]];
		v[13] = SWAPUINT2( v[13] ^ v[2]);
		v[ 8] = v[ 8] + v[13];
		v[ 7] = ROR24( v[7] ^ v[8] );
		v[ 2] = v[ 2] + v[ 7] + m[blake2b_sigma[11][13]];
		v[13] = ROR16( v[13] ^ v[2] );
		v[ 8] = v[ 8] + v[13];
		
		if (xor3x(v[8].x, v[0].x, 0xf2bdc928) == 0){
			if (cuda_swab32(0x6a09e667 ^ v[0].y ^ v[8].y ) <= target6) {
				uint32_t tmp = atomicExch(&resNonce[0], nonce);
				if (tmp != UINT32_MAX)
					resNonce[1] = tmp;
			}
		}
	}
}

__host__
uint32_t blake2b_hash_cuda(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint32_t target6, uint32_t &secNonce)
{
	uint32_t resNonces[NBN] = { UINT32_MAX, UINT32_MAX };
	uint32_t result = UINT32_MAX;

	if (cudaSuccess == cudaMemcpy(resNonces, d_resNonces[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		result = resNonces[0];
		secNonce = resNonces[1];
		if (secNonce == result) secNonce = UINT32_MAX;
	}
	return result;
}

__host__
void blake2b_setBlock(uint32_t *data)
{
	uint64_t v[16] = {
		0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
		0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1, 0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294, 0x5be0cd19137e2179
	};
	uint64_t m[16];
	memcpy(m,data,80);
	memset(&m[10],0x00,6*sizeof(uint64_t));

	v[ 0]+= v[ 4] + m[ 0];
	v[12] = ROTR64(v[12] ^ v[ 0],32);
	v[ 8]+= v[12];
	v[ 4] = ROTR64(v[ 4] ^ v[ 8],24);
	v[ 0]+= v[ 4] + m[ 1];
	v[12] = ROTR64(v[12] ^ v[ 0],16);
	v[ 8]+= v[12];
	v[ 4] = ROTR64(v[ 4] ^ v[ 8],63);

	v[ 1] = v[ 1] + v[ 5] + m[ 2];
	v[13] = ROTR64( v[13] ^ v[1],32);
	v[ 9] = v[ 9] + v[13];
	v[ 5] = ROTR64( v[5] ^ v[9],24);
	v[ 1] = v[ 1] + v[ 5] + m[ 3];
	v[13] = ROTR64( v[13] ^ v[1],16);
	v[ 9] = v[ 9] + v[13];
	v[ 5] = ROTR64( v[5] ^ v[9], 63);

	v[ 2] = v[ 2] + v[ 6];

	v[ 3] = v[ 3] + v[ 7] + m[6];
	v[15] = ROTR64( v[15] ^ v[3] ,32);
	v[11] = v[11] + v[15];
	v[ 7] = ROTR64( v[7] ^ v[11] ,24);
	v[ 3] = v[ 3] + v[ 7] + m[7];
	v[15] = ROTR64( v[15] ^ v[3] ,16);
	v[11] = v[11] + v[15];
	v[ 7] = ROTR64( v[7] ^ v[11], 63);

	v[ 0] = v[ 0] + v[ 5] + m[8];
	v[15] = ROTR64( v[15] ^ v[0] ,32);
	v[ 0] = v[ 0] + m[9];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_data, data, 80, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_v, v, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_sia(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done){

	int dev_id = device_map[thr_id];
	
	uint32_t _ALIGN(64) hash[8];
	uint32_t _ALIGN(64) vhashcpu[8];
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t Htarg = ptarget[7];
	const uint32_t first_nonce = pdata[8];

	int intensity = (device_sm[dev_id] > 500)?29:28;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO,dev_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonces[thr_id], NBN * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonces[thr_id], NBN * sizeof(uint32_t)), -1);

		init[thr_id] = true;
	}
	const dim3 grid((throughput + TPB-1)/TPB);
	const dim3 block(TPB);

	memcpy(endiandata, pdata, 80);
	endiandata[11] = 0; // nbits

	blake2b_setBlock(endiandata);

	cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t));

	do {
		blake2b_gpu_hash <<<grid, block, 8>>> (throughput, pdata[8], d_resNonces[thr_id], ptarget[6]);
		
		cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);
		
		if (h_resNonces[thr_id][0] != UINT32_MAX){
			int res = 0;
			endiandata[8] = h_resNonces[thr_id][0];
			blake2b_hash(hash, endiandata);
			// sia hash target is reversed (start of hash)
			swab256(vhashcpu, hash);
			if (vhashcpu[7] <= Htarg && fulltest(vhashcpu, ptarget)) {
				work_set_target_ratio(work, vhashcpu);
				*hashes_done = pdata[8] - first_nonce + throughput +1;
				work->nonces[0] = h_resNonces[thr_id][0];
				pdata[8] = h_resNonces[thr_id][0];
				res=1;
				if (h_resNonces[thr_id][1] != UINT32_MAX) {
					endiandata[8] = h_resNonces[thr_id][1];
					blake2b_hash(hash, endiandata);
//					if(!opt_quiet)
//						gpulog(LOG_BLUE, dev_id, "Found 2nd nonce: %08x", h_resNonces[thr_id][1]);
					swab256(vhashcpu, hash);
					work->nonces[1] = h_resNonces[thr_id][1];
					pdata[21] = h_resNonces[thr_id][1];
					if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio[0]) {
						work_set_target_ratio(work, vhashcpu);
						xchg(work->nonces[0], work->nonces[1]);
						xchg(pdata[8], pdata[21]);
					}
					res=2;
				}
				return res;
			}
		}

		pdata[8] += throughput;

	}while(!work_restart[thr_id].restart && ((uint64_t)max_nonce > (uint64_t)throughput + pdata[8]));

	*hashes_done = pdata[8] - first_nonce +1;

	return 0;
}

// cleanup
extern "C" void free_sia(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_resNonces[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}

// ---- SIA LONGPOLL --------------------------------------------------------------------------------

struct data_buffer {
	void *buf;
	size_t len;
};

extern void calc_network_diff(struct work *work);


size_t sia_data_cb(const void *ptr, size_t size, size_t nmemb, void *user_data){
	struct data_buffer *db = (struct data_buffer *)user_data;
	size_t len = size * nmemb;
	size_t oldlen, newlen;
	void *newmem;
	static const uchar zero = 0;

	oldlen = db->len;
	newlen = oldlen + len;

	newmem = realloc(db->buf, newlen + 1);
	if (!newmem)
		return 0;

	db->buf = newmem;
	db->len = newlen;
	memcpy((char*)db->buf + oldlen, ptr, len);
	memcpy((char*)db->buf + newlen, &zero, 1);	/* null terminate */

	return len;
}

char* sia_getheader(CURL *curl, struct pool_infos *pool)
{
	char curl_err_str[CURL_ERROR_SIZE] = { 0 };
	struct data_buffer all_data = { 0 };
	struct curl_slist *headers = NULL;
	char data[256] = { 0 };
	char url[512];

	// nanopool
	snprintf(url, 512, "%s/miner/header?address=%s&worker=%s", //&longpoll
		pool->url, pool->user, pool->pass);

	if (opt_protocol)
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_POST, 0);
	curl_easy_setopt(curl, CURLOPT_ENCODING, "");
	curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0);
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
	curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, opt_timeout);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);
	curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_err_str);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, sia_data_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &all_data);

	headers = curl_slist_append(headers, "Accept: application/octet-stream");
	headers = curl_slist_append(headers, "Expect:"); // disable Expect hdr
	headers = curl_slist_append(headers, "User-Agent: Sia-Agent"); // required for now
//	headers = curl_slist_append(headers, "User-Agent: " USER_AGENT);
//	headers = curl_slist_append(headers, "X-Mining-Extensions: longpoll");

	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

	int rc = curl_easy_perform(curl);
	if (rc && strlen(curl_err_str)) {
		applog(LOG_WARNING, "%s", curl_err_str);
	}

	if (all_data.len >= 112)
		cbin2hex(data, (const char*) all_data.buf, 112);
	if (opt_protocol || all_data.len != 112)
		applog(LOG_DEBUG, "received %d bytes: %s", (int) all_data.len, data);

	curl_slist_free_all(headers);

	return rc == 0 && all_data.len ? strdup(data) : NULL;
}

bool sia_work_decode(const char *hexdata, struct work *work)
{
	uint8_t target[32];
	if (!work) return false;

	hex2bin((uchar*)target, &hexdata[0], 32);
	swab256(work->target, target);
	work->targetdiff = target_to_diff(work->target);

	hex2bin((uchar*)work->data, &hexdata[64], 80);
	// high 16 bits of the 64 bits nonce
	work->data[9] = rand() << 16;

	// use work ntime as job id
	cbin2hex(work->job_id, (const char*)&work->data[10], 4);
	calc_network_diff(work);

	if (stratum_diff != work->targetdiff) {
		stratum_diff = work->targetdiff;
		applog(LOG_WARNING, "Pool diff set to %g", stratum_diff);
	}

	return true;
}

extern int share_result(int result, int pooln, double sharediff, const char *reason);

bool sia_submit(CURL *curl, struct pool_infos *pool, struct work *work){

	char curl_err_str[CURL_ERROR_SIZE] = { 0 };
	struct data_buffer all_data = { 0 };
	struct curl_slist *headers = NULL;
	char buf[256] = { 0 };
	char url[512];

	if (opt_protocol)
		applog_hex(work->data, 80);
	//applog_hex(&work->data[8], 16);
	//applog_hex(&work->data[10], 4);

	// nanopool
	snprintf(url, 512, "%s/miner/header?address=%s&worker=%s",
		pool->url, pool->user, pool->pass);

	if (opt_protocol)
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_ENCODING, "");
	curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);
	curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1);
	curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_err_str);
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10);

	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &all_data);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, sia_data_cb);

	memcpy(buf, work->data, 80);
	curl_easy_setopt(curl, CURLOPT_POST, 1);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 80);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, (void*) buf);

//	headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
//	headers = curl_slist_append(headers, "Content-Length: 80");
	headers = curl_slist_append(headers, "Accept:"); // disable Accept hdr
	headers = curl_slist_append(headers, "Expect:"); // disable Expect hdr
	headers = curl_slist_append(headers, "User-Agent: Sia-Agent");
//	headers = curl_slist_append(headers, "User-Agent: " USER_AGENT);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

	int res = curl_easy_perform(curl) == 0;
	long errcode;
	CURLcode c = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &errcode);
	if (errcode != 204) {
		if (strlen(curl_err_str))
			applog(LOG_ERR, "submit err %ld %s", errcode, curl_err_str);
		res = 0;
	}
	share_result(res, work->pooln, work->sharediff[0], res ? NULL : (char*) all_data.buf);

	curl_slist_free_all(headers);
	return true;
}
