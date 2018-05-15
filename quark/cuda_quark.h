#include "cuda_helper.h"

/* commonly used cuda quark kernels prototypes */

extern void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_bmw512_cpu_hash_64_quark(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void blake512_bmw512_cpu_setBlock_80(int thr_id, uint32_t *endiandata);
extern void quark_blake512_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash);

extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_groestl512_sm20_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_keccak512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash, uint64_t target, uint32_t *d_resNonce);

extern void quark_keccak_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_keccak_skein512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash,uint32_t *d_resNonce,const uint64_t highTarget);

extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_jh512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash, uint64_t target, uint32_t *d_resNonce);
extern void quark_jh512_4way_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t* d_hash);

extern void quark_compactTest_cpu_init(int thr_id, uint32_t threads);
extern void quark_compactTest_cpu_free(int thr_id);
extern void quark_compactTest_cpu_hash_64(int thr_id,uint32_t threads,uint32_t *inpHashes,uint32_t *d_validNonceTable,uint32_t *d_nonces1,uint32_t *nrm1,uint32_t *d_nonces2,uint32_t *nrm2);
extern void quark_compactTest_single_false_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *inpHashes, uint32_t *d_validNonceTable,uint32_t *d_nonces1, uint32_t *nrm1);

