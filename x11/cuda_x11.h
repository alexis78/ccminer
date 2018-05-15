#include "quark/cuda_quark.h"

extern void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x11_cubehash_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x11_simd512_cpu_init(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_simd512_cpu_free(int thr_id);

extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads,uint32_t *d_hash);
extern void x11_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);

extern void x11_simd_echo_512_cpu_init(int thr_id, uint32_t threads);
extern void x11_simd_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);
extern void x11_simd_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x11_simd_echo_512_cpu_free(int thr_id);

