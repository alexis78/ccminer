# allow to use Host cuda functions in C/C++
DEF_INCLUDES = @CUDA_INCLUDES@

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.md README.txt LICENSE.txt \
			  ccminer.sln ccminer.vcxproj ccminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c cpuminer-config.h.in

SUBDIRS = compat

bin_PROGRAMS = ccminer

ccminer_SOURCES	= elist.h miner.h compat.h \
			compat/inttypes.h compat/stdbool.h compat/unistd.h \
			compat/sys/time.h compat/getopt/getopt.h \
			crc32.c \
			ccminer.cpp pools.cpp util.cpp bench.cpp bignum.cpp \
			api.cpp hashlog.cpp nvml.cpp stats.cpp sysinfos.cpp cuda.cpp \
			Algo256/decred.cu \
			Algo256/blake256_8round.cu Algo256/blake256_14round.cu Algo256/cuda_blake256_14round.cu \
			Algo256/cuda_keccak256.cu Algo256/keccak256.cu \
			Algo256/cuda_skein256.cu \
			Algo256/cuda_groestl256.cu \
			Algo256/cuda_bmw256.cu \
			Algo256/cuda_blake2s.cu \
			Algo256/cuda_cubehash256.cu \
			lyra2/Lyra2.c lyra2/Sponge.c lyra2/lyra2RE.cu lyra2/cuda_lyra2.cu lyra2/lyra2REv2.cu lyra2/cuda_lyra2v2.cu \
			cuda_checkhash.cu \
			sph/blake.c sph/keccak.c sph/shavite.c sph/simd.c sph/groestl.c\
			sph/sha2.c sph/cubehash.c sph/skein.c sph/bmw.c sph/jh.c \
			sph/echo.c sph/luffa.c sph/fugue.c sph/hamsi.c \
			sph/whirlpool.c sph/shabal.c sph/sha2big.c sph/haval.c sph/ripemd.c sph/sph_sha2.c \
			sph/streebog.c \
			sph/blake2b.c \
			neoscrypt/neoscrypt-cpu.c neoscrypt/cuda_neoscrypt.cu \
			neoscrypt/neoscrypt.cu \
			skein/skein.cu skein/skein2.cu \
			nist5/cuda_nist5.cu \
			quark/cuda_keccak_skein512.cu \
			quark/cuda_quark_jh512.cu quark/cuda_quark_blake512.cu quark/cuda_quark_groestl512.cu quark/cuda_quark_skein512.cu \
			quark/cuda_bmw512.cu quark/cuda_quark_keccak512.cu quark/cuda_quark_compactionTest.cu \
			quark/quarkcoin.cu \
			qubit/qubit_luffa512.cu \
			qubit/qubit.cu \
			x11/cuda_x11_cubehash_shavite_512.cu \
			x11/cuda_x11_simd_echo512.cu \
			x11/cuda_x11_cubehash512.cu x11/cuda_x11_shavite512.cu x11/cuda_x11_simd512.cu x11/cuda_x11_echo.cu \
			x11/x11.cu x11/x11evo.cu x11/c11.cu \
			streebog/cuda_streebog.cu streebog/cuda_keccak_streebog.cu \
			streebog/sib.cu streebog/veltor.cu \
			x13/cuda_x13_hamsi_fugue512.cu \
			x13/cuda_x13_fugue512.cu x13/cuda_x13_hamsi512.cu \
			x13/x13.cu \
			x13/hsr.cu x13/cuda_hsr_sm3.cu x13/sm3.c \
			x14/cuda_x14_shabal512.cu \
			x14/x14.cu \
			x15/cuda_x15_whirlpool.cu \
			x15/x15.cu \
			x15/whirlpool.cu \
			x17/cuda_x17_haval256.cu x17/cuda_x17_sha512.cu \
			x17/x17.cu \
			lbry/cuda_sha256_lbry.cu lbry/cuda_sha512_lbry.cu \
			lbry/cuda_lbry_merged.cu \
			lbry/lbry.cu \
			sia.cu \
			cuda_myriadgroestl.cu \
			myriadgroestl.cu

if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif

if HAVE_WINDOWS
ccminer_SOURCES += compat/winansi.c
endif

ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @PCILIB@ @OPENMP_CFLAGS@ @LIBS@ $(nvml_libs)
ccminer_CPPFLAGS = @LIBCURL_CPPFLAGS@ @PCIFLAGS@ @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) $(DEF_INCLUDES) $(nvml_defs)

nvcc_ARCH = -gencode=arch=compute_52,code=\"sm_52,compute_52\"
nvcc_ARCH += -gencode=arch=compute_50,code=\"sm_50,compute_50\"
#nvcc_ARCH = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
#nvcc_ARCH += -gencode=arch=compute_30,code=\"sm_30,compute_30\"
#nvcc_ARCH += -gencode=arch=compute_20,code=\"sm_21,compute_20\"

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. @CUDA_CFLAGS@
nvcc_FLAGS += $(JANSSON_INCLUDES) --ptxas-options="-v" -lineinfo

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

streebog/cuda_streebog.o : streebog/cuda_streebog.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=72 -o $@ -c $<

x11/cuda_x11_simd_echo512.o: x11/cuda_x11_simd_echo512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=72 -o $@ -c $<

x15/cuda_x15_whirlpool.o: x15/cuda_x15_whirlpool.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=72 -o $@ -c $<
