#ifndef ALGOS_H
#define ALGOS_H

#include <string.h>
#include "compat.h"

enum sha_algos {
	ALGO_BLAKE,
	ALGO_DECRED,
	ALGO_VCASH,
	ALGO_BLAKECOIN,
	ALGO_BLAKE2S,
	ALGO_KECCAK,
	ALGO_HSR,
	ALGO_LYRA2,
	ALGO_LYRA2v2,
	ALGO_SKEIN,
	ALGO_SKEIN2,
	ALGO_NIST5,
	ALGO_QUARK,
	ALGO_QUBIT,
	ALGO_WHIRLPOOL,
	ALGO_X11,
	ALGO_X11EVO,
	ALGO_C11,
	ALGO_SIB,
	ALGO_PHI,
	ALGO_X13,
	ALGO_X14,
	ALGO_X15,
	ALGO_X17,
	ALGO_LBRY,
	ALGO_NEOSCRYPT,
	ALGO_SIA,
	ALGO_MYR_GR,
	ALGO_VELTOR,
//	ALGO_YESCRYPT,
	ALGO_AUTO,
	ALGO_COUNT
};

extern volatile enum sha_algos opt_algo;

static const char *algo_names[] = {
	"blake",
	"decred",
	"vcash",
	"blakecoin",
	"blake2s",
//	"whirlpoolx",
	"keccak",
	"hsr",
	"lyra2",
	"lyra2v2",
	"skein",
	"skein2",
	"nist5",
	"quark",
	"qubit",
	"whirlpool",
	"x11",
	"x11evo",
	"c11",
	"sib",
	"phi",
	"x13",
	"x14",
	"x15",
	"x17",
	"lbry",
	"neoscrypt",
	"sia",
	"myr-gr",
	"veltor",
//	"yescrypt",
	"auto", /* reserved for multi algo */
	""
};

// string to int/enum
static inline int algo_to_int(char* arg)
{
	int i;

	for (i = 0; i < ALGO_COUNT; i++) {
		if (algo_names[i] && !strcasecmp(arg, algo_names[i])) {
			return i;
		}
	}

	if (i == ALGO_COUNT) {
		// some aliases...
		if (!strcasecmp("all", arg))
			i = ALGO_AUTO;
		else if (!strcasecmp("flax", arg))
			i = ALGO_C11;
		else if (!strcasecmp("x13sm3", arg))
			i = ALGO_HSR;
		else if (!strcasecmp("lyra2re", arg))
			i = ALGO_LYRA2;
		else if (!strcasecmp("lyra2rev2", arg))
			i = ALGO_LYRA2v2;
		else if (!strcasecmp("thorsriddle", arg))
			i = ALGO_VELTOR;
		else if (!strcasecmp("whirl", arg))
			i = ALGO_WHIRLPOOL;
		else
			i = -1;
	}

	return i;
}

#endif
