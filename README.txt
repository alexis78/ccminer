
ccMiner release 1.7.1 (Jan 2015) "Sibcoin & Whirlpool midstate"
---------------------------------------------------------------

***************************************************************
If you find this tool useful and like to support its continuous
          development, then consider a donation.

tpruvot@github:
  BTC  : 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo
  DRK  : XeVrkPrWB7pDbdFLfKhF1Z3xpqhsx6wkH3
  ZRC  : ZEcubH2xp2mpuwxMjy7wZThr5AzLGu3mqT

DJM34:
  BTC donation address: 1NENYmxwZGHsKFmyjTc5WferTn5VTFb7Ze

cbuchner v1.2:
  LTC donation address: LKS1WDKGED647msBQfLBHV3Ls8sveGncnm
  BTC donation address: 16hJF5mceSojnTD3ZTUDqdRhDyPJzoRakM

***************************************************************

>>> Introduction <<<

This is a CUDA accelerated mining application which handle :

Saffroncoin blake (256 14-rounds)
Decred
BlakeCoin (256 8-rounds)
Vcash (Blake256 8-rounds - double sha256)
Blake2s (Neva/XVG/TAJ)
Keccak (Maxcoin)
LyraBar Lyra2
Vertcoin Lyra2v2
Skein (Skein + SHA)
Woodcoin (Double Skein)
Nist-5 (Talkcoin/Power)
QuarkCoin family & AnimeCoin
Qubit (Digibyte, ...)
DarkCoin and other X11 coins
Chaincoin and Flaxscript (C11)
Sibcoin (sib)
Revolvercoin (x11evo)
MaruCoin and other X13 coins
BernCoin (X14)
JoinCoin (X15)
VergeCoin(X17)
Library Credits (LBRY)
Neoscrypt (FeatherCoin)
SiaCoin (blake2b)
Myriad-Groestl
Veltor

where some of these coins have a VERY NOTABLE nVidia advantage
over competing AMD (OpenCL Only) implementations.

We did not take a big effort on improving usability, so please set
your parameters carefuly.

THIS PROGRAMM IS PROVIDED "AS-IS", USE IT AT YOUR OWN RISK!

If you're interessted and read the source-code, please excuse
that the most of our comments are in german.

>>> Command Line Interface <<<

This code is based on the pooler cpuminer and inherits
its command line interface and options.

  -a, --algo=ALGO       specify the algorithm to use
		  	blake       Blake256-14rounds(SFR)
		  	decred      Blake256-14rounds(DCR)
  			blakecoin   Blake256-8rounds (BLC)
			vcash       Blake256-8rounds (XVC)
			blake2s	    Blake2s          (NEVA/XVG)
			keccak	    keccak256        (Maxcoin)
			lyra2                        (LyraBar)
			lyra2v2                      (VertCoin)
			skein       Skein SHA2       (AUR/DGB/SKC)
			skein2      Double Skein     (Woodcoin)
			nist5       NIST5            (Talkcoin/Power)
			quark       Quark            (Quarkcoin)
			qubit       Qubit
			x11         X11              (DarkCoin)
			c11         C11              (Chaincoin)
			sib         X11+gost         (Sibcoin)
			x11evo      Permuted x11     (Revolver)
			x13         X13              (MaruCoin)
			x14         X14              (BernCoin)
			x15         X15              (Joincoin)
			x17         X17              (XVG)
			whirlpool   whirlpool        (JoinCoin)
                        lbry        Lbry             (Library Credits)
                        neoscrypt   Neoscrypt        (FTC/PXC/UFO)
                        sia         Sia              (SIAcoin)
                        myr-gr      Myriad-Groestl   (SFR/AUR/DGB/XVG/MYR)
                        veltor      Thor's Riddle(?) (Veltor)

  -d, --devices         gives a comma separated list of CUDA device IDs
                        to operate on. Device IDs start counting from 0!
                        Alternatively give string names of your card like
                        gtx780ti or gt640#2 (matching 2nd gt640 in the PC).

  -i, --intensity=N[,N] GPU threads per call 8-25 (2^N + F, default: 0=auto)
                        Decimals and multiple values are allowed for fine tuning
      --cuda-schedule   Set device threads scheduling mode (default: auto)
  -f, --diff-factor     Divide difficulty by this factor (default 1.0)
  -m, --diff-multiplier Multiply difficulty by this value (default 1.0)
      --vote=VOTE       block reward vote (for HeavyCoin)
      --trust-pool      trust the max block reward vote (maxvote) sent by the pool
  -o, --url=URL         URL of mining server
  -O, --userpass=U:P    username:password pair for mining server
  -u, --user=USERNAME   username for mining server
  -p, --pass=PASSWORD   password for mining server
      --cert=FILE       certificate for mining server using SSL
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy
  -t, --threads=N       number of miner threads (default: number of nVidia GPUs in your system)
  -r, --retries=N       number of times to retry if a network call fails
                          (default: retry indefinitely)
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 15)
      --shares-limit    maximum shares to mine before exiting the program.
      --time-limit      maximum time [s] to mine before exiting the program.
  -T, --timeout=N       network timeout, in seconds (default: 300)
  -s, --scantime=N      upper bound on time spent scanning current work when
                        long polling is unavailable, in seconds (default: 5)
  -n, --ndevs           list cuda devices
  -N, --statsavg        number of samples used to display hashrate (default: 30)
      --no-gbt          disable getblocktemplate support (height check in solo)
      --no-longpoll     disable X-Long-Polling support
      --no-stratum      disable X-Stratum support
  -q, --quiet           disable per-thread hashmeter output
      --no-color        disable colored output
  -D, --debug           enable debug output
  -P, --protocol-dump   verbose dump of protocol-level activities
  -b, --api-bind        IP/Port for the miner API (default: 127.0.0.1:4068)
      --api-remote      Allow remote control
      --max-temp=N      Only mine if gpu temp is less than specified value
      --max-rate=N[KMG] Only mine if net hashrate is less than specified value
      --max-diff=N      Only mine if net difficulty is less than specified value
      --pstate=0        will force the Geforce 9xx to run in P0 P-State
      --plimit=150W     set the gpu power limit, allow multiple values for N cards
      --tlimit=85       Set the gpu thermal limit (windows only)
      --keep-clocks     prevent reset clocks and/or power limit on exit
  -B, --background      run the miner in the background
      --benchmark       run in offline benchmark mode
      --cputest         debug hashes from cpu algorithms
      --cpu-affinity    set process affinity to specific cpu core(s) mask
      --cpu-priority    set process priority (default: 0 idle, 2 normal to 5 highest)
  -c, --config=FILE     load a JSON-format configuration file
                        can be from an url with the http:// prefix
  -V, --version         display version information and exit
  -h, --help            display this help text and exit


>>> Examples <<<


Example for Heavycoin Mining on heavycoinpool.com with a single gpu in your system
    ccminer -t 1 -a heavy -o stratum+tcp://stratum01.heavycoinpool.com:5333 -u <<username.worker>> -p <<workerpassword>> -v 8


Example for Heavycoin Mining on hvc.1gh.com with a dual gpu in your system
    ccminer -t 2 -a heavy -o stratum+tcp://hvcpool.1gh.com:5333/ -u <<WALLET>> -p x -v 8


Example for Fuguecoin solo-mining with 4 gpu's in your system and a Fuguecoin-wallet running on localhost
    ccminer -q -s 1 -t 4 -a fugue256 -o http://localhost:9089/ -u <<myusername>> -p <<mypassword>>


Example for Fuguecoin pool mining on dwarfpool.com with all your GPUs
    ccminer -q -a fugue256 -o stratum+tcp://erebor.dwarfpool.com:3340/ -u YOURWALLETADDRESS.1 -p YOUREMAILADDRESS


Example for Groestlcoin solo mining
    ccminer -q -s 1 -a groestl -o http://127.0.0.1:1441/ -u USERNAME -p PASSWORD


Example for Scrypt-N (2048) on Nicehash
    ccminer -a scrypt:10 -o stratum+tcp://stratum.nicehash.com:3335 -u 3EujYFcoBzWvpUEvbe3obEG95mBuU88QBD -p x

For solo-mining you typically use -o http://127.0.0.1:xxxx where xxxx represents
the rpcport number specified in your wallet's .conf file and you have to pass the same username
and password with -O (or -u -p) as specified in the wallet config.

The wallet must also be started with the -server option and/or with the server=1 flag in the .conf file

>>> Configuration files <<<

With the -c parameter you can use a json config file to set your prefered settings.
An example is present in source tree, and is also the default one when no command line parameters are given.
This allow you to run the miner without batch/script.


>>> API and Monitoring <<<

With the -b parameter you can open your ccminer to your network, use -b 0.0.0.0:4068 if required.
On windows, setting 0.0.0.0 will ask firewall permissions on the first launch. Its normal.

Default API feature is only enabled for localhost queries by default, on port 4068.

You can test this api on linux with "telnet <miner-ip> 4068" and type "help" to list the commands.
Default api format is delimited text. If required a php json wrapper is present in api/ folder.

>>> Additional Notes <<<

This code should be running on nVidia GPUs of compute capability 5.0 and beyond.
Support for lower than compute 5.0 devices has been dropped so we can more efficiently implement new algorithms using the latest
hardware features.

>>> RELEASE HISTORY <<<

  Oct 28th 2016  Release version alexis-1.0 - source code developed under CUDA7.5 for compute 5.0 and 5.2
                 
                 Changed the output in order to display how many blocks have been solved by the miner
                 [S/A/T] stands for [Solved blocks / Accepted (shares) / Total (shares)]
                                  
                 Display the hardware state (temperatures, fan percentage, core/memory clocks, wattage)
                 by implementing hardware monitoring threads which are sampling the hardware through
                 NVML concurrently with the scanhash functions execution in order to fetch as accurate
                 as possible results.
                 Hardware sampling happens once every 5 minutes. An example output is shown bellow:
                 
                 [2016-10-22 04:20:26] GPU#0:ASUS GTX 970, 6530.09kH/s
                 [2016-10-22 04:20:26] GPU#0:ASUS GTX 970, 0.045MH/W, 0.0053MH/Mhz
                 [2016-10-22 04:20:26] GPU#0:ASUS GTX 970, 67C(F:44%) 1235/3004MHz(145W)
                 
                 API functionality was not changed, the returned results are the values of that moment.
                 ------------------------------------------------------------------------------------
                 
                 All data bellow are from testing the source code under Linux, compiled with gcc 4.8.4
                 and the CUDA7.5 toolkit.
                 
                 The gpus used for tuning/testing the source are:
                 ASUS DIRECTCUII GTX970 OC running at 1290MHz maximum core clock with 163W maximum TDP
                 GB Windforce  GTX750Ti OC running at 1320MHz maximum core clock with  46W maximum TDP
                 (The latter was kindly donated from Mr. Tanguy)
                 
                 Unless stated otherwise, the throughput increases bellow refer to the GTX970.
                 
                 +2.72% and +1.1% on Decred/Saffron.
                 Using the hybrid approach and storing the most frequently accessed precomputed 
                 values in registers yields a further +2.72% improvement in decred's hashing 
                 function and +1.1% in 14-round blake.
                 ------------------------------------------------------------------------------------

                 +12.5% Keccak.
                 ------------------------------------------------------------------------------------
                 
                 +120% blake2s implementation.
                 There are 2 hashing functions implemented in order to decrease the number of
                 operations in the final block when difficulty is set to 1 or higher.
                 ------------------------------------------------------------------------------------
                 
                 + 8% on skein  on compute5.2 devices.
                 +26% on skein  on compute5.0 devices.
                 +40% on skein2 on compute5.2 devices.
                 +27% on skein2 on compute5.0 devices.
                 Skein hashing function consists of ~2000 operations, from which almost half are
                 64-bit additions. Deepening the precomputations, at least 56 of these operations
                 can be computed once, yielding a 2.8% reduction in the instructions used.
                 Carefully increasing the register usage and tuning the threads per block parameter
                 leads to the above throughput increases. Moreover, on skein2, as noted from skein,
                 merging the hash functions allows the gpu to better utilize it's computational resources.
                 ------------------------------------------------------------------------------------
                 
                 +3.15% lyra2v2, tuned under CUDA7.5, in order to reach CUDA6.5 reported throughput.
                 ------------------------------------------------------------------------------------
                 
                 +21% Lyra2re implementation
                 By reducing the code size of the main bottleneck function.
                 Applied partial shared memory utilization on groestl256 hashing function (see qubit)
                 ------------------------------------------------------------------------------------
                 
                 +95% Whirlpool implementation
                 Applied the same precomputations with whirlpoolx implementation in order to decrease
                 shared memory's bank conflicts.
                 Applied partial shared memory utilization (see qubit)
                 ------------------------------------------------------------------------------------
                 
                 +10% and 13.2% nist5/quark
                 ------------------------------------------------------------------------------------
                 
                 +43.5% Qubit implementation / +21% on 750ti
                 
                 Qubit is a sequential appliance of:
                 luffa(5.1%),cubehash(13.4%),shavite(19.6%),simd(28.1%) and echo(33.7%)
                 Shavite512 and Echo512 are both utilizing shared memory in order to perform random
                 table lookups and therefore being bounded by the bank conflicts which occur, making them
                 responsible for 53.3% of the total execution time while keeping the gpu under-utilized.
                 Since there is no way to precompute any part of these values, a different method needs
                 to be applied. A combination of texture memory and shared memory utilization, which was
                 applied to Kepler gpus, was found by previous developers to not be efficient in terms 
                 of hardware utilization in latest architectures. However, a combination of shared memory
                 accesses and __ldg() loads, cached in the texture cache, was found to significantly 
                 increase the performance.
                 Moreover, warps are executing instructions from the instruction cache independently to
                 each other, unless otherwise specified with synchronization routines. Therefore,
                 merging a memory dependent hashing function, such as simd_compress, with a shared memory
                 dependent hashing function such as echo, allows the warps to propagate through the 
                 instruction path creating an - in warp level - overlap which allows a far better gpu 
                 utilization.
                 Applying the above methods to the rest of the hashing functions the bellow increases
                 were noticed:
                 
                 +28% increase in X11/C11 performance / +21% on 750ti
                 
                 +20% increase in x11evo performance (without warp-level overlap) / +21% on 750ti
                 
                 +27% increase in X13 performance / +21% on 750ti
                 
                 +29% increase in X14 performance / +24% on 750ti
                 
                 +33% increase in X15 performance / +30% on 750ti
                 
                 +32% increase in X17 performance / +28% on 750ti

                 +332% and 835% increase in sib/veltor performance
                 By utilizing shared memory for the look up tables and applying the partial shared
                 memory utilization on streebog hashing function.
                 ------------------------------------------------------------------------------------
                 
                 +3.5% LBRY in order to partially reach CUDA8.0 throughput when compiled under CUDA7.5
                 ------------------------------------------------------------------------------------

                 +4.6% neoscrypt / +7% on 750Ti
                 Used video SIMD instruction for cumulative sum of bufidx
                 Replaced 64bit shifts with byte permutations, since shifting is always 0,8,16 or 24
                 Performed more precomputations. 
                 (On high power consumption cards there seemed to be a decrease in hashrate)
                 ------------------------------------------------------------------------------------
                 
                 +6% on SIA
                 Mostly precomputations and final block reduction of operations
                 ------------------------------------------------------------------------------------
                 
                 +8.2% on myr-gr / +8.45% on 750ti
                 Using groestl512 of quark and replacement of the sha256 hashing function with
                 the one used on LBRY.
                 ------------------------------------------------------------------------------------
                 
  Apr 11th 2016  Release faster decred implementation.
                 From the previously mentioned methods, it was found that passing the precomputed 
                 values in constant memory can yield 1% increase from storing them in registers.

  Apr 8th 2016   Release 10% faster 14-rounds blake
                 Following the exact same approach with 8-rounds blake leads to increased performance,
                 despite the fact of increased register pressure and lower occupancy per SM. However,
                 a hybrid approach is applicable and will potentially allow a better utilization.

  Mar 21st 2016  Fixed a CUDA7.5 issue for 8-rounds blake which was not allowing the compiler to store
                 the precomputed values in registers. Set CUDA7.5 as the default toolkit for future 
                 development.

  Feb 13th 2016  Release 15% faster 8-rounds blake under CUDA6.5 toolkit.
                 8-rounds blake hashing function is a simple and computational bounded function.
                 Each core block consists of 14 operations. Since a big part of the messsage is 
                 held constant, most of the xors of the message values with the constant table
                 can be precomputed and used during the execution of the hashing function, reducing
                 each block's operations by up to 14.28%. Moreover, 8-round blake is not a register
                 intensive function. Therefore, in contrast with the whirlpoolx implementation, 
                 we can store these -reusable through the core blocks - precomputed values in registers
                 in order to eliminate the constant memory loading times throughout the execution.
                 This is achieved by utilizing each unique thread to compute multiple hashes 
                 in it's lifespan. A healthy combination of threads per block and hashes per thread
                 can yield the best throughput.

  Jan 31st 2016  Update to 1.7.1 version from Tanguy Pruvot repository

  Mar 7th 2015   Initial Release with 30% faster whirlpoolx.
                 The current implementation of whirlpool hashing function rely heavily in Table Lookups.
                 Since the look ups are dependent on the state of each thread, a random access pattern 
                 occurs, disallowing the use of the device's global memory as a memory to host the LUTs.
                 In order to improve read time, LUTs are stored in shared memory, which is the fastest
                 of device's memories. However, since each thread inside a warp is reading from a random
                 location of the LUTs, the execution time heavily relys on the bank conflicts which emerge.
                 In order to reduce bank conflicts and thus the serialized loads inside the warp, a deeper
                 precomputation of the round keys can emerge, by taking advantage of the fact that a big
                 part of the input is held constant. The precomputed round keys are stored in constant 
                 memory relieving the hashing function from repetitive logical operations and redundant 
                 accesses to shared memory.

>>> AUTHORS <<<

Notable contributors to this application are:

Christian Buchner, Christian H. (Germany): Initial CUDA implementation

djm34, tsiv, sp and klausT for cuda algos implementation and optimisation

Tanguy Pruvot : 750Ti tuning, blake, colors, zr5, skein, general code cleanup
                API monitoring, linux Config/Makefile and vstudio libs...

and also many thanks to anyone else who contributed to the original
cpuminer application (Jeff Garzik, pooler)

Source code is included to satisfy GNU GPL V3 requirements.


With kind regards,

   Christian Buchner ( Christian.Buchner@gmail.com )
   Christian H. ( Chris84 )
   Tanguy Pruvot ( tpruvot@github )
