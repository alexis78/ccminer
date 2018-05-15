#!/bin/bash
#
#USE THIS STARTUP SCRIPT TO AUTOMATICALLY MINE THE MOST PROFITABLE COIN IN ZPOOL UNDER LINUX

while :
do

./ccminer -r 0 -a quark --url  stratum+tcp://mine.zpool.ca:4033 -u 13VDy6uNHnoChvR8mW68KJqorBVG93eK5p -p quark,skein,lyra2v2,nist5 --show-diff
./ccminer -r 0 -a skein --url  stratum+tcp://mine.zpool.ca:4933 -u 13VDy6uNHnoChvR8mW68KJqorBVG93eK5p -p quark,skein,lyra2v2,nist5 --show-diff -d 0,1 -i 28,26
./ccminer -r 0 -a lyra2v2 --url  stratum+tcp://mine.zpool.ca:4533 -u 13VDy6uNHnoChvR8mW68KJqorBVG93eK5p -p quark,skein,lyra2v2,nist5 --show-diff -i 21
./ccminer -r 0 -a nist5 --url  stratum+tcp://mine.zpool.ca:3833 -u 13VDy6uNHnoChvR8mW68KJqorBVG93eK5p -p quark,skein,lyra2v2,nist5 --show-diff -d 0,1 -i 21,19

sleep 5
done
