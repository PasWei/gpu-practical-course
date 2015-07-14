#!/bin/bash
for i in `seq 7 20`;
do
	echo "cpu - gpu:";
	./assignment -H $((i * 50)) -e 1 -t BACKPROP_STOCH_CPU | grep 'The task took' | tee -a ./speed/BACKPROP_STOCH_CPU1.txt;
	./assignment -H $((i * 50)) -e 1 -t BACKPROP_STOCH_GPU | grep 'The task took' | tee -a ./speed/BACKPROP_STOCH_GPU1.txt;
done
for i in `seq 1 20`;
do
	echo "cpu - gpu:";
	./assignment -H 1000 -H $((i * 50)) -e 1 -t BACKPROP_STOCH_CPU | grep 'The task took' | tee -a ./speed/BACKPROP_STOCH_CPU2.txt;
	./assignment -H 1000 -H $((i * 50)) -e 1 -t BACKPROP_STOCH_GPU | grep 'The task took' | tee -a ./speed/BACKPROP_STOCH_GPU2.txt;
done
#grep -o '[0-9]\{1,\}\.[0-9]*'
