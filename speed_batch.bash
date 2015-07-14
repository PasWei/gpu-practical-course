#!/bin/bash
for ((i = 100; i <= 1000; i += 100));
do
	echo "cpu - gpu: $i";
	./assignment -H $i -b 600 -e 1 -t BACKPROP_BATCH_CPU -i ./data/t10k-images-idx3-ubyte -l ./data/t10k-labels-idx1-ubyte | grep 'The task took' | tee -a ./speed/BACKPROP_BATCH_CPU.txt;
	./assignment -H $i -b 600 -e 1 -t BACKPROP_BATCH_GPU -i ./data/t10k-images-idx3-ubyte -l ./data/t10k-labels-idx1-ubyte | grep 'The task took' | tee -a ./speed/BACKPROP_BATCH_GPU.txt;
done

for ((i = 100; i <= 1000; i += 100));
do
	echo "cpu - gpu (2): $i";
	./assignment -H 1000 -H $i -b 600 -e 1 -t BACKPROP_BATCH_CPU -i ./data/t10k-images-idx3-ubyte -l ./data/t10k-labels-idx1-ubyte | grep 'The task took' | tee -a ./speed/BACKPROP_BATCH_CPU_2.txt;
	./assignment -H 1000 -H $i -b 600 -e 1 -t BACKPROP_BATCH_GPU -i ./data/t10k-images-idx3-ubyte -l ./data/t10k-labels-idx1-ubyte | grep 'The task took' | tee -a ./speed/BACKPROP_BATCH_GPU_2.txt;
done

#for i in `seq 1 20`;
#do
#	echo "cpu - gpu:";
#	./assignment -H 1000 -H $((i * 50)) -e 1 -t BACKPROP_STOCH_CPU | grep 'The task took' | tee -a ./speed/BACKPROP_STOCH_CPU2.txt;
#	./assignment -H 1000 -H $((i * 50)) -e 1 -t BACKPROP_STOCH_GPU | grep 'The task took' | tee -a ./speed/BACKPROP_STOCH_GPU2.txt;
#done
#grep -o '[0-9]\{1,\}\.[0-9]*'
