//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock
__kernel void feedForward(
	const __global float* weightBuffer,
	const __global float* inputBuffer,
	__global float* outputBuffer,
	const int inputSize,
	const int numNeurons,
	const int threadsPerInputVector,
	const int inputBufferOffset,
	const int useActivationFunction,
	__local float* inputCache
	)
{
	//get IDs
	uint GID = get_global_id(0);
	uint LID = get_local_id(0);

	//for wich input vector does the thread work?
	uint inputVectorNumber = GID / threadsPerInputVector;
	uint neuronNumber = GID - threadsPerInputVector * inputVectorNumber;

	//get the information for addressing the input vector and cache
	uint localWorkgroupSize = get_local_size(0);
	uint stillToRead = inputSize;
	uint cacheOffset = 0;
	
	//read the input vector into the cache
	while (localWorkgroupSize <= stillToRead) {
		inputCache[cacheOffset + LID] = inputBuffer[LID + cacheOffset + inputBufferOffset +
			inputVectorNumber * inputSize];
		cacheOffset = cacheOffset + localWorkgroupSize;
		stillToRead = stillToRead - localWorkgroupSize;
	}

	if (LID < stillToRead) {
		inputCache[cacheOffset + LID] = inputBuffer[LID + cacheOffset + inputBufferOffset +
			inputVectorNumber * inputSize];
	} 

	//wait for the threads to fill the cache
	barrier(CLK_LOCAL_MEM_FENCE);

	//the actual computation
	if (neuronNumber < numNeurons) {
		//the output
		float output = 0.0;
		//iterate over all neurons
		for (int i = 0; i < inputSize; i++) {
			output += weightBuffer[i * numNeurons + neuronNumber] * inputCache[i];
				/*inputBuffer[i + inputBufferOffset + inputVectorNumber * inputSize];*/
		}
		//DONT forget the constant 1
		output += weightBuffer[inputSize * numNeurons + neuronNumber];

		if (useActivationFunction == 1) {
			output = 1.0f / (1.0f + exp(-output));
		}
		outputBuffer[inputVectorNumber * numNeurons + neuronNumber] = output;
	}
		
	/*outputBuffer[5] = (float) inputSize;
	outputBuffer[6] = (float) numNeurons;
	outputBuffer[7] = (float) threadsPerInputVector;
	outputBuffer[8] = (float) inputBufferOffset;
	outputBuffer[9] = (float) useActivationFunction;*/	
}

//this kernel assumes that the size of the input vector is smaller than the work group size
__kernel void softMax(__global float* outputBuffer, const uint numNeurons, __local float* sumCache) {
	//get IDs
	uint GID = get_global_id(0);
	uint LID = get_local_id(0);

	uint inputVectorNumber = get_group_id(0);

	//copy the values to local memory and exp() them
	if (LID < numNeurons) {
		sumCache[LID] = exp(outputBuffer[LID + inputVectorNumber * numNeurons]);
	}
	 
	//wait for all threads 
	barrier(CLK_LOCAL_MEM_FENCE);

	uint leftOver = numNeurons % 2;
	uint numThreads = numNeurons / 2;
	
	//add two values
	while (numThreads >= 1) {
		if (LID < numThreads-1) {
			sumCache[LID] = sumCache[2 * LID] + sumCache[2 * LID + 1]; 
		}
		if (LID == numThreads-1) {
			if (leftOver == 1) {
				sumCache[LID] = sumCache[2 * LID] + sumCache[2 * LID + 1] + sumCache[2 * LID + 2]; 
			} else {
				sumCache[LID] = sumCache[2 * LID] + sumCache[2 * LID + 1];
			}
		}
		
		leftOver = numThreads % 2;
		numThreads = numThreads / 2;		

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//the sum is stored in sumCache[0]
	if (LID < numNeurons) {
		outputBuffer[LID + inputVectorNumber * numNeurons] =
			exp(outputBuffer[LID + inputVectorNumber * numNeurons]) / sumCache[0];
	}

}

//this kernel writes a zero to every position in the buffer
__kernel void zeroBuffer(__global float* buffer, uint len) {
	
	uint GID = get_global_id(0);

	if (GID < len) { 
		buffer[GID] = 0.0f;
	}

}

__kernel void gradientDescentOutputLayer(
	const __global float* labelBuffer,
	const __global float* inputBuffer,
	const __global float* neuronalNetworkResultBuffer,
	__global float* deltaUpdateBuffer,
	const uint labelBufferOffset,
	const uint numberOfNeurons,
	const uint threadsPerInputVector,
	const uint numberOfInputs,
	__local float* inputCache
	) 
{
	//get IDs
	uint GID = get_global_id(0);
	uint LID = get_local_id(0);

	//for wich input vector does the thread work?
	uint inputVectorNumber = GID / threadsPerInputVector;
	uint neuronNumber = GID - threadsPerInputVector * inputVectorNumber;

	//get the information for addressing the input vector and cache
	uint localWorkgroupSize = get_local_size(0);
	uint stillToRead = numberOfInputs;
	uint cacheOffset = 0;

	//read the input vector into the cache
	while (localWorkgroupSize <= stillToRead) {
		inputCache[cacheOffset + LID] = inputBuffer[LID + cacheOffset +
			inputVectorNumber * numberOfInputs];
		cacheOffset = cacheOffset + localWorkgroupSize;
		stillToRead = stillToRead - localWorkgroupSize;
	}

	if (LID < stillToRead) {
		inputCache[cacheOffset + LID] = inputBuffer[LID + cacheOffset +
			inputVectorNumber * numberOfInputs];
	} 

	//wait for the threads to fill the cache
	barrier(CLK_LOCAL_MEM_FENCE);

	//calculate the deltas for the deltaUpdates buffer
	if (neuronNumber < numberOfNeurons) {
		//calculate the delta value
		float delta = labelBuffer[labelBufferOffset + numberOfNeurons * inputVectorNumber + neuronNumber] - 
			neuronalNetworkResultBuffer[numberOfNeurons * inputVectorNumber + neuronNumber];
		
		//add deltas to the buffers
		float input;
		for (int i = 0; i < numberOfInputs + 1; i++) {
			//get the right input
			if (i == numberOfInputs) {
				input = 1.0f;
			} else {
				input = inputCache[i];
			}
			//write changes to the delta buffer
			deltaUpdateBuffer[i * numberOfNeurons + neuronNumber] += delta * input;
		}
	}
}

__kernel void gradientDescentHiddenLayer(
	const __global float* weightBuffer,
	const __global float* neuronInputBuffer,
	const __global float* neuronOutputBuffer,
	__global float* deltaUpdateBuffer,
	const uint inputSize,
	const uint numberOfNeurons,
	const uint threadsPerInputVector,
	const uint inputBufferOffset,
	__local float* inputCache
	)
{

}
