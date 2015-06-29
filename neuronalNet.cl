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
	
	//TODO: caching here!

	//the actual computation
	if (neuronNumber < numNeurons) {
		//the output
		float output = 0.0;
		//iterate over all neurons
		for (int i = 0; i < inputSize; i++) {
			output += weightBuffer[i * numNeurons + neuronNumber] *
				inputBuffer[i + inputBufferOffset + inputVectorNumber * inputSize];
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

	//try without input cache first TODO: use caching!
	
}

