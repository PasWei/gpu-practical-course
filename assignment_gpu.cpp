#include <string> //for string
#include <iostream> //for cout, endl
#include <utility> //for swap (since c++11)

#include "CLUtil.h"
#include "assignment.h"

#define PRINT_INFO(title, buffer, bufferSize, maxBufferSize, expr) { expr; buffer[bufferSize] = '\0'; std::cout << title << ": " << buffer << std::endl; }

bool Assignment::InitCLResources() {
	
	std::cout << "InitCLResources(): Initialize the opencl buffers on the device" << std::endl; 

	//clCreateBuffer: context, flags, size, *host_ptr, *error

	cl_int clError;

	//training data
	this->d_trainingInputBuffer = clCreateBuffer(
		this->h_CLContext,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * this->trainingData->numberOfSamples * this->trainingData->numberOfInputs,
		this->trainingInputBuffer,
		&clError
	);
	V_RETURN_FALSE_CL(clError, "Error allocating device buffer d_trainingInputBuffer");

	this->d_trainingLabelBuffer = clCreateBuffer(
		this->h_CLContext,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * this->trainingData->numberOfSamples * this->trainingData->numberOfOutputs,
		this->trainingLabelBuffer,
		&clError
	);
	V_RETURN_FALSE_CL(clError, "Error allocating device buffer d_trainingLabelBuffer");

	//weight buffers and delta update buffers
	for (unsigned int i = 0; i < this->sizeOfWeightBuffer.size(); i++) {
		this->d_weightBuffers.push_back(
			clCreateBuffer(
				this->h_CLContext,
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				sizeof(float) * this->sizeOfWeightBuffer[i],
				this->h_weightBuffers[i],
				&clError
			)
		);
		V_RETURN_FALSE_CL(clError, "Error allocating device buffer d_weightBuffers[]"); 
		this->d_deltaUpdates.push_back(
			clCreateBuffer(
				this->h_CLContext,
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				sizeof(float) * this->sizeOfWeightBuffer[i],
				this->h_weightBuffers[i],
				&clError
			)
		);
		V_RETURN_FALSE_CL(clError, "Error allocating device buffer d_deltaUpdates[]"); 	
	}

	//partial result buffers and delta update buffers
	for (unsigned int i = 0; i < this->hiddenLayers.size(); i++) {
		//weight buffer
		this->d_partialResults.push_back(
			clCreateBuffer(
				this->h_CLContext,
				CL_MEM_READ_WRITE,
				sizeof(float) * this->hiddenLayers[i] * this->parallelBackpropagationSize,
				NULL,
				&clError
			)
		);
		V_RETURN_FALSE_CL(clError, "Error allocating device buffer d_partialResults[]");
	}
	//output layer weight buffer
	this->d_partialResults.push_back(
		clCreateBuffer(
			this->h_CLContext,
			CL_MEM_READ_WRITE,
			sizeof(float) * this->trainingData->numberOfOutputs * this->parallelBackpropagationSize,
			NULL,
			&clError
		)
	);
	V_RETURN_FALSE_CL(clError, "Error allocating device buffer d_partialResults[]");

	//load and compile kernels
	std::string programCode;
	//size_t programSize = 0;

	CLUtil::LoadProgramSourceToMemory("neuronalNet.cl", programCode);
	this->h_Program = CLUtil::BuildCLProgramFromMemory(this->h_CLDevice, this->h_CLContext, programCode);
	if(this->h_Program == nullptr) return false;

	//create kernels
	h_feedForwardKernel = clCreateKernel(this->h_Program, "feedForward", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: feedForward.");

	h_softMaxKernel = clCreateKernel(this->h_Program, "softMax", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: softMax.");

	h_zeroBufferKernel = clCreateKernel(this->h_Program, "zeroBuffer", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: zeroBuffer.");

	h_gradientDescentOutputLayerKernel = clCreateKernel(this->h_Program, "gradientDescentOutputLayer", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: gradientDescentOutputLayer.");

	h_gradientDescentHiddenLayerKernel = clCreateKernel(this->h_Program, "gradientDescentHiddenLayer", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: gradientDescentHiddenLayer.");

	//set kernel arguments: cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value

	return true;
}

void Assignment::zeroDeltaBuffersGPU() {
	
	cl_int clError;

	//the length of the delta buffer
	int bufferLength;

	for (unsigned int i = 0; i < this->d_deltaUpdates.size(); i++) {
		//set arguments
		//Argument 0: the buffer to write zeroes into
		clError = clSetKernelArg(h_zeroBufferKernel, 0, sizeof(cl_mem), (void*)&this->d_deltaUpdates[i]);
		
		//Argument 1: buffer length
		bufferLength = this->sizeOfWeightBuffer[i];
		clError |= clSetKernelArg(h_zeroBufferKernel, 1, sizeof(cl_int), (void*)&bufferLength);
		V_RETURN_CL(clError, "Failed to set kernel args: zeroBufferKernel");

		//calculate local and global work size
		size_t LocalWorkSize[3] = {(size_t)this->localGroupSize, 1, 1};
		int numWorkGroups = (bufferLength / this->localGroupSize) + 1;
		size_t GlobalWorkSize = numWorkGroups * this->localGroupSize;

		//std::cout << "GlobalWorkSize: " << GlobalWorkSize << std::endl;
		//std::cout << "bufferLength: " << bufferLength << std::endl;

		//launch the kernel
		clError = clEnqueueNDRangeKernel(
			this->h_CLCommandQueue, 
			this->h_zeroBufferKernel, 
			1, 
			NULL, 
			&GlobalWorkSize, 
			LocalWorkSize, 
			0, 
			NULL, 
			NULL
		);
		V_RETURN_CL(clError, "Error executing zeroBufferKernel!");

		//read back the buffer to test if its 0:
		/*float* tmpBuff = new float[bufferLength];

		V_RETURN_CL(
			clEnqueueReadBuffer(
				this->h_CLCommandQueue,
				this->d_deltaUpdates[i],
				CL_TRUE,
				0,
				bufferLength * sizeof(float),
				tmpBuff,
				0,
				NULL,
				NULL
			), 
			"Error reading data from device!"
		);

		//print it
		std::cout << "The buffer " << i << " should be zeroed: ";  
		for (int j = 0; j < bufferLength; j++) {
			std::cout << tmpBuff[j] << " ";
		}
		std::cout << std::endl;
		
		//delete the tmp buffer
		delete[] tmpBuff;*/
	}
}

void Assignment::gradientDescentGPU(unsigned int indexOfInput, unsigned int numInputVectors) {

	//sanity check
	if (numInputVectors > this->parallelBackpropagationSize) {
		std::cout << "buffer for gradinetDescent too small!" << std::endl;
		return;
	}

	//error handling for openCL
	cl_int clError;

	//compute index in label buffer
	int labelIndex = indexOfInput * this->trainingData->numberOfOutputs;
	int inputIndex = this->trainingData->numberOfInputs * indexOfInput;	

	//the output layer first
	//Argument 0: the label buffer
	clError = clSetKernelArg(
		h_gradientDescentOutputLayerKernel, 0, sizeof(cl_mem), (void*)&this->d_trainingLabelBuffer);
	//Argument 1: the input buffer for the output layer
	clError |= clSetKernelArg(
			h_gradientDescentOutputLayerKernel,
			1,
			sizeof(cl_mem),
			(void*)&this->d_partialResults[this->d_partialResults.size()-2]
		);
	//Argument 2: the result buffer
	clError |= clSetKernelArg(
		h_gradientDescentOutputLayerKernel, 2, sizeof(cl_mem), (void*)&this->d_partialResults.back());		
	//Argument 3: the delta update buffer
	clError |= clSetKernelArg(
		h_gradientDescentOutputLayerKernel, 3, sizeof(cl_mem), (void*)&this->d_deltaUpdates.back());
	//Argument 4: the label buffer offset
	clError |= clSetKernelArg(
		h_gradientDescentOutputLayerKernel, 4, sizeof(cl_int), (void*)&labelIndex);
	//Argument 5: the number of neurons
	clError |= clSetKernelArg(
		h_gradientDescentOutputLayerKernel, 5, sizeof(cl_int), (void*)&this->trainingData->numberOfOutputs);
	//Argument 6: the the number of threads per input vector
	int threadsPerInputVector = this->trainingData->numberOfOutputs/this->localGroupSize;
		if (this->trainingData->numberOfOutputs % this->localGroupSize != 0) {
			threadsPerInputVector++;
		}
	//inputSize/this->localGroupSize is always a multiple of localGroupSize
	threadsPerInputVector = threadsPerInputVector*this->localGroupSize;
	
	//std::cout << "threadsPerInputVector: " << threadsPerInputVector << std::endl;

	clError |= clSetKernelArg(
		h_gradientDescentOutputLayerKernel, 6, sizeof(cl_int), (void*)&threadsPerInputVector);
	//Argument 7: the size of the input vector
	int inputSize = hiddenLayers.back();
	clError |= clSetKernelArg(
		h_gradientDescentOutputLayerKernel, 7, sizeof(cl_int), (void*)&inputSize);
	//Argument 8: the input cache
		clError |= clSetKernelArg(h_gradientDescentOutputLayerKernel, 8, inputSize * sizeof(cl_float), NULL);
	V_RETURN_CL(clError, "Failed to set kernel args: gradientDescentOutputLayerKernel");

	//calculate local and global work size
	size_t LocalWorkSize[3] = {(size_t)this->localGroupSize, 1, 1};
	size_t GlobalWorkSize = threadsPerInputVector * numInputVectors;

	//launch the kernel
	clError = clEnqueueNDRangeKernel(
		this->h_CLCommandQueue, 
		this->h_gradientDescentOutputLayerKernel, 
		1, 
		NULL, 
		&GlobalWorkSize, 
		LocalWorkSize, 
		0, 
		NULL, 
		NULL
	);
	V_RETURN_CL(clError, "Error executing gradientDescentOutputLayerKernel!");

	//now the hideen layers
	for (int i = this->hiddenLayers.size()-1; i >= 0; i--) {
		//Argument 0: the weight buffer of the layer above
		clError = clSetKernelArg(
			h_gradientDescentHiddenLayerKernel, 0, sizeof(cl_mem), (void*)&this->d_weightBuffers[i+1]);
		//Argument 1: the input buffer
		if (i == 0) {
			clError |= clSetKernelArg(
				h_gradientDescentHiddenLayerKernel, 1, sizeof(cl_mem), (void*)&this->d_trainingInputBuffer);
		} else {
			clError |= clSetKernelArg(
				h_gradientDescentHiddenLayerKernel, 1, sizeof(cl_mem), (void*)&this->d_partialResults[i-1]);
		}
		//Argument 2: the output buffer of this layer
		clError |= clSetKernelArg(
				h_gradientDescentHiddenLayerKernel, 2, sizeof(cl_mem), (void*)&this->d_partialResults[i]);
		//Argument 3: the delta buffer of the layer above
		clError = clSetKernelArg(
			h_gradientDescentHiddenLayerKernel, 3, sizeof(cl_mem), (void*)&this->d_deltaUpdates[i+1]);
		//Argument 4: the delta buffer of this layer
		clError = clSetKernelArg(
			h_gradientDescentHiddenLayerKernel, 4, sizeof(cl_mem), (void*)&this->d_deltaUpdates[i]);
		//Argument 5: the number of Inputs
		int inputSize;
		int inputBufferOffset;
		if (i == 0) {
			inputSize = this->trainingData->numberOfInputs;
			inputBufferOffset = inputIndex;
		} else {
			inputSize = this->hiddenLayers[i-1];
			inputBufferOffset = 0;
		}
		clError |= clSetKernelArg(
			h_gradientDescentHiddenLayerKernel, 5, sizeof(cl_int), (void*)&inputSize);
		//Argument 6: the number of neurons in this layer
		clError |= clSetKernelArg(
			h_gradientDescentHiddenLayerKernel, 6, sizeof(cl_int), (void*)&this->hiddenLayers[i]);
		//Argument 7: the number of Neurons in the next layer
		int numNeuronsHigherLayer;
		if (i == (int) this->hiddenLayers.size()-1) {
			numNeuronsHigherLayer = this->trainingData->numberOfOutputs;
		} else {
			numNeuronsHigherLayer = this->hiddenLayers[i+1];
		}
		clError |= clSetKernelArg(
			h_gradientDescentHiddenLayerKernel, 7, sizeof(cl_int), (void*)&numNeuronsHigherLayer);
		//Argument 8: the number of threads per input vector
		int threadsPerInputVector = this->hiddenLayers[i]/this->localGroupSize;
		if (this->hiddenLayers[i] % this->localGroupSize != 0) {
			threadsPerInputVector++;
		}
		//inputSize/this->localGroupSize is always a multiple of localGroupSize
		threadsPerInputVector = threadsPerInputVector*this->localGroupSize;
		clError |= clSetKernelArg(
			h_gradientDescentHiddenLayerKernel, 8, sizeof(cl_int), (void*)&threadsPerInputVector);
		//Argument 9: offset into the input buffer
		clError |= clSetKernelArg(
			h_gradientDescentHiddenLayerKernel, 9, sizeof(cl_int), (void*)&inputBufferOffset);
		//Argument 10: the input cache
		clError |= clSetKernelArg(h_gradientDescentHiddenLayerKernel, 10, inputSize * sizeof(cl_float), NULL);
		V_RETURN_CL(clError, "Failed to set kernel args: gradientDescentOutputLayerKernel");

		//calculate local and global work size
		size_t LocalWorkSize[3] = {(size_t)this->localGroupSize, 1, 1};
		size_t GlobalWorkSize = threadsPerInputVector * numInputVectors;

		//launch the kernel
		clError = clEnqueueNDRangeKernel(
			this->h_CLCommandQueue, 
			this->h_gradientDescentHiddenLayerKernel, 
			1, 
			NULL, 
			&GlobalWorkSize, 
			LocalWorkSize, 
			0, 
			NULL, 
			NULL
		);
		V_RETURN_CL(clError, "Error executing gradientDescentOutputLayerKernel!");
	}
}

void Assignment::compareDeltaBuffers() {
	for (unsigned int i = 0; i < this->h_deltaUpdates.size(); i++) {
		//temp buffer for gpu output
		float* tmpBuff = new float[this->sizeOfWeightBuffer[i]];
		
		V_RETURN_CL(
			clEnqueueReadBuffer(
				this->h_CLCommandQueue,
				this->d_deltaUpdates[i],
				CL_TRUE,
				0,
				this->sizeOfWeightBuffer[i] * sizeof(cl_float),
				tmpBuff,
				0,
				NULL,
				NULL
			), 
			"Error reading data from device!"
		);

		//compare the buffers
		std::cout << "differences of delta buffers in layer " << i << ": ";
		for (int j = 0; j < this->sizeOfWeightBuffer[i]; j++) {
			std::cout << tmpBuff[j] - this->h_deltaUpdates[i][j] /*<< " (" << tmpBuff[j] << "-" <<
			this->h_deltaUpdates[i][j] << ")" */<< " ";
		}
		std::cout << std::endl;

		delete[] tmpBuff;
	}
}

void Assignment::printDeltaBufferOutputLayerGPU() {
	//get buffer of right size:
	float* tmpBuff = new float[this->sizeOfWeightBuffer.back()];

	//read back
	V_RETURN_CL(
		clEnqueueReadBuffer(
			this->h_CLCommandQueue,
			this->d_deltaUpdates.back(),
			CL_TRUE,
			0,
			this->sizeOfWeightBuffer.back() * sizeof(cl_float),
			tmpBuff,
			0,
			NULL,
			NULL
		), 
		"Error reading data from device!"
	);

	std::cout << "Deltas of output Layer (GPU): ";
	for (int i = 0; i < this->sizeOfWeightBuffer.back(); i++) {
		std::cout << tmpBuff[i] << " ";
	}
	std::cout << std::endl;

	//delte buffer
	delete[] tmpBuff;
}

void Assignment::feedForwardGPU(unsigned int indexOfInput,  unsigned int numInputVectors) {
	
	if (numInputVectors > this->parallelBackpropagationSize) {
		std::cout << "buffer for feedForward too small!" << std::endl;
		return;
	}

	int inputIndex = this->trainingData->numberOfInputs * indexOfInput;
	//int labelIndex = indexOfInput * this->trainingData->numberOfOutputs;

	cl_int clError;

	//determine the number of neurons
	int numNeurons;

	for (unsigned int i = 0; i < this->d_partialResults.size(); i++) {
		//determine the number of inputs
		int inputSize;
		//offset in case the input buffer is needed
		int inputOffset = 0;
		//Argument 0: the weight buffer
		clError = clSetKernelArg(h_feedForwardKernel, 0, sizeof(cl_mem), (void*)&this->d_weightBuffers[i]);
		//Argument 1: the input buffer 
		//do we need the input vector buffer?
		if (i == 0) {
			//adjust input offset
			inputOffset = inputIndex;	
			//set input size
			inputSize = this->trainingData->numberOfInputs;
			//use the input buffer as input for the net
			clError |= clSetKernelArg(
				h_feedForwardKernel,
				1,
				sizeof(cl_mem),
				(void*)&this->d_trainingInputBuffer
			);
		} else {
			//set input size
			inputSize = this->hiddenLayers[i-1];
			//use the previous result as input
			clError |= clSetKernelArg(
				h_feedForwardKernel, 
				1, 
				sizeof(cl_mem), 
				(void*)&this->d_partialResults[i-1]
			);
		}
		//Argument 2: the output buffer
		clError |= clSetKernelArg(h_feedForwardKernel, 2, sizeof(cl_mem), (void*)&this->d_partialResults[i]);
		//Argument 3: input size
		clError |= clSetKernelArg(h_feedForwardKernel, 3, sizeof(cl_int), (void*)&inputSize);
		//Argument 4: number of neurons
		if (i == this->d_partialResults.size() - 1) {
			numNeurons = this->trainingData->numberOfOutputs;
		} else {
			numNeurons = this->hiddenLayers[i];
		} 
		clError |= clSetKernelArg(h_feedForwardKernel, 4, sizeof(cl_int), (void*)&numNeurons);		
		//Argument 5: threads per inputVector
		int threadsPerInputVector = /*inputSize*/numNeurons/this->localGroupSize;
		if (numNeurons % this->localGroupSize != 0) {
			threadsPerInputVector++;
		}
		//inputSize/this->localGroupSize is always a multiple of localGroupSize
		threadsPerInputVector = threadsPerInputVector*this->localGroupSize;
		clError |= clSetKernelArg(h_feedForwardKernel, 5, sizeof(cl_int), (void*)&threadsPerInputVector);
		//Argument 6: input buffer offset
		clError |= clSetKernelArg(h_feedForwardKernel, 6, sizeof(cl_int), (void*)&inputOffset);
		//Argument 7: is the activation function to be used? (one if true, 0 otherwise)
		int useActivationFunction = 1;
		if (i == this->d_partialResults.size() - 1) {
			useActivationFunction = 0;
		}
		clError |= clSetKernelArg(h_feedForwardKernel, 7, sizeof(cl_int), (void*)&useActivationFunction);
		//Argument 8: the input cache
		clError |= clSetKernelArg(h_feedForwardKernel, 8, inputSize * sizeof(cl_float), NULL);
		V_RETURN_CL(clError, "Failed to set kernel args: feedForwardKernel");

		//calculate local and global work size
		size_t LocalWorkSize[3] = {(size_t)this->localGroupSize, 1, 1};
		size_t GlobalWorkSize = threadsPerInputVector * numInputVectors;

		//launch the kernel
		clError = clEnqueueNDRangeKernel(
			this->h_CLCommandQueue, 
			this->h_feedForwardKernel, 
			1, 
			NULL, 
			&GlobalWorkSize, 
			LocalWorkSize, 
			0, 
			NULL, 
			NULL
		);
		V_RETURN_CL(clError, "Error executing feedForwardKernel!");

		//read back the result and print it for debug purposes

		/*float* tmpBuff = new float[numNeurons];		

		V_RETURN_CL(
			clEnqueueReadBuffer(
				this->h_CLCommandQueue,
				this->d_partialResults[i],
				CL_TRUE,
				0,
				numNeurons * sizeof(cl_float),
				tmpBuff,
				0,
				NULL,
				NULL
			), 
			"Error reading data from device!"
		);

		//debug output:
		std::cout << "Output of the neuronal network on GPU (layer " << i << "): ";
		for (int j = 0; j < numNeurons; j++) {
			std::cout << tmpBuff[i] << " "; 
		}
		std::cout << std::endl;

		delete[] tmpBuff;*/

	}

	//set up the softMax kernel
	//Argument 0: the output buffer
	clError = clSetKernelArg(h_softMaxKernel, 0, sizeof(cl_mem), (void*)&this->d_partialResults.back());
	//Argument 1: number of neurons
	clError |= clSetKernelArg(h_softMaxKernel, 1, sizeof(cl_int), (void*)&numNeurons);
	//Argument 2: the sum cache
	clError |= clSetKernelArg(h_softMaxKernel, 2, numNeurons * sizeof(cl_float), NULL);
	V_RETURN_CL(clError, "Failed to set kernel args: softMaxKernel");
	//calculate local and global work size
	size_t LocalWorkSize[3] = {(size_t)numNeurons, 1, 1};
	size_t GlobalWorkSize = numNeurons * numInputVectors;
	//launch the kernel
		clError = clEnqueueNDRangeKernel(
			this->h_CLCommandQueue, 
			this->h_softMaxKernel, 
			1, 
			NULL, 
			&GlobalWorkSize, 
			LocalWorkSize, 
			0, 
			NULL, 
			NULL
		);
	V_RETURN_CL(clError, "Error executing feedForwardKernel!");

	//read back the result and print it
	float* tmpBuf = new float[this->trainingData->numberOfOutputs];

	V_RETURN_CL(
		clEnqueueReadBuffer(
			this->h_CLCommandQueue,
			this->d_partialResults.back(),
			CL_TRUE,
			0 * sizeof(cl_float),
			this->trainingData->numberOfOutputs * sizeof(cl_float),
			tmpBuf,
			0,
			NULL,
			NULL
		), 
		"Error reading data from device!"
	);

	/*float crossEntropy = 0.0f;

	//compute the output
	for (unsigned int i = 0; i < this->trainingData->numberOfOutputs; i++) {
		this->h_partialResults.back()[i] = std::exp(this->h_partialResults.back()[i])/expSum;
		float target = this->trainingLabelBuffer[labelIndex + i];
		float output = this->h_partialResults.back()[i];
		crossEntropy += -1.0f * (target * std::log(output) + (1.0f - target) * std::log(1.0f - output));
	}*/

	//output the values
	std::cout << "Output of the neuronal network (GPU): ";
	for (unsigned int i = 0; i < this->trainingData->numberOfOutputs; i++) {
		std::cout << tmpBuf[i] << " "; 
	}
	std::cout << std::endl;

	delete[] tmpBuf;
}

void Assignment::ReleaseClResources() {

	//release buffers
	SAFE_RELEASE_MEMOBJECT(d_trainingInputBuffer);
	SAFE_RELEASE_MEMOBJECT(d_trainingLabelBuffer);
	for (unsigned int i = 0; i < this->d_weightBuffers.size(); i++) {
		SAFE_RELEASE_MEMOBJECT(this->d_weightBuffers[i]);
		SAFE_RELEASE_MEMOBJECT(this->d_partialResults[i]);
		SAFE_RELEASE_MEMOBJECT(this->d_deltaUpdates[i]);
	}

	//release kernels
	SAFE_RELEASE_KERNEL(this->h_feedForwardKernel);
	SAFE_RELEASE_KERNEL(this->h_softMaxKernel);
	SAFE_RELEASE_KERNEL(this->h_zeroBufferKernel);
	SAFE_RELEASE_KERNEL(this->h_gradientDescentOutputLayerKernel);
	SAFE_RELEASE_KERNEL(this->h_gradientDescentHiddenLayerKernel);

	//release program
	SAFE_RELEASE_PROGRAM(this->h_Program);
}

bool Assignment::InitCLContext() {

	std::cout << std::endl << "InitCLContext():" << std::endl;
	// 1. get all platform IDs
	std::vector<cl_platform_id> platformIds;
	const cl_uint c_MaxPlatforms = 16;
	platformIds.resize(c_MaxPlatforms);
	
	cl_uint countPlatforms;
	V_RETURN_FALSE_CL(clGetPlatformIDs(c_MaxPlatforms, &platformIds[0], &countPlatforms),
		"Failed to get CL platform ID");
	platformIds.resize(countPlatforms);

	// 2. find all available GPU devices
	std::vector<cl_device_id> deviceIds;
	const int maxDevices = 16;
	deviceIds.resize(maxDevices);
	int countAllDevices = 0;

	//look for gpus only
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

	for (size_t i = 0; i < platformIds.size(); i++)
	{
		// Getting the available devices.
		cl_uint countDevices;
		clGetDeviceIDs(platformIds[i], deviceType, 1, &deviceIds[countAllDevices], &countDevices);
		countAllDevices += countDevices;
	}
	deviceIds.resize(countAllDevices);

	if (countAllDevices == 0)
	{
		std::cout << "No device of the selected type with OpenCL support was found.";
		return false;
	}
	// Choosing the first available device.
	this->h_CLDevice = deviceIds[0];
	clGetDeviceInfo(this->h_CLDevice, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &this->h_CLPlatform, NULL);

	// Printing platform and device data.
	const int maxBufferSize = 1024;
	char buffer[maxBufferSize];
	size_t bufferSize;

	std::cout << "OpenCL platform:" << std::endl << std::endl;
	PRINT_INFO(
		"Name",
		buffer,
		bufferSize,
		maxBufferSize,
		clGetPlatformInfo(
			this->h_CLPlatform,
			CL_PLATFORM_NAME,
			maxBufferSize,
			(void*)buffer,
			&bufferSize
		)
	);

	PRINT_INFO(
		"Vendor", 
		buffer, 
		bufferSize, 
		maxBufferSize, 
		clGetPlatformInfo(
			this->h_CLPlatform, 
			CL_PLATFORM_VENDOR, 
			maxBufferSize, 
			(void*)buffer, 
			&bufferSize
		)
	);

	PRINT_INFO(
		"Version",
		buffer, 
		bufferSize, 
		maxBufferSize, 
		clGetPlatformInfo(
			this->h_CLPlatform, 
			CL_PLATFORM_VERSION, 
			maxBufferSize, 
			(void*)buffer, 
			&bufferSize
		)
	);

	PRINT_INFO(
		"Profile", 
		buffer, 
		bufferSize, 
		maxBufferSize, 
		clGetPlatformInfo(
			this->h_CLPlatform, 
			CL_PLATFORM_PROFILE, 
			maxBufferSize, 
			(void*)buffer, 
			&bufferSize
		)
	);

	std::cout << std::endl << "Device:" << std::endl << std::endl;

	PRINT_INFO(
		"Name", 
		buffer, 
		bufferSize, 
		maxBufferSize, 
		clGetDeviceInfo(
			this->h_CLDevice, 
			CL_DEVICE_NAME, 
			maxBufferSize, 
			(void*)buffer, 
			&bufferSize
		)
	);

	PRINT_INFO(
		"Vendor", 
		buffer, 
		bufferSize, 
		maxBufferSize, 
		clGetDeviceInfo(
			this->h_CLDevice, 
			CL_DEVICE_VENDOR, 
			maxBufferSize, 
			(void*)buffer, 
			&bufferSize
		)
	);

	PRINT_INFO(
		"Driver version", 
		buffer, 
		bufferSize, 
		maxBufferSize, 
		clGetDeviceInfo(
			this->h_CLDevice, 
			CL_DRIVER_VERSION, 
			maxBufferSize, 
			(void*)buffer, 
			&bufferSize
		)
	);

	cl_ulong localMemorySize;
	clGetDeviceInfo(
		this->h_CLDevice, 
		CL_DEVICE_LOCAL_MEM_SIZE, 
		sizeof(cl_ulong), 
		&localMemorySize, 
		&bufferSize
	);

	std::cout << "Local memory size: " << localMemorySize << " Byte" << std::endl;
	std::cout << std::endl << "******************************" << std::endl << std::endl;
        
	cl_int clError;

	this->h_CLContext = clCreateContext(NULL, 1, &this->h_CLDevice, NULL, NULL, &clError);	
	V_RETURN_FALSE_CL(clError, "Failed to create OpenCL context.");

	// Finally, create a command queue. All the asynchronous commands to the device will be issued
	// from the CPU into this queue. This way the host program can continue the execution
	// until some results from that device are needed.

	this->h_CLCommandQueue = clCreateCommandQueue(this->h_CLContext, this->h_CLDevice, 0, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create the command queue in the context");

	return true;
}

void Assignment::ReleaseCLContext() {
	if (this->h_CLCommandQueue != nullptr)
	{
		clReleaseCommandQueue(this->h_CLCommandQueue);
		this->h_CLCommandQueue = nullptr;
	}

	if (this->h_CLContext != nullptr)
	{
		clReleaseContext(this->h_CLContext);
		this->h_CLContext = nullptr;
	}
}
