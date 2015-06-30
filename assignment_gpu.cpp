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

	//weight buffers
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
	}

	//partial result buffers
	for (unsigned int i = 0; i < this->hiddenLayers.size(); i++) {
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

	//set kernel arguments: cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value

	return true;
}

void Assignment::feedForwardGPU(unsigned int indexOfInput, bool singleInput) {
	
	int inputIndex = this->trainingData->numberOfInputs * indexOfInput;
	int labelIndex = indexOfInput * this->trainingData->numberOfOutputs;

	cl_int clError;

	for (unsigned int i = 0; i < this->d_partialResults.size(); i++) {
		//determine the number of inputs
		int inputSize;
		//offset in case the input buffer is needed
		int inputOffset = 0;
		//determine the number of neurons
		int numNeurons;
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
		V_RETURN_CL(clError, "Failed to set kernel args: h_feedForwardKernel");

		//calculate local and global work size
		size_t LocalWorkSize[3] = {(size_t)this->localGroupSize, 1, 1};
		size_t GlobalWorkSize;
		if (singleInput) {
			GlobalWorkSize = threadsPerInputVector;
		} else {
			GlobalWorkSize = threadsPerInputVector * this->parallelBackpropagationSize;
		}

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
		V_RETURN_CL(clError, "Error executing kernel!");

		//read back the result and print it for debug purposes
		/*V_RETURN_CL(
			clEnqueueReadBuffer(
				this->h_CLCommandQueue,
				this->d_partialResults[i],
				CL_TRUE,
				0,
				numNeurons * sizeof(cl_float),
				this->h_partialResults[i],
				0,
				NULL,
				NULL
			), 
			"Error reading data from device!"
		);

		//debug output:
		std::cout << "Output of the neuronal network on GPU (layer " << i << "): ";
		for (int j = 0; j < numNeurons; j++) {
			std::cout << this->h_partialResults[i][j] << " "; 
		}
		std::cout << std::endl;*/

	}

	//read back the result and print it
	V_RETURN_CL(
		clEnqueueReadBuffer(
			this->h_CLCommandQueue,
			this->d_partialResults.back(),
			CL_TRUE,
			0/*40 * sizeof(float)*/,
			this->trainingData->numberOfOutputs * sizeof(cl_float),
			this->h_partialResults.back(),
			0,
			NULL,
			NULL
		), 
		"Error reading data from device!"
	);

	//compute the actual output (softmax activation function)
	//compute sum of exponents for softmax
	float expSum = 0.0f;
	for (unsigned int i = 0; i < this->trainingData->numberOfOutputs; i++) {
		expSum += std::exp(this->h_partialResults.back()[i]);
	}

	float crossEntropy = 0.0f;

	//compute the output
	for (unsigned int i = 0; i < this->trainingData->numberOfOutputs; i++) {
		this->h_partialResults.back()[i] = std::exp(this->h_partialResults.back()[i])/expSum;
		float target = this->trainingLabelBuffer[labelIndex + i];
		float output = this->h_partialResults.back()[i];
		crossEntropy += -1.0f * (target * std::log(output) + (1.0f - target) * std::log(1.0f - output));
	}

	//output the values
	std::cout << "Output of the neuronal network (GPU): ";
	for (unsigned int i = 0; i < this->trainingData->numberOfOutputs; i++) {
		std::cout << this->h_partialResults.back()[i] << " "; 
	}
	std::cout << std::endl;
}

void Assignment::ReleaseClResources() {
	SAFE_RELEASE_MEMOBJECT(d_trainingInputBuffer);
	SAFE_RELEASE_MEMOBJECT(d_trainingLabelBuffer);
	for (unsigned int i = 0; i < this->d_weightBuffers.size(); i++) {
		SAFE_RELEASE_MEMOBJECT(this->d_weightBuffers[i]);
		SAFE_RELEASE_MEMOBJECT(this->d_partialResults[i]);
	}

	//release kernels
	SAFE_RELEASE_KERNEL(this->h_feedForwardKernel);

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
