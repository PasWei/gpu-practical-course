#include <string> //for string
#include <iostream> //for cout, endl
#include <random> //for device_random
#include <algorithm> //for max
#include <utility> //for swap (since c++11)
#include <cmath> //for exp

#include "assignment.h"

#include "tclap/CmdLine.h"
#include "tgawriter.h"
#include "binaryInputData.h"
#include "xmlInputData.h"

Assignment::Assignment(int argc, char** argv) {

	this->trainingData = NULL;
	this->trainingInputBuffer = NULL;
	this->trainingLabelBuffer = NULL;

	//parse the command line args
	parseCMDArgs(argc, argv);

	//print some information about the training set
	this->trainingData->printInformation();

	//allocate and fill the buffer for the training data
	this->trainingInputBuffer = new float[this->trainingData->numberOfInputs * this->trainingData->numberOfSamples];
	this->trainingData->getInputBuffer(this->trainingInputBuffer);

	//allocate and fill the buffer for the training labels
	this->trainingLabelBuffer = new float[this->trainingData->numberOfOutputs * this->trainingData->numberOfSamples];
	this->trainingData->getLabelBuffer(this->trainingLabelBuffer);
	
	this->initWeightBuffer();
	this->randomizeWeights();
	//this code displays one image from the mnist set as ascii art and the corresponding label	

	for (int k = 0; k < 10; k++) {

	int number = k;
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			int addr = i*28 + j + 28*28*number;
			if (this->trainingInputBuffer[addr] < 0.05f) {
				std::cout << " ";
			} else if (this->trainingInputBuffer[addr] < 0.5f) {
				std::cout << ".";
			} else {
				std::cout << "0";
			}
		}
		std::cout << std::endl;
	}
	std::cout << "label: ";
	for (int i = 0; i < 10; i++) {
		int addr = 10 * number + i;
		std::cout << this->trainingLabelBuffer[addr] << " ";	
	}
	std::cout << std::endl;

	}
	
}

//////////////////////////////////////////////////////////////////////////////
//initialize host memory weight vectors for each hidden layer and output layer
//////////////////////////////////////////////////////////////////////////////
void Assignment::initWeightBuffer() {
	

	std::cout << std::endl << "initialize the weight buffers" << std::endl;
	
	//dont forget the constant 1 weights!
	//the first hidden layer needs the number of input neurons
	this->sizeOfWeightBuffer.push_back(this->trainingData->numberOfInputs * this->hiddenLayers[0] + this->hiddenLayers[0]);
	this->h_weightBuffers.push_back(new float[this->sizeOfWeightBuffer.back()]);
	
	std::cout << "layer 0 has " << this->sizeOfWeightBuffer.back() << " weights" << std::endl;

	//hideden layers
	for (unsigned int i = 1; i < this->hiddenLayers.size(); i++) {
		this->sizeOfWeightBuffer.push_back(this->hiddenLayers[i-1] * this->hiddenLayers[i] + this->hiddenLayers[i]);
		this->h_weightBuffers.push_back(new float[this->sizeOfWeightBuffer.back()]);

		std::cout << "layer " << i << " has " << this->sizeOfWeightBuffer.back() << " weights" << std::endl;
	}
	
	//the last layer needs the number of outputs
	this->sizeOfWeightBuffer.push_back(
		this->hiddenLayers.back() * this->trainingData->numberOfOutputs + this->trainingData->numberOfOutputs
	);
	this->h_weightBuffers.push_back(new float[this->sizeOfWeightBuffer.back()]);

	std::cout << "outputlayer (layer " << this->h_weightBuffers.size()-1 << ") has " <<
		this->sizeOfWeightBuffer.back() << " weights" << std::endl;

	//init the temp buffer for feedForward partial results and backPropagation delta updates 
	for (unsigned int i = 0; i < this->hiddenLayers.size(); i++) {
		this->h_partialResults.push_back(new float[this->hiddenLayers[i]]);
		this->h_deltaUpdates.push_back(new float[this->hiddenLayers[i]]);
	}
	//dont forget the output layer also has output and deltas
	this->h_partialResults.push_back(new float[this->trainingData->numberOfOutputs]);
	this->h_deltaUpdates.push_back(new float[this->trainingData->numberOfOutputs]);
}

////////////////////////////////////////////////////////////////////////
//randomize the weight vectors of the hidden layers and the output layer
////////////////////////////////////////////////////////////////////////
void Assignment::randomizeWeights() {
	
	//random device for unsigned integers
	std::random_device rd;
	unsigned int random;
	float sign;
	for (unsigned int i = 0; i < this->h_weightBuffers.size(); i++) {
		for (int j = 0; j < this->sizeOfWeightBuffer[i]; j++) {

			//determine the sign
			random = rd();
			sign = -1.0f;
			if (random > rd.max()/2) {
				sign = 1.0f;
			}

			//compute random float
			this->h_weightBuffers[i][j] = sign * (((float) rd()) / ((float) rd.max()));
			/*if (i == 0) {
				std::cout << this->h_weightBuffers[i][j] << std::endl;
			}*/
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
//computes the output of the neuronal network given an index in the input array
///////////////////////////////////////////////////////////////////////////////
float Assignment::feedForwardCPU(unsigned int indexOfInput) {
	
	//is the index valid?
	if (indexOfInput >= this->trainingData->numberOfSamples) {
		std::cout << indexOfInput << " is not a valid index (out of bounds)!" << std::endl;
	}

	//calculate the actual index in the input array and label array
	int index = this->trainingData->numberOfInputs * indexOfInput;
	int labelIndex = indexOfInput * this->trainingData->numberOfOutputs;

	//number of inputs for each layer. Initialized with the first.
	int inputSize = this->trainingData->numberOfInputs;

	//compute the hidden layers
	//iterate over all hidden layers and the output layer (+1)
	for (unsigned int i = 0; i < hiddenLayers.size() + 1; i ++) {

		//we need to make case distinction between hidden and output layer
		int numNeurons;
		//the output layer
		if (i == hiddenLayers.size()) {
			numNeurons = this->trainingData->numberOfOutputs;
		//a hidden layer
		} else {
			numNeurons = hiddenLayers[i];
		}

		//iterate over all neurons
		for (int j = 0; j < numNeurons; j++) {
			float sum = 0.0f;
			//iterate over all weights of the neuron, dont forget the const 1 input
			for (int k = 0; k < inputSize; k++) {
				//remember, the weights for one neuron are not consecutive but
				//stored in the form of transposed vectors, so all first weights
				//of all neurons first, then all second weights and so on
				//this is to be conform with the gpu implementation with coalesced load
				if (i == 0) {
					sum += this->trainingInputBuffer[k + index] * h_weightBuffers[i][k * numNeurons + j];
				} else {
					sum += this->h_partialResults[i-1][k] * h_weightBuffers[i][k * numNeurons + j];
				}

				//std::cout << "i: " << i << " j: " << j << " k: " << k << " input: " << buf1[k] << " weight: " <<
				//	h_weightBuffers[i][k * numNeurons + j] << " sum: " << sum << " inputSize: " << inputSize << std::endl;
			}
			//the constant one
			sum += h_weightBuffers[i][inputSize * numNeurons + j];

			//std::cout << "i: " << i << " j: " << j << " k: "<< inputSize << " input: 1 weight: " <<
			//		h_weightBuffers[i][inputSize * numNeurons + j] << " sum: " << sum << std::endl;

			//again we need to distinguish between output layer and hidden layer for the
			//activation function
			//since we need all outputs of the output layer before we can compute the softmax
			//activation function, we have to delay it
			if (i != hiddenLayers.size()) {
				//now the sum of all weighted inputs needs to pass the sigmoid output function
				//since we are in the hidden layer
				this->h_partialResults[i][j] = 1.0f / (1.0f + std::exp(-sum));
			//its the output layer
			} else {
				this->h_partialResults[i][j] = sum;
				//std::cout << "output: " << buf2[j] << std::endl;
			}
		}

 		//set the correct number of inputs for the next layer
		inputSize = numNeurons;
	}

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
	/*std::cout << "Output of the neuronal network: ";
	for (unsigned int i = 0; i < this->trainingData->numberOfOutputs; i++) {
		std::cout << this->h_partialResults.back()[i] << " "; 
	}
	std::cout << std::endl;*/
	
	return crossEntropy;
}


///////////////////////////////////////////////////////////////////////////////
//computes gradient of the neuronal network given an index in the input array
//it is assumed that the feed forward output corresponding to indexOfInput
//is present in h_partialResults
///////////////////////////////////////////////////////////////////////////////
void Assignment::backPropagationCPU(unsigned int indexOfInput) {

	//compute index in label buffer
	int labelIndex = indexOfInput * this->trainingData->numberOfOutputs;
	int inputIndex = this->trainingData->numberOfInputs * indexOfInput;
	
	//compute deltas for the output layer
	for (unsigned int i = 0; i < this->trainingData->numberOfOutputs; i++) {
		this->h_deltaUpdates.back()[i] = this->trainingLabelBuffer[labelIndex + i] - h_partialResults.back()[i];	
	}

	//compute deltas for the hidden layers
	int numNeuronsNextLayer = this->trainingData->numberOfOutputs;
	//iterate over all hidden layers
	for (int i = this->hiddenLayers.size()-1; i >= 0; i--) {
		//iterate over all neurons in that layer
		for (int j = 0; j < this->hiddenLayers[i]; j++) {
			//the delta
			float delta = 0.0f;

			for (int k = 0; k < numNeuronsNextLayer; k++) {
				delta += this->h_deltaUpdates[i+1][k] *  h_weightBuffers[i+1][k * numNeuronsNextLayer + j];
			}
			
			//the second part of the derivative
			delta *= this->h_partialResults[i][j] * (1.0f - this->h_partialResults[i][j]);
			//write to buffer
			this->h_deltaUpdates[i][j] = delta;
		}
		
		//set number of neurons for the next layer
		numNeuronsNextLayer = this->hiddenLayers[i];
	}

	//update the weights

	int numNeuronsPreviousLayer = this->trainingData->numberOfInputs;
	int numNeuronsThisLayer = this->hiddenLayers[0];

	//iterate over all layers
	for (unsigned int i = 0; i < this->h_weightBuffers.size(); i++) {		
		//enumerate all inputs
		for (int j = 0; j < numNeuronsPreviousLayer; j++) {
			//get the input x_{ij}
			float input;
			if (i == 0) {
				input = this->trainingInputBuffer[inputIndex + j];
			} else {
				input = this->h_partialResults[i-1][j];
			}
			//enumerate all deltas
			for (int k = 0; k < numNeuronsThisLayer; k++) {
				float delta = this->h_deltaUpdates[i][k] * this->learningRate;
				h_weightBuffers[i][j * k] += delta * input; 
			}
		}
		
		//update numbers
		if (i == this->h_weightBuffers.size() - 2) {
			numNeuronsThisLayer = this->trainingData->numberOfOutputs;
		} else {
			numNeuronsThisLayer = this->hiddenLayers[i+1];
		}

		numNeuronsPreviousLayer = this->hiddenLayers[i];
	}
}

/////////////////////////////////////////////////////////////////////////////////
// This function parses all the cmd arguments into member variables for later use
// It sets the training set object
/////////////////////////////////////////////////////////////////////////////////
void Assignment::parseCMDArgs(int argc, char** argv) {
	//initialize cmd line parsing
	TCLAP::CmdLine cmd(COMMAND_DESC, ' ', VERSION_STRING);

	//argument for specification of the training data file
	TCLAP::ValueArg<std::string> inputDataArg(
		INPUT_DATA_SHORT_ARG,
		INPUT_DATA_LONG_ARG,
		INPUT_DATA_DESC,
		false, //required argument
		INPUT_DATA_DEFAULT_PATH, //default value
		INPUT_DATA_TYPE_DESC);

	cmd.add( inputDataArg );

	//argument for specification of the training data file
	TCLAP::ValueArg<std::string> inputLabelArg(
		INPUT_LABEL_SHORT_ARG,
		INPUT_LABEL_LONG_ARG,
		INPUT_LABEL_DESC,
		false, //required argument
		INPUT_LABEL_DEFAULT_PATH, //default value
		INPUT_LABEL_TYPE_DESC);

	cmd.add( inputLabelArg );

	//argument for specification of the xml data file
	TCLAP::ValueArg<std::string> inputXMLArg(
		INPUT_XML_SHORT_ARG,
		INPUT_XML_LONG_ARG,
		INPUT_XML_DESC,
		false, //required argument
		INPUT_XML_DEFAULT_PATH, //default value
		INPUT_XML_TYPE_DESC);

	cmd.add( inputXMLArg );

	TCLAP::MultiArg<int> hiddenLayerMultiArg(
		INPUT_HIDDEN_SHORT_ARG,
		INPUT_HIDDEN_LONG_ARG,
		INPUT_HIDDEN_DESC,
		true,
		INPUT_HIDDEN_TYPE_DESC);

	cmd.add(hiddenLayerMultiArg);	

	// Parse the argv array.
	try {  
		cmd.parse( argc, argv );
	}
	catch (TCLAP::ArgException &e) { // catch any exceptions
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}

	if (inputXMLArg.isSet()) {
		//construct the training data object from xml file if the option was specified
		this->trainingData = new XMLInputData(inputXMLArg.getValue());
	} else {
		//construct the training data object from MNIST data if no xml is present
		//TODO: error handling is needed if file is not present->buffer of InputData will be NULL!
		this->trainingData = new BinaryInputData(inputDataArg.getValue(), inputLabelArg.getValue());
	}

	this->hiddenLayers = hiddenLayerMultiArg.getValue();
}

////////////////////////////////////////////////////////////////
// Destructor
///////////////////////////////////////////////////////////////
Assignment::~Assignment() {
	//delete the training data object if initialized
	if (this->trainingData != NULL) {	
		delete this->trainingData;
		this->trainingData = NULL;
	}
	
	//delete the training data and label buffers
	if (this->trainingInputBuffer != NULL) {	
		delete[] this->trainingInputBuffer;
		this->trainingInputBuffer = NULL;
	}

	if (this->trainingLabelBuffer != NULL) {	
		delete[] this->trainingLabelBuffer;
		this->trainingLabelBuffer = NULL;
	}

	//delete the weight buffers
	for (unsigned int i = 0; i < this->h_weightBuffers.size(); i++) {
		delete[] this->h_weightBuffers[i];
	}

	//delete the temporary buffers
	for (unsigned int i = 0; i < this->hiddenLayers.size() + 1; i++) {
		delete[] this->h_partialResults[i];
		delete[] this->h_deltaUpdates[i];
	}

}

	//this code writes out an tga image
	/*
	for (int j = 0; j < 19; j++) {
		tgawriter::RGB_t img[28*28];

		for (unsigned int i = 0; i < 28*28; i++) {
			img[i].red = data[i+16 + j*28*28];
			img[i].green = data[i+16 + j*28*28];
			img[i].blue = data[i+16 + j*28*28];
		}

		std::string outname = "out.tga" + std::to_string(j);

		tgawriter::write_truecolor_tga(outname , img, 28, 28);
	}*/
