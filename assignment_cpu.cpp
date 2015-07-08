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
#include "CTimer.h"

Assignment::Assignment(int argc, char** argv) {

	this->trainingData = NULL;
	this->trainingInputBuffer = NULL;
	this->trainingLabelBuffer = NULL;

	//parse the command line args
	parseCMDArgs(argc, argv);

	//print some information about the training set
	this->trainingData->printInformation();

	//allocate and fill the buffer for the training data
	this->trainingInputBuffer =
		new float[this->trainingData->numberOfInputs * this->trainingData->numberOfSamples];
	this->trainingData->getInputBuffer(this->trainingInputBuffer);

	//allocate and fill the buffer for the training labels
	this->trainingLabelBuffer =
		new float[this->trainingData->numberOfOutputs * this->trainingData->numberOfSamples];
	this->trainingData->getLabelBuffer(this->trainingLabelBuffer);
	
	this->initWeightBuffer();
	this->randomizeWeights();

	//this code displays one image from the mnist set as ascii art and the corresponding label	

	/*for (int k = 0; k < 10; k++) {

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

	}*/
	
}

//////////////////////////////////////////////////////////////////////////////
//initialize host memory weight vectors for each hidden layer and output layer
//////////////////////////////////////////////////////////////////////////////
void Assignment::initWeightBuffer() {
	

	std::cout << std::endl << "initWeightBuffer(): initialize the weight buffers" << std::endl;
	
	//dont forget the constant 1 weights!
	//the first hidden layer needs the number of input neurons
	this->sizeOfWeightBuffer.push_back(
		this->trainingData->numberOfInputs * this->hiddenLayers[0] + this->hiddenLayers[0]
		);
	this->h_weightBuffers.push_back(new float[this->sizeOfWeightBuffer.back()]);
	
	std::cout << "layer 0 has " << this->sizeOfWeightBuffer.back() << " weights" << std::endl;

	//hideden layers
	for (unsigned int i = 1; i < this->hiddenLayers.size(); i++) {
		this->sizeOfWeightBuffer.push_back(
			this->hiddenLayers[i-1] * this->hiddenLayers[i] + this->hiddenLayers[i]
			);
		this->h_weightBuffers.push_back(new float[this->sizeOfWeightBuffer.back()]);

		std::cout << "layer " << i << " has " << this->sizeOfWeightBuffer.back() << " weights" << std::endl;
	}
	
	//the last layer needs the number of outputs
	this->sizeOfWeightBuffer.push_back(
		this->hiddenLayers.back() * this->trainingData->numberOfOutputs + this->trainingData->numberOfOutputs
	);
	this->h_weightBuffers.push_back(new float[this->sizeOfWeightBuffer.back()]);

	//std::cout << "this->sizeOfWeightBuffer: " << this->sizeOfWeightBuffer.back() << std::endl;

	std::cout << "outputlayer (layer " << this->h_weightBuffers.size()-1 << ") has " <<
		this->sizeOfWeightBuffer.back() << " weights" << std::endl;

	//init the temp buffer for feedForward partial results and backPropagation deltas 
	for (unsigned int i = 0; i < this->hiddenLayers.size(); i++) {
		this->h_partialResults.push_back(new float[this->hiddenLayers[i]]);
		this->h_deltaBuffer.push_back(new float[this->hiddenLayers[i]]);
	}
	//dont forget the output layer also has output and deltas
	this->h_partialResults.push_back(new float[this->trainingData->numberOfOutputs]);
	this->h_deltaBuffer.push_back(new float[this->trainingData->numberOfOutputs]);

	//init the deltaUpdates buffer
	for (unsigned int i = 0; i < this->sizeOfWeightBuffer.size(); i++) {
		this->h_deltaUpdates.push_back(new float[this->sizeOfWeightBuffer[i]]);
	}
}

////////////////////////////////////////////////////////////////////////
//randomize the weight vectors of the hidden layers and the output layer
////////////////////////////////////////////////////////////////////////
void Assignment::randomizeWeights() {
	
	std::cout << std::endl << "randomizeWeights(): intializing random weights." << std::endl;

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
		}
	}

	/*std::cout << "first ten weights in cpu buffer (level 0): ";
	for (int i = 0; i < 10; i++) {
		std::cout << this->h_weightBuffers[0][i] << " ";
	}
	std::cout << std::endl;*/

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
					sum += this->trainingInputBuffer[k + index] * this->h_weightBuffers[i][k * numNeurons + j];
				} else {
					sum += this->h_partialResults[i-1][k] * this->h_weightBuffers[i][k * numNeurons + j];
				}
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
			}
		}

		/*std::cout << "Output of the neuronal network (layer " << i << "): ";
		for (int j = 0; j < numNeurons; j++) {
			std::cout << this->h_partialResults[i][j] << " "; 
		}
		std::cout << std::endl;*/

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
	/*std::cout << "Output of the neuronal network (CPU): ";
	for (unsigned int i = 0; i < this->trainingData->numberOfOutputs; i++) {
		std::cout << this->h_partialResults.back()[i] << " "; 
	}
	std::cout << std::endl;
	std::cout << "crossEntropy: " << crossEntropy << std::endl;*/
	
	return crossEntropy;
}


///////////////////////////////////////////////////////////////////////////////
//computes gradient of the neuronal network given an index in the input array
//it is assumed that the feed forward output corresponding to indexOfInput
//is present in h_partialResults
///////////////////////////////////////////////////////////////////////////////
void Assignment::gradientDescentCPU(unsigned int indexOfInput) {

	//compute index in label buffer
	int labelIndex = indexOfInput * this->trainingData->numberOfOutputs;
	int inputIndex = this->trainingData->numberOfInputs * indexOfInput;

	//compute deltas for the output layer
	for (unsigned int i = 0; i < this->trainingData->numberOfOutputs; i++) {
		//compute the delta
		float delta = this->trainingLabelBuffer[labelIndex + i] - h_partialResults.back()[i];
		this->h_deltaBuffer.back()[i] = delta;
		float input;
		//iterate over all inputs
		for (int j = 0; j < hiddenLayers.back() + 1; j++) {
			//get the right input
			if (j == hiddenLayers.back()) {
				input = 1.0f;
			} else {
				input = this->h_partialResults[hiddenLayers.size()-1][j]; 
			}
			//write back. Keep in mind that the delta buffer is organized like the weight buffer
			this->h_deltaUpdates.back()[j * this->trainingData->numberOfOutputs + i] += delta * input;
		}
	}

	//compute deltas for the hidden layers
	int numNeuronsNextLayer = this->trainingData->numberOfOutputs;
	int numberOfInputs;
	for (int i = this->hiddenLayers.size()-1; i >= 0; i--) {
		//get the number of inputs for this layer
		if (i == 0) {
			numberOfInputs = this->trainingData->numberOfInputs;
		} else {
			numberOfInputs = this->hiddenLayers[i-1];
		}
		//iterate over all neurons
		for (int j = 0; j < this->hiddenLayers[i]; j++) {
			//the delta
			float delta = 0.0f;
			//get the first part of the derivative 
			for (int k = 0; k < numNeuronsNextLayer; k++) {
				delta += this->h_deltaBuffer[i+1][k] * this->h_weightBuffers[i+1][j * numNeuronsNextLayer + k];
			}		
			//the second part of the derivative
			delta *= this->h_partialResults[i][j] * (1.0f - this->h_partialResults[i][j]);
			//save the delta
			this->h_deltaBuffer[i][j] = delta;
			//get the right input and write into the deltaUpdates buffer
			//iterate over the inputs (dont forget the constant input)
			float input;
			for (int k = 0; k < numberOfInputs + 1; k++) {
				//get the right input (3 possible )
				if(k == numberOfInputs) {
					input = 1.0f;
				} else {
					if (i == 0) {
						input = this->trainingInputBuffer[inputIndex + k];
					} else {
						input = this->h_partialResults[i-1][k];
					}
				}
				this->h_deltaUpdates[i][k * this->hiddenLayers[i] + j] += delta * input;
			}		
		}
		//set number of neurons for the next layer
		numNeuronsNextLayer = this->hiddenLayers[i];
	}
}

void Assignment::printDeltaUpdatesCPU() {
	for (unsigned int i = 0; i < this->h_weightBuffers.size(); i++) {
		//iterate over all deltas
		std::cout << "deltaUpdates for layer " << i << ": ";
		for (int j = 0; j < this->sizeOfWeightBuffer[i]; j++) {
			std::cout << this->h_deltaUpdates[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

void Assignment::updateWeightsCPU() {
	
	//iterate over all layers
	for (unsigned int i = 0; i < this->h_deltaUpdates.size(); i++) {
		//iterate over all weights
		for (int j = 0; j < this->sizeOfWeightBuffer[i]; j++) {
			this->h_weightBuffers[i][j] += this->h_deltaUpdates[i][j] * this->learningRate;
		}
	}
}

void Assignment::zeroDeltaBuffersCPU() {
	for (unsigned int i = 0; i < this->h_weightBuffers.size(); i++) {
		//iterate over all deltas
		for (int j = 0; j < this->sizeOfWeightBuffer[i]; j++) {
			this->h_deltaUpdates[i][j] = 0.0f;
		}
	}
}

void  Assignment::printDeltaBufferOutputLayerCPU() {
	std::cout << "Deltas of output Layer (CPU): ";
	for (int i = 0; i < this->sizeOfWeightBuffer.back(); i++) {
		std::cout << this->h_deltaUpdates.back()[i] << " ";
	}
	std::cout << std::endl;
}

void Assignment::printDeltaBufferCPU() {
	for (unsigned int i = 0; i < this->h_deltaBuffer.size(); i++) {
		int numNeurons;
		if (i == this->h_deltaBuffer.size() - 1) {
			numNeurons = this->trainingData->numberOfOutputs;
		} else {
			numNeurons = this->hiddenLayers[i];
		}
		std::cout << "deltaBuffer of layer " << i << ": ";
		for (int j = 0; j < numNeurons; j++) {
			std::cout << this->h_deltaBuffer[i][j] << " ";
		}
		std::cout << std::endl;
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

	std::vector<std::string> allowed;
	allowed.push_back("BACKPROP_STOCH_CPU");
	allowed.push_back("BACKPROP_STOCH_GPU");
	allowed.push_back("BACKPROP_BATCH_CPU");
	allowed.push_back("BACKPROP_BATCH_GPU");
	allowed.push_back("FEEDFORWARD_CPU");
	allowed.push_back("FEEDFORWARD_GPU");
	TCLAP::ValuesConstraint<std::string> allowedVals(allowed);
        
	TCLAP::ValueArg<std::string> taskArg(
		TASK_SHORT_ARG,
		TASK_LONG_ARG,
		TASK_DESC,
		false,
		TASK_DEFAULT_TASK,
		&allowedVals);
	cmd.add( taskArg );

	TCLAP::ValueArg<int> numEpochsArg(
		NUM_EPOCHS_SHORT_ARG,
		NUM_EPOCHS_LONG_ARG,
		NUM_EPOCHS_DESC,
		false, //required argument
		NUM_EPOCHS_DEFAULT_NUMBER, //default value
		NUM_EPOCHS_TYPE_DESC);

	cmd.add( numEpochsArg );

	TCLAP::ValueArg<int> batchSizeArg(
		BATCH_SIZE_SHORT_ARG,
		BATCH_SIZE_LONG_ARG,
		BATCH_SIZE_DESC,
		false, //required argument
		BATCH_SIZE_DEFAULT_NUMBER, //default value
		BATCH_SIZE_TYPE_DESC);

	cmd.add( batchSizeArg );

	TCLAP::ValueArg<float> learningRateArg(
		LEARNING_RATE_SHORT_ARG,
		LEARNING_RATE_LONG_ARG,
		LEARNING_RATE_DESC,
		false, //required argument
		LEARNING_RATE_DEFAULT_VALUE, //default value
		LEARNING_RATE_TYPE_DESC);

	cmd.add( learningRateArg );

	// Parse the argv array.
	try {  
		cmd.parse( argc, argv );
	}
	catch (TCLAP::ArgException &e) { // catch any exceptions
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}

	//read a file
	if (inputXMLArg.isSet()) {
		if (inputLabelArg.isSet() || inputDataArg.isSet()) {
			std::cout << "You have specified multiple data sources. The XML input will" <<
				" be used and the rest discarded." << std::endl;
		}
		//construct the training data object from xml file if the option was specified
		this->trainingData = new XMLInputData(inputXMLArg.getValue());
	} else {
		//construct the training data object from MNIST data if no xml is present
		//TODO: error handling is needed if file is not present->buffer of InputData will be NULL!
		this->trainingData = new BinaryInputData(inputDataArg.getValue(), inputLabelArg.getValue());
	}

	//the hidden layers
	this->hiddenLayers = hiddenLayerMultiArg.getValue();

	//the task at hand
	std::string taskStr = taskArg.getValue();
	if (taskStr == "BACKPROP_STOCH_CPU") this->task = BACKPROP_STOCH_CPU;
	else if (taskStr == "BACKPROP_STOCH_GPU") this->task = BACKPROP_STOCH_GPU;
	else if (taskStr == "BACKPROP_BATCH_CPU") this->task = BACKPROP_BATCH_CPU;
	else if (taskStr == "BACKPROP_BATCH_GPU") this->task = BACKPROP_BATCH_GPU;
	else if (taskStr == "FEEDFORWARD_CPU") this->task = FEEDFORWARD_CPU;
	else this->task = FEEDFORWARD_GPU;

	//the number of epochs
	this->numEpochs = numEpochsArg.getValue();

	//size of one batch for back propagation
	this->batchSize = batchSizeArg.getValue();

	//the learning rate
	this->learningRate = learningRateArg.getValue();
}

void Assignment::scheduleTask() {
	switch(this->task) {
		case FEEDFORWARD_CPU:
			feedForwardTaskCPU();
			break;
		case BACKPROP_STOCH_CPU:
			StochasticBackPropagateTaskCPU(this->numEpochs);
			break;
		case BACKPROP_BATCH_CPU:
			batchBackPropagateTaskCPU(this->numEpochs, this->batchSize);
			break;
		default:
			std::cout << "something went very wrong with the enum";
			break;
	}
}

void Assignment::feedForwardTaskCPU() {
	std::cout << std::endl;
	std::cout << "You have selected the feed forward task on the CPU." << std::endl;
	std::cout << "Will now feed forward each sample, stop the time" <<
		" and accumulate the crossEntropy error." << std::endl;
	
	double crossEntropy = 0.0f;
	CTimer timer;

	timer.Start();
	for (unsigned int i = 0; i < this->trainingData->numberOfSamples; i++) {
		crossEntropy += feedForwardCPU(i);
	}
	timer.Stop();

	std::cout << "done." << std::endl;
	std::cout << "The crossEntropy error is " << crossEntropy << "." << std::endl;
	std::cout << "The task took " << timer.GetElapsedMilliseconds() <<
		" milliseconds to complete." << std::endl;
}

void Assignment::StochasticBackPropagateTaskCPU(unsigned int numEpochs) {
	std::cout << std::endl;
	std::cout << "You have selected the back propagation task on the CPU " << 
		"using stochastic gradient descent." << std::endl;
	std::cout << "The neuronal network will now train " << this->trainingData->numberOfSamples <<
		" samples for " << numEpochs << " epochs with a learning rate of " <<
		this->learningRate << " and stop the time." << std::endl;
	
	CTimer timer;
	timer.Start();
	for (unsigned int i = 0; i < numEpochs; i++) {
		std::cout << "Starting with epoch " << i << std::endl;
		double crossEntropy = 0.0f;
		for (unsigned int j = 0; j < this->trainingData->numberOfSamples; j++) {
			zeroDeltaBuffersCPU();
			crossEntropy += feedForwardCPU(j);
			gradientDescentCPU(j);
			updateWeightsCPU();
		}
		std::cout << "Done." << std::endl;
		std::cout << "Accumulated crossEntropy error for this epoch: " << crossEntropy << std::endl;
	}
	timer.Stop();

	std::cout << "Done with back propagation (stochastic)." << std::endl;
	std::cout << "The task took " << timer.GetElapsedMilliseconds() <<
		" milliseconds to complete." << std::endl;
}

void Assignment::batchBackPropagateTaskCPU(unsigned int numEpochs, unsigned int batchSize) {
	std::cout << std::endl;
	std::cout << "You have selected the back propagation task on the CPU " << 
		"using batch gradient descent with a batch size of " << batchSize << "." << std::endl;
	std::cout << "The neuronal network will now train " << this->trainingData->numberOfSamples <<
		" samples for " << numEpochs << " epochs with a learning rate of " <<
		this->learningRate << " and stop the time." << std::endl;
	
	CTimer timer;
	timer.Start();
	for (unsigned int i = 0; i < numEpochs; i++) {
		std::cout << "Starting with epoch " << i << std::endl;
		double crossEntropy = 0.0f;
		int fullBatches =  this->trainingData->numberOfSamples / batchSize;
		//full batches			
		for (int j = 0; j < fullBatches; j++) {
			zeroDeltaBuffersCPU();
			for (unsigned int k = 0; k < batchSize; k++) {
				crossEntropy += feedForwardCPU(j * batchSize + k);
				gradientDescentCPU(j * batchSize + k);
			}
			updateWeightsCPU();
		}
		//the last batch is smaller
		zeroDeltaBuffersCPU();
		for (unsigned int j = fullBatches * batchSize; j < this->trainingData->numberOfSamples; j++) {
			crossEntropy += feedForwardCPU(j);
			gradientDescentCPU(j);
		}
		updateWeightsCPU();

		std::cout << "Done." << std::endl;
		std::cout << "Accumulated crossEntropy error for this epoch: " << crossEntropy << std::endl;
	}
	timer.Stop();

	std::cout << "Done with back propagation (batch)." << std::endl;
	std::cout << "The task took " << timer.GetElapsedMilliseconds() <<
		" milliseconds to complete." << std::endl;
}

///////////////////////////////////////////////////////////////
//trains the neuronal network using stochastic gradient descent
//epoch: number of samples per epoch
//numEpochs: number of epochs to train
///////////////////////////////////////////////////////////////
void Assignment::stochasticGradientDescentCPU(unsigned int epoch, unsigned int numEpochs) {
	
		std::random_device rd;
		unsigned int random;
		for (unsigned int j = 0; j < numEpochs; j++) {
			double entropy = 0.0f;
			for (unsigned int i = 0; i < epoch; i++) {
				//random = rd() % this->trainingData->numberOfSamples;
				random = rd() % 10;
				//random = 0;
				zeroDeltaBuffersCPU();
				entropy += feedForwardCPU(random);
				gradientDescentCPU(random);
				updateWeightsCPU();
			}
			std::cout << "Entropy: " << entropy << " Epoch " << j << std::endl;
			
			int number = random;
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
			
			std::cout << "output: ";
			int guess;
			float max = 0.0;
			for (int i = 0; i < 10; i++) {
				std::cout << this->h_partialResults.back()[i] << " ";
				if (std::max(max, this->h_partialResults.back()[i]) == this->h_partialResults.back()[i]) {
					max = this->h_partialResults.back()[i];
					guess = i;
				}			
			}
			std::cout << std::endl;
			std::cout << "guess: " << guess << std::endl;
		}
}

////////////////////////////////////////////////////////////////
// Destructor
////////////////////////////////////////////////////////////////
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
		delete[] this->h_deltaBuffer[i];
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
