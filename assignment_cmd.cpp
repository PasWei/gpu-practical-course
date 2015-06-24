#include <string>
#include <iostream>
#include <random>

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
	/*
	int number = 59998;
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
	*/
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
	
	if (this->trainingInputBuffer != NULL) {	
		delete[] this->trainingInputBuffer;
		this->trainingInputBuffer = NULL;
	}

	if (this->trainingLabelBuffer != NULL) {	
		delete[] this->trainingLabelBuffer;
		this->trainingLabelBuffer = NULL;
	}

	for (unsigned int i = 0; i < this->h_weightBuffers.size(); i++) {
		delete[] this->h_weightBuffers[i];
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
