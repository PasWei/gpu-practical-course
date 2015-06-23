#include <string>
#include <iostream>

#include "assignment.h"

#include "tclap/CmdLine.h"
#include "tgawriter.h"
#include "binaryInputData.h"
#include "xmlInputData.h"

Assignment::Assignment(int argc, char** argv) {

	this->trainingData = NULL;

	parseCMDArgs(argc, argv);

	this->trainingData->printInformation();
	
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
}

////////////////////////////////////////////////////////////////
// Destructor
///////////////////////////////////////////////////////////////
Assignment::~Assignment() {
	//delete the image buffer if initialized
	if (this->trainingData != NULL) {	
		delete this->trainingData;
		this->trainingData = NULL;
	}
}

	/*//try
	std::ifstream inputData(inputDataFilepath, std::ios::binary);
	std::ifstream inputLabels(inputLabelFilepath, std::ios::binary);

	//see if the file was opened, exit with error otherwise
	if(!inputData.is_open()) {
		std::cout << "data file " << inputDataFilepath << " was not found!" << std::endl;
		exit(1);
	}

	if(!inputLabels.is_open()) {
		std::cout << "label file " << inputDataFilepath << " was not found!" << std::endl;
		exit(1);
	}

	//copy the data into buffers
	std::vector<char> data(
		(std::istreambuf_iterator<char>(inputData)),
		(std::istreambuf_iterator<char>()));
	std::vector<char> label(
		(std::istreambuf_iterator<char>(inputLabels)),
		(std::istreambuf_iterator<char>()));
	
	int dataType = data[2];

	std::cout << "the data filename is: " << inputDataFilepath <<
	std::endl << " the label filename is: " << inputLabelFilepath <<
	std::endl << " buffer size " << data.size() << std::endl <<
	" data type is: " << dataType << std::endl;
	
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
