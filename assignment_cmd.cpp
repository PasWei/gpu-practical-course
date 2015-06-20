#include <string>
#include <iostream>

#include "assignment.h"

#include "tclap/CmdLine.h"
#include "tgawriter.h"

Assignment::Assignment(int argc, char** argv) {

	this->h_trainingImageBuffer = NULL;
	this->h_trainingLabelBuffer = NULL;

	parseCMDArgs(argc, argv);

	std::cout << "the data filename is: " << this->h_trainingDataPath <<
	std::endl << "the label filename is: " << this->h_trainingLabelPath <<
	std::endl;

	this->h_trainingImageBuffer = parseFileToBuffer(this->h_trainingDataPath);
	this->h_trainingLabelBuffer = parseFileToBuffer(this->h_trainingLabelPath);

	
}

/////////////////////////////////////////////////////////////////////////////////
// This function parses all the cmd arguments into member variables for later use 
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

	// Parse the argv array.
	try {  
		cmd.parse( argc, argv );
	}
	catch (TCLAP::ArgException &e) { // catch any exceptions
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}

	// Get the input data value
	this->h_trainingDataPath = inputDataArg.getValue();

	// Get the input label value
	this->h_trainingLabelPath = inputLabelArg.getValue();
}

/////////////////////////////////////////////////////////////////////////////////////////
// this function loads a binary file into abuffer
// filePath: the path to the file to be loaded
// return: Pointer to a binary array. The array is allocated with new and has to be freed
/////////////////////////////////////////////////////////////////////////////////////////
uint8_t* Assignment::parseFileToBuffer(std::string filePath) {

	std::ifstream is (filePath, std::ifstream::binary);

	//check if the file is valid
  	if (is) {
		// get length of file:
		is.seekg (0, is.end);
		int length = is.tellg();
		is.seekg (0, is.beg);

		uint8_t* buffer = new uint8_t[length];

		//std::cout << "Reading " << length << " characters... ";
		// read data as a block:
		is.read ((char*)buffer,length);

		//check if all input was read
		if (!is) {
			std::cout << "error: only " << is.gcount() << " could be read from " << filePath << std::endl;
			delete[] buffer;
			return NULL;
		}

		//close
		is.close();

		return buffer;

	//not a valid file
	} else {
		std::cout << "cound not load file " << filePath << std::endl;
		return NULL;
	}
}

////////////////////////////////////////////////////////////////
// Destructor
///////////////////////////////////////////////////////////////
Assignment::~Assignment() {
	if (this->h_trainingImageBuffer != NULL) {	
		delete[] this->h_trainingImageBuffer;
		this->h_trainingImageBuffer = NULL;
	}
	if (this->h_trainingLabelBuffer != NULL) {	
		delete[] this->h_trainingLabelBuffer;
		this->h_trainingLabelBuffer = NULL;
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
