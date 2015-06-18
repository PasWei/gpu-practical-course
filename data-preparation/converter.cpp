#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>

#include <tclap/CmdLine.h>

//command description
	//the description of this command
	#define COMMAND_DESC "Command for converting MNIST IDX data to readable xml data"

	//the version string of this command
	#define VERSION_STRING "0.999"

//parameter for input data file
	//the short arg name
	#define INPUT_DATA_SHORT_ARG "i"

	//long arg name
	#define INPUT_DATA_LONG_ARG "input-data"

	//description
	#define INPUT_DATA_DESC "location of the file containing the image data for training or testing"

	//description of the expected format
	#define INPUT_DATA_TYPE_DESC "file system path to the MNIST image data set"

//parameter for input label file
	//the short arg name
	#define INPUT_LABEL_SHORT_ARG "l"

	//long arg name
	#define INPUT_LABEL_LONG_ARG "input-label"

	//description
	#define INPUT_LABEL_DESC "location of the file containing the label data for training or testing"

	//description of the expected format
	#define INPUT_LABEL_TYPE_DESC "file system path to the MNIST label data set"

int main(int argc, char** argv)
{
	//initialize cmd line parsing
	TCLAP::CmdLine cmd(COMMAND_DESC, ' ', VERSION_STRING);

	//argument for specification of the training data file
	TCLAP::ValueArg<std::string> inputDataArg(
		INPUT_DATA_SHORT_ARG,
		INPUT_DATA_LONG_ARG,
		INPUT_DATA_DESC,
		true, //required argument
		"", //default value
		INPUT_DATA_TYPE_DESC);

	cmd.add( inputDataArg );

	//argument for specification of the training data file
	TCLAP::ValueArg<std::string> inputLabelArg(
		INPUT_LABEL_SHORT_ARG,
		INPUT_LABEL_LONG_ARG,
		INPUT_LABEL_DESC,
		true, //required argument
		"", //default value
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
	std::string inputDataFilepath = inputDataArg.getValue();

	// Get the input label value
	std::string inputLabelFilepath = inputLabelArg.getValue();

	//try
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

	std::cout << "the data filename is: " << inputDataFilepath << std::endl << " the label filename is: " << inputLabelFilepath << std::endl << " buffer size " << data.size() << std::endl << " data type is: " << dataType << std::endl;

}
