#pragma once

//command description
	//the description of this command
	#define COMMAND_DESC "Assignment for the gpu programming practical course."

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

	//default path for the image training set
	#define INPUT_DATA_DEFAULT_PATH "./data/train-images-idx3-ubyte"

//parameter for input label file
	//the short arg name
	#define INPUT_LABEL_SHORT_ARG "l"

	//long arg name
	#define INPUT_LABEL_LONG_ARG "input-label"

	//description
	#define INPUT_LABEL_DESC "location of the file containing the label data for training or testing"

	//description of the expected format
	#define INPUT_LABEL_TYPE_DESC "file system path to the MNIST label data set"

	//default path for the label training set
	#define INPUT_LABEL_DEFAULT_PATH "./data/train-labels-idx1-ubyte"

//parameter for input xml file
	//the short arg name
	#define INPUT_XML_SHORT_ARG "x"

	//long arg name
	#define INPUT_XML_LONG_ARG "input-xml"

	//description
	#define INPUT_XML_DESC "location of the file containing a data set in xml data format"

	//description of the expected format
	//TODO: expand on the actual file format
	#define INPUT_XML_TYPE_DESC "file system path to xml data set"

	//default path for the label training set
	#define INPUT_XML_DEFAULT_PATH "./data/xor.xml"

#include <stdint.h>
#include <string>

#include "inputData.h"

class Assignment {

	private:
		InputData* trainingData;

		float* trainingInputBuffer;
		
		float* trainingLabelBuffer;

		//parse the cmd args
		void parseCMDArgs(int argc, char** argv);

	public:
		Assignment(int argc, char** argv);
		~Assignment();
};
