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

//multi parameter to read the number and size of the neuronal net hidden layer
	//the short arg name
	#define INPUT_HIDDEN_SHORT_ARG "H"

	//long arg name
	#define INPUT_HIDDEN_LONG_ARG "hidden-layer"

	//description
	#define INPUT_HIDDEN_DESC "number of neurons in a new hidden layer. Can be specified multiple times. The order matters."

	//description of the actual format
	#define INPUT_HIDDEN_TYPE_DESC "Name intergers in the order you want to arrange the hidden layers."




#include <stdint.h>
#include <string>
#include <vector>

#include "inputData.h"

class Assignment {

	private:
		InputData* trainingData;

		float* trainingInputBuffer;
		
		float* trainingLabelBuffer;

		//number of hidden layers and number of neurons in each layer
		std::vector<int> hiddenLayers;

		//contains the sizes of each weight buffer
		std::vector<int> sizeOfWeightBuffer;

		//pointers to the weight buffers of each layer
		std::vector<float*> h_weightBuffers;

		/////////////////////////////////////////////////////////////////////////////////
		// This function parses all the cmd arguments into member variables for later use
		// It sets the training set object
		/////////////////////////////////////////////////////////////////////////////////
		void parseCMDArgs(int argc, char** argv);

		//////////////////////////////////////////////////////////////////////////////
		//initialize host memory weight vectors for each hidden layer and output layer
		//////////////////////////////////////////////////////////////////////////////
		void initWeightBuffer();

		////////////////////////////////////////////////////////////////////////
		//randomize the weight vectors of the hidden layers and the output layer
		////////////////////////////////////////////////////////////////////////
		void randomizeWeights();

	public:
		Assignment(int argc, char** argv);
		~Assignment();
};
