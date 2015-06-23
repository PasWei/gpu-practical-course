#pragma once

#include <stdint.h>
#include <string>

class InputData {
	
	//private:

	public:

		//number of training samples in the loaded file
		unsigned int numberOfSamples;
		
		//number of inputs for one sample
		unsigned int numberOfInputs;

		//number of outputs for one sample
		unsigned int numberOfOutputs;

		virtual void printInformation() = 0;

		InputData() {};
		virtual ~InputData() {};
};
