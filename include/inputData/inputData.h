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

		/////////////////////////////////////////////////////////////
		//This method writes all input vectors into a provided buffer
		//It is assumed that the buffer has appropriate length
		/////////////////////////////////////////////////////////////
		virtual void getInputBuffer(float* buffer) = 0;

		/////////////////////////////////////////////////////////////
		//This method writes all label vectors into a provided buffer
		//It is assumed that the buffer has appropriate length
		/////////////////////////////////////////////////////////////
		virtual void getLabelBuffer(float* buffer) = 0;

		InputData() {};
		virtual ~InputData() {};
};
