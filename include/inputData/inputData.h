#pragma once

#include <stdint.h>
#include <string>

class InputData {
	
	//private:

	public:

		//number of training samples in the loaded image and label files
		unsigned int numberOfSamples;

		virtual void printInformation() = 0;

		InputData() {};
		virtual ~InputData() {};
};
