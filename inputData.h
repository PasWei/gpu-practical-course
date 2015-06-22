#pragma once

#include <stdint.h>
#include <string>

class InputData {
	
	private:

		//byte buffer for the data and labels
		uint8_t* ImageBuffer;
		uint8_t* LabelBuffer;

		//parse training data 
		uint8_t* parseFileToBuffer(std::string filePath);

		//file paths to the data and labels
		std::string DataPath;
		std::string LabelPath;

	public:

		//number of training samples in the loaded image and label files
		unsigned int numberOfSamples;

		InputData(std::string DataPath, std::string LabelPath);
		~InputData();
};
