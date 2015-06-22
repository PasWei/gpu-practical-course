#pragma once

#include <stdint.h>
#include <string>

#include "inputData.h"



class BinaryInputData : public InputData {
	
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

		void printInformation();
		BinaryInputData(const std::string DataPath, const std::string LabelPath);
		~BinaryInputData();
};
