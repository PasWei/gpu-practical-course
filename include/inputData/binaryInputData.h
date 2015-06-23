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

		/////////////////////////////////////////////////////////////
		//This method writes all input vectors into a provided buffer
		//It is assumed that the buffer has appropriate length
		/////////////////////////////////////////////////////////////
		void getInputBuffer(float* buffer);
		/////////////////////////////////////////////////////////////
		//This method writes all label vectors into a provided buffer
		//It is assumed that the buffer has appropriate length
		/////////////////////////////////////////////////////////////
		void getLabelBuffer(float* buffer);

};
