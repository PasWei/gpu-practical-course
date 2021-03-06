#include <iostream>
#include <fstream>

#include "binaryInputData.h"

BinaryInputData::BinaryInputData(std::string DataPath, std::string LabelPath):InputData() {
	
	this->ImageBuffer = NULL;
	this->LabelBuffer = NULL;

	this->numberOfSamples = 0;

	this->DataPath = DataPath;
	this->LabelPath = LabelPath;

	this->ImageBuffer = parseFileToBuffer(this->DataPath);
	this->LabelBuffer = parseFileToBuffer(this->LabelPath);

	//check if the files are in the right format and coincide
	//TODO: if the buffers are NULL, abort

	//check image file
	int magicNumber = ImageBuffer[0] << 24 | ImageBuffer[1] << 16 | ImageBuffer[2] << 8 | ImageBuffer[3];  
	if (magicNumber != 2051) {
		std::cout << DataPath << " is not a file containing MNIST images" << std::endl;
	}

	//check label file
	magicNumber = LabelBuffer[0] << 24 | LabelBuffer[1] << 16 | LabelBuffer[2] << 8 | LabelBuffer[3];  
	if (magicNumber != 2049) {
		std::cout << DataPath << " is not a file containing MNIST labels" << std::endl;
	}

	//check if the number of images is the same as the number of labels
	unsigned int imageCount = ImageBuffer[4] << 24 | ImageBuffer[5] << 16 | ImageBuffer[6] << 8 | ImageBuffer[7];
	unsigned int labelCount = LabelBuffer[4] << 24 | LabelBuffer[5] << 16 | LabelBuffer[6] << 8 | LabelBuffer[7];

	if (imageCount != labelCount) {
		std::cout << "Error: the number of images in " << this->DataPath <<
			" differs from the number of labels in " << this->LabelPath << std::endl;
	} else {
		this->numberOfSamples = imageCount;
		this->numberOfInputs = 28*28;
		this->numberOfOutputs = 10; 
	}
}

/////////////////////////////////////////////////////////////////////////////////////////
// this function loads a binary file into a buffer
// filePath: the path to the file to be loaded
// return: Pointer to a binary array. The array is allocated with new and has to be freed
/////////////////////////////////////////////////////////////////////////////////////////
uint8_t* BinaryInputData::parseFileToBuffer(std::string filePath) {

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
	}
	return NULL;
}

void BinaryInputData::printInformation() {

	std::cout << std::endl << "InputData: This is a BinaryInputData instance." << std::endl; 	

	//status of image file
	std::cout << "loaded binary image file " << this->DataPath << ":";
	if (this->ImageBuffer != NULL) {
		std::cout << " success." << std::endl;
	} else {
		std::cout << " failed." << std::endl;
		return;
	}

	//status of label file
	std::cout << "loaded binary image file " << this->LabelPath << ":";
	if (this->LabelBuffer != NULL) {
		std::cout << " success." << std::endl;
	} else {
		std::cout << " failed." << std::endl;
		return;
	}

	if (this->numberOfSamples == 0) {
		std::cout << " the files do not coincide!" << std::endl;
		return;
	} else {
		std::cout << "loaded " << this->numberOfSamples << " labeled samples" << std::endl;
	}

}

/////////////////////////////////////////////////////////////
//This method writes all input vectors into a provided buffer
//It is assumed that the buffer has appropriate length
/////////////////////////////////////////////////////////////
void BinaryInputData::getInputBuffer(float* buffer) {

	int addr;

	for (unsigned int i = 0; i < this->numberOfSamples; i++) {
		for (unsigned int j = 0; j < 28*28; j++) {
			addr = i * this->numberOfInputs + j;
			buffer[addr] = ((float) this->ImageBuffer[16 + addr])/255.0f;
		}		
	}	
}

/////////////////////////////////////////////////////////////
//This method writes all input vectors into a provided buffer
//It is assumed that the buffer has appropriate length
/////////////////////////////////////////////////////////////
void BinaryInputData::getLabelBuffer(float* buffer) {

	int addr;

	for (unsigned int i = 0; i < this->numberOfSamples; i++) {
		for (unsigned int j = 0; j < 10; j++) {
			addr = i * this->numberOfOutputs + j;
			if (this->LabelBuffer[8 + i] == j)	{
				buffer[addr] = 1.0f;
			} else {
				buffer[addr] = 0.0f;
			}
		}		
	}	
}

///////////////////////////////////////////////////////////////////////////////
// Destructor
///////////////////////////////////////////////////////////////////////////////
BinaryInputData::~BinaryInputData() {

	

	//delete the image buffer if initialized
	if (this->ImageBuffer != NULL) {	
		delete[] this->ImageBuffer;
		this->ImageBuffer = NULL;
	}

	//delete the label buffer if initialized
	if (this->LabelBuffer != NULL) {	
		delete[] this->LabelBuffer;
		this->LabelBuffer = NULL;
	}
}
