#include <iostream>
#include <string>

#include "xmlInputData.h"

XMLInputData::XMLInputData(const std::string filePath) {

	this->doc = NULL;

	this->filePath = filePath;
	this->doc = new tinyxml2::XMLDocument;
	this->doc->LoadFile(filePath.c_str());

	tinyxml2::XMLText* textNode = doc->FirstChildElement("samples")->FirstChildElement("sampleCount")->FirstChild()->ToText();
	std::string numSamples = textNode->Value();

	this->numberOfSamples = std::stoi(numSamples, NULL, 0);

	textNode = doc->FirstChildElement("samples")->FirstChildElement("sampleInputCount")->FirstChild()->ToText();
	std::string numInputs = textNode->Value();

	this->numberOfInputs = std::stoi(numInputs, NULL, 0);

	textNode = doc->FirstChildElement("samples")->FirstChildElement("sampleOutputCount")->FirstChild()->ToText();
	std::string numOutputs = textNode->Value();

	this->numberOfOutputs = std::stoi(numOutputs, NULL, 0);
}

void XMLInputData::getInputBuffer(float* buffer) {

	int addr;
	tinyxml2::XMLElement* sample = this->doc->FirstChildElement("samples")->FirstChildElement("sample");
	
	if (sample == NULL) {
		std::cout << "The xml file has no tag named sample!" << std::endl;
	}

	//std::cout << "sample has children: " << sample->NoChildren() << std::endl;

	for (unsigned int i = 0; i < this->numberOfSamples; i++) {
	
		tinyxml2::XMLElement* inputNode = sample->FirstChildElement("sampleInput");
		
		if (inputNode == NULL) {
			std::cout << "The sample tag has no sampleInput tag" << std::endl;
		}

		for (unsigned int j = 0; j < this->numberOfInputs; j++) {
			addr = i * this->numberOfInputs + j;
			buffer[addr] = std::stof(inputNode->FirstChild()->ToText()->Value(), NULL);
			inputNode = inputNode->NextSiblingElement("sampleInput");
		}

		sample = sample->NextSiblingElement("sample");
	}
}

void XMLInputData::printInformation() {
	std::cout << "This is a XMLInputData instance." << std::endl;
	std::cout << "There are " << this->numberOfSamples <<" samples in " << this->filePath
		 << " with " << this->numberOfInputs << " inputs and " << this->numberOfOutputs << " outputs." << std::endl;	
}

XMLInputData::~XMLInputData() {
	if (this->doc != NULL) {
		delete this->doc;
		this->doc = NULL;
	}
}
