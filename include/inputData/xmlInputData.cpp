#include <iostream>

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
