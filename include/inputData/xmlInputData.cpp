#include <iostream>

#include "xmlInputData.h"

XMLInputData::XMLInputData(const std::string filePath) {

	this->doc = NULL;

	this->filePath = filePath;
	this->doc = new tinyxml2::XMLDocument;
	this->doc->LoadFile(filePath.c_str());
}

void XMLInputData::printInformation() {
	tinyxml2::XMLText* textNode = doc->FirstChildElement("samples")->FirstChildElement("sampleCount")->FirstChild()->ToText();
	std::string numSamples = textNode->Value();
	std::cout << "This is a XMLInputData instance." << std::endl;
	std::cout << "There are " << numSamples << " samples in this file." << std::endl;	
}

XMLInputData::~XMLInputData() {
	if (this->doc != NULL) {
		delete this->doc;
		this->doc = NULL;
	}
}
