#pragma once

#include <stdint.h>
#include <string>

#include "inputData.h"
#include "tinyxml2.h"

class XMLInputData : public InputData {
	
	private:

		//path to the xml file
		std::string filePath;

		//xml document object
		tinyxml2::XMLDocument* doc;
	
	public:
		
		XMLInputData(const std::string filePath);
		~XMLInputData();
		void printInformation();

};
