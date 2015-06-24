#include <iostream>
#include "assignment.h"

int main(int argc, char** argv)
{

	std::cout << std::endl;

	Assignment assign (argc, argv);

	for (int i = 0; i < 1; i++) {
		assign.feedForwardCPU(0);
	}

	std::cout << std::endl;

	return(0);
}
