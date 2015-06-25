#include <iostream>
#include "assignment.h"

int main(int argc, char** argv)
{

	std::cout << std::endl;

	Assignment assign (argc, argv);

	for (int i = 0; i < 10; i++) {
		std::cout << "CrossEntropy = " << assign.feedForwardCPU(0) << std::endl;
		assign.backPropagationCPU(0);
	}

	std::cout << std::endl;

	return(0);
}
