#include <iostream>
#include "assignment.h"

int main(int argc, char** argv)
{

	std::cout << std::endl;

	Assignment assign (argc, argv);
	//assign.stochasticGradientDescentCPU(60000, 100);

	assign.InitCLContext();
	assign.ReleaseCLContext();

	std::cout << std::endl;

	return(0);
}
