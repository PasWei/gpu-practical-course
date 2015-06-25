#include <iostream>
#include "assignment.h"

int main(int argc, char** argv)
{

	std::cout << std::endl;

	Assignment assign (argc, argv);
	assign.stochasticGradientDescent(1000);

	std::cout << std::endl;

	return(0);
}
