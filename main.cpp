#include <iostream>

#include "assignment.h"
#include "CTimer.h"

int main(int argc, char** argv)
{

	CTimer timerCPU;
	CTimer timerGPU;

	std::cout << std::endl;

	Assignment assign (argc, argv);
	
	assign.InitCLContext();

	assign.InitCLResources();

///////////////////////////////////////////////////////////

	assign.zeroDeltaBuffersCPU();
	assign.zeroDeltaBuffersGPU();

	assign.feedForwardCPU(0);
	assign.feedForwardGPU(0, 1);

	assign.gradientDescentCPU(0);
	assign.gradientDescentGPU(0, 1);

///////////////////////////////////////////////////////////

	assign.ReleaseClResources();

	assign.ReleaseCLContext();

	std::cout << std::endl;

	//assign.stochasticGradientDescentCPU(10000, 100);
	
	/*	
	assign.InitCLContext();

	assign.InitCLResources();

	//timerCPU.Start();
	//for (int i = 0; i < 1000; i++) {
		assign.feedForwardCPU(0);
	//}
	//timerCPU.Stop();
	//std::cout << "GPU time: " << timerCPU.GetElapsedMilliseconds() << std::endl;
	

	//timerGPU.Start();
	//for (int i = 0; i < 1000; i++) {
		assign.feedForwardGPU(0, 50);
	//}
	//timerGPU.Stop();
	//std::cout << "GPU time: " << timerGPU.GetElapsedMilliseconds() << std::endl;
	
	//std::cout << "speedup: " << timerCPU.GetElapsedMilliseconds()/timerGPU.GetElapsedMilliseconds() << std::endl;

	assign.backPropagationCPU(0);

	assign.zeroDeltaBuffers();

	assign.gradientDescentGPU(0, 50);

	assign.ReleaseClResources();

	assign.ReleaseCLContext();

	std::cout << std::endl;
	*/
	return(0);
}
