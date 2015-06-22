SOURCE_FILES=main.cpp assignment_cmd.cpp tgawriter.o inputDataClass.o
COMPILER_ARGS=-std=c++11 -Wall -B ./ -I ./include/tgawriter -I ./include/inputData
OUTPUT_NAME=assignment

$(OUTPUT_NAME): makefile ./include/tclap/CmdLine.h $(SOURCE_FILES)
	g++ $(COMPILER_ARGS) -o $(OUTPUT_NAME) $(SOURCE_FILES)

#compile the input data classes
INPUT_DATA_CLASS_COMPILER_ARGS =-std=c++11 -Wall -c
inputDataClass.o: ./include/inputData/binaryInputData.cpp ./include/inputData/binaryInputData.h ./include/inputData/inputData.h
	g++ $(INPUT_DATA_CLASS_COMPILER_ARGS) -o $@ ./include/inputData/binaryInputData.cpp 

#compile the little tga module
#$@ is the name of the target, $^ is the list of dependencies
TGA_COMPILER_ARGS =-std=c++11 -Wall -c 
tgawriter.o: ./include/tgawriter/tgawriter.cpp ./include/tgawriter/tgawriter.h
	g++ $(TGA_COMPILER_ARGS) -o $@ $^

#downloads tinyxml2 and puts the files in director "include/tinyxml2"
#TinyXML-2 is a simple, small, efficient, C++ XML parser that can be easily integrated into other programs.
./include/tinyxml2/tinyxml2.h:
	wget -q https://raw.githubusercontent.com/leethomason/tinyxml2/master/tinyxml2.h
	wget -q https://raw.githubusercontent.com/leethomason/tinyxml2/master/tinyxml2.cpp
	mkdir -p ./include/tinyxml2
	mv tinyxml2.h ./include/tinyxml2
	mv tinyxml2.cpp ./include/tinyxml2

# downloads TCLAP and puts the files in directory "include/tclap"
# TCLAP is a small, flexible library that provides a simple interface for defining and accessing command line arguments
./include/tclap/CmdLine.h:
	wget -q http://netcologne.dl.sourceforge.net/project/tclap/tclap-1.2.1.tar.gz
	tar -zxf tclap-1.2.1.tar.gz
	mkdir -p include
	cp -r tclap-1.2.1/include/tclap include/tclap
	rm tclap-1.2.1.tar.gz
	rm -r tclap-1.2.1

data: ./data/train-images-idx3-ubyte ./data/train-labels-idx1-ubyte ./data/t10k-images-idx3-ubyte ./data/t10k-labels-idx1-ubyte

# downloads the MNIST data set and puts the files in directory "data"
./data/train-images-idx3-ubyte:
	wget -q http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	gunzip train-images-idx3-ubyte.gz
	mkdir -p data
	mv train-images-idx3-ubyte ./data

./data/train-labels-idx1-ubyte:
	wget -q http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	gunzip train-labels-idx1-ubyte.gz
	mkdir -p data
	mv train-labels-idx1-ubyte ./data

./data/t10k-images-idx3-ubyte:
	wget -q http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	gunzip t10k-images-idx3-ubyte.gz
	mkdir -p data
	mv t10k-images-idx3-ubyte ./data

./data/t10k-labels-idx1-ubyte:
	wget -q http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	gunzip t10k-labels-idx1-ubyte.gz
	mkdir -p data
	mv t10k-labels-idx1-ubyte ./data

#always execute clean
.PHONY: clean

clean:
	rm -f $(OUTPUT_NAME)
	rm -fr include/tclap
	rm -fr data
	rm -f tgawriter.o
	rm -f inputDataClass.o
