SOURCE_FILES=assignment.cpp assignment.h ./include/tgawriter/tgawriter.h ./include/tgawriter/tgawriter.cpp
COMPILER_ARGS=-std=c++11 -Wall -B ./
OUTPUT_NAME=assignment

$(OUTPUT_NAME): makefile ./include/tclap/CmdLine.h $(SOURCE_FILES)
	g++ $(COMPILER_ARGS) -o $(OUTPUT_NAME) $(SOURCE_FILES)

# downloads TCLAP and puts the files in directory "include/tclap"
# TCLAP is a small, flexible library that provides a simple interface for defining and accessing command line arguments
./include/tclap/CmdLine.h:
	wget -q http://netcologne.dl.sourceforge.net/project/tclap/tclap-1.2.1.tar.gz
	tar -zxf tclap-1.2.1.tar.gz
	mkdir -p include
	cp -r tclap-1.2.1/include/tclap include/tclap
	rm tclap-1.2.1.tar.gz
	rm -r tclap-1.2.1

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
	rm -rf include/tclap
	rm -rf data