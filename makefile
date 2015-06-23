#make sure to list tinyxml2 after xmlInputDataClass to avoid compiler errors
SOURCE_FILES=main.cpp assignment_cmd.cpp tgawriter.o binaryInputDataClass.o xmlInputDataClass.o tinyxml2.o
COMPILER_ARGS=-std=c++11 -Wall -B ./ -I ./include/tgawriter -I ./include/inputData -I ./include/tinyxml2
OUTPUT_NAME=assignment

$(OUTPUT_NAME): makefile ./include/tclap/CmdLine.h $(SOURCE_FILES)
	g++ $(COMPILER_ARGS) -o $(OUTPUT_NAME) $(SOURCE_FILES)

#compile the input data classes
BINARY_INPUT_DATA_CLASS_COMPILER_ARGS =-std=c++11 -Wall -c
binaryInputDataClass.o: ./include/inputData/binaryInputData.cpp ./include/inputData/binaryInputData.h ./include/inputData/inputData.h
	g++ $(BINARY_INPUT_DATA_CLASS_COMPILER_ARGS) -o $@ $<

BINARY_INPUT_DATA_CLASS_COMPILER_ARGS =-std=c++11 -Wall -c -I ./include/tinyxml2
xmlInputDataClass.o: ./include/inputData/xmlInputData.cpp ./include/inputData/inputData.h ./include/inputData/xmlInputData.h
	g++ $(BINARY_INPUT_DATA_CLASS_COMPILER_ARGS) -o $@ $<

#compile the little tga module
#$@ is the name of the target, $^ is the list of dependencies, $< is the first dependency
TGA_COMPILER_ARGS =-std=c++11 -Wall -c 
tgawriter.o: ./include/tgawriter/tgawriter.cpp ./include/tgawriter/tgawriter.h
	g++ $(TGA_COMPILER_ARGS) -o $@ $<

#compile the tinyxml library
TINYXML2_COMPILER_FLAGS=-std=c++11 -Wall -c 
tinyxml2.o: ./include/tinyxml2/tinyxml2.cpp ./include/tinyxml2/tinyxml2.h
	g++ $(TINYXML2_COMPILER_FLAGS) -o $@ $<
	
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

./data/xor.xml:
	echo "<?xml version=\"1.0\"?>" > ./data/xor.xml
	echo "<samples>" >> ./data/xor.xml
	echo "	<sampleCount>4</sampleCount>" >> ./data/xor.xml
	echo "	<sample>" >> ./data/xor.xml
	echo "		<sampleInput>0</sampleInput>" >> ./data/xor.xml
	echo "		<sampleInput>0</sampleInput>" >> ./data/xor.xml
	echo "		<sampleOutput>0</sampleOutput>" >> ./data/xor.xml
	echo "	</sample>" >> ./data/xor.xml
	echo "	<sample>" >> ./data/xor.xml
	echo "		<sampleInput>1</sampleInput>" >> ./data/xor.xml
	echo "		<sampleInput>0</sampleInput>" >> ./data/xor.xml
	echo "		<sampleOutput>1</sampleOutput>" >> ./data/xor.xml
	echo "	</sample>" >> ./data/xor.xml
	echo "	<sample>" >> ./data/xor.xml
	echo "		<sampleInput>0</sampleInput>" >> ./data/xor.xml
	echo "		<sampleInput>1</sampleInput>" >> ./data/xor.xml
	echo "		<sampleOutput>1</sampleOutput>" >> ./data/xor.xml
	echo "	</sample>" >> ./data/xor.xml
	echo "	<sample>" >> ./data/xor.xml
	echo "		<sampleInput>1</sampleInput>" >> ./data/xor.xml
	echo "		<sampleInput>1</sampleInput>" >> ./data/xor.xml
	echo "		<sampleOutput>0</sampleOutput>" >> ./data/xor.xml
	echo "	</sample>" >> ./data/xor.xml
	echo "</samples>" >> ./data/xor.xml

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
.PHONY: clean clean_prog

clean_prog:
	rm -f $(OUTPUT_NAME)
	rm -f tgawriter.o
	rm -f binaryInputDataClass.o
	rm -f xmlInputDataClass.o
	rm -f tinyxml2.o	

clean:
	rm -f $(OUTPUT_NAME)
	rm -fr include/tclap
	rm -fr data
	rm -f tgawriter.o
	rm -f binaryInputDataClass.o
	rm -f xmlInputDataClass.o
	rm -f tinyxml2.o	
