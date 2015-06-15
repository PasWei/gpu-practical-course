SOURCE_FILES=converter.cpp
COMPILER_ARGS=-std=c++11 -Wall -B ./
OUTPUT_NAME=converter

all: ./include/tclap/CmdLine.h $(SOURCE_FILES)
	g++ $(COMPILER_ARGS) -o $(OUTPUT_NAME) $(SOURCE_FILES)

# downloads TCLAP and puts the files in directory "include/tclap"
# TCLAP is a small, flexible library that provides a simple interface for defining and accessing command line arguments
./include/tclap/CmdLine.h:
	wget http://netcologne.dl.sourceforge.net/project/tclap/tclap-1.2.1.tar.gz
	tar -zxf tclap-1.2.1.tar.gz
	mkdir -p include
	cp -r tclap-1.2.1/include/tclap include/tclap
	rm tclap-1.2.1.tar.gz
	rm -r tclap-1.2.1
	

.PHONY: clean

clean:
	rm -f $(OUTPUT_NAME)
	rm -rf include 
