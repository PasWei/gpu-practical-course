// It is presumed that the image is stored in memory as 
//   RGB_t data[ height ][ width ]
// where lines are top to bottom and columns are left to right
// (the same way you view the image on the display)

// The routine makes all the appropriate adjustments to match the TGA format specification.
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>

#include "tgawriter.h"

using namespace std;

namespace tgawriter {
	bool write_truecolor_tga(const std::string& filename, RGB_t* data, unsigned int width, unsigned int height) {
		ofstream tgafile( filename.c_str(), ios::binary );
		if (!tgafile) return false;

		// The image header
		unsigned char header[ 18 ] = { 0 };
		header[  2 ] = 2;  // truecolor
		header[ 12 ] =  width        & 0xFF;
		header[ 13 ] = (width  >> 8) & 0xFF;
		header[ 14 ] =  height       & 0xFF;
		header[ 15 ] = (height >> 8) & 0xFF;
		header[ 16 ] = 24;  // bits per pixel

		tgafile.write((const char*)header, 18);

		// The image data is stored bottom-to-top, left-to-right
		for (int y = height -1; y >= 0; y--) {
			for (int x = 0; (unsigned int) x < width; x++)
			{
				tgafile.put( (char)data[ (y * width) + x ].blue  );
				tgafile.put( (char)data[ (y * width) + x ].green );
				tgafile.put( (char)data[ (y * width) + x ].red   );
			}
		}

		// The file footer (tga 2.0)
		static const char footer[ 26 ] = "\0\0\0\0\0\0\0\0TRUEVISION-XFILE.";
		tgafile.write( footer, 26 );

		tgafile.close();
		return true;
	}
}
