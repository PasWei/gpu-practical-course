#include <fstream>
#include <string>

typedef unsigned char byte;
typedef struct {
	byte red, green, blue;
} RGB_t;

bool write_truecolor_tga( const std::string& filename, RGB_t* data, unsigned width, unsigned height);
