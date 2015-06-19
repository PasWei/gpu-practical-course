#include <fstream>
#include <string>

namespace tgawriter {
	typedef struct {
		unsigned char red, green, blue;
	} RGB_t;

	bool write_truecolor_tga(const std::string& filename, RGB_t* data, unsigned int width, unsigned int height);
}
