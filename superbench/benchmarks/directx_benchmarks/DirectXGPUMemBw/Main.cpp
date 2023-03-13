// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>

#include "GPUMemRwBw.h"

int main(int argc, char* argv[])
{
	Options* opts = new Options(argc, argv);
	for (SIZE_T usize = opts->maxbytes; usize >= opts->minbytes; usize /= 2) {
        std::cout << "copy_size " << usize << ", ";
		auto gpucopy = new GPUMemRwBw(opts, usize);
		gpucopy->Run();
	}
}
