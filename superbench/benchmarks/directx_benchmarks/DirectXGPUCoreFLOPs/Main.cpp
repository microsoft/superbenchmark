// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>
#include "GPUCore.h"


int main(int argc, char* argv[])
{
	Options* opts = new Options(argc, argv);
	auto gpucopy = new GPUCore(opts);
	gpucopy->Run();
}
