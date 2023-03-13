// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

struct Data
{
	float v1;
	float v2;
};

StructuredBuffer<Data> gInputA : register(t0);
StructuredBuffer<Data> gInputB : register(t1);
RWStructuredBuffer<Data> gOutput : register(u0);

cbuffer ParamBuffer:register(b0) {
	int numLoop;
	int numThread;
};

[numthreads(256, 1, 1)]
void MulAddCS(uint threadID : SV_GroupIndex, uint3 groupID : SV_GroupID)
{
	int index = threadID + groupID.x * numThread;

	min16float c1 = min16float(gOutput[index].v1);
	min16float c2 = min16float(gOutput[index].v2);
	min16float a1 = min16float(gInputA[index].v1);
	min16float a2 = min16float(gInputA[index].v2);
	min16float b1 = min16float(gInputB[index].v1);
	min16float b2 = min16float(gInputB[index].v2);

	int loops = numLoop + 1;

	for (int i = 0; i < loops; ++i) {

		c1 += a1 * b1;
		c2 += a2 * b2;

	}
	gOutput[index].v1 = c1;
	gOutput[index].v2 = c2;
}
