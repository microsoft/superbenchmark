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

	float c1 = gOutput[index].v1;
	float c2 = gOutput[index].v2;
	float a1 = gInputA[index].v1;
	float a2 = gInputA[index].v2;
	float b1 = gInputB[index].v1;
	float b2 = gInputB[index].v2;

	int loops = numLoop + 1;

	for (int i = 0; i < loops; ++i) {

		c1 += a1 * b1;
		c2 += a2 * b2;

	}
	gOutput[index].v1 = c1;
	gOutput[index].v2 = c2;
}
