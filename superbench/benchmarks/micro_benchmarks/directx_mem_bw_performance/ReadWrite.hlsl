// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

StructuredBuffer<float> gInputA : register(t0);
RWStructuredBuffer<float> gOutput : register(u0);

cbuffer ParamBuffer : register(b0) {
	int numLoop;
	uint3 numThreads;
	uint3 numDispatch;
};

[numthreads(X, Y, Z)]
void Read(uint threadID : SV_GroupIndex, uint3 groupID : SV_GroupID, uint3 dispatchId : SV_DispatchThreadID)
{
	uint idStart = dispatchId.x +
		dispatchId.y * numDispatch.x * numThreads.x +
		dispatchId.z * numDispatch.x * numThreads.x * numDispatch.y * numThreads.y;

	uint start = idStart * numLoop;
	uint end = start + numLoop;
	for (uint i = start; i < end; i++)
	{
		float c = gOutput[i];
		if (c == -1)
		{
			// This condition should never access since gOutput init as zero.
			// It is for avoid compile optimization.
			gOutput[i] = 0;
		}
	}
}

[numthreads(X, Y, Z)]
void Write(uint threadID : SV_GroupIndex, uint3 groupID : SV_GroupID, uint3 dispatchId : SV_DispatchThreadID)
{
	uint idStart = dispatchId.x +
		dispatchId.y * numDispatch.x * numThreads.x +
		dispatchId.z * numDispatch.x * numThreads.x * numDispatch.y * numThreads.y;

	uint start = idStart * numLoop;
	uint end = start + numLoop;
	for (uint i = start; i < end; i++)
	{
		gOutput[i] =  i % 256;
	}
}

[numthreads(X, Y, Z)]
void ReadWrite(uint threadID : SV_GroupIndex, uint3 groupID : SV_GroupID, uint3 dispatchId : SV_DispatchThreadID)
{
	uint idStart = dispatchId.x +
		dispatchId.y * numDispatch.x * numThreads.x +
		dispatchId.z * numDispatch.x * numThreads.x * numDispatch.y * numThreads.y;

	uint start = idStart * numLoop;
	uint end = start + numLoop;
	for (uint i = start; i < end; i++)
	{
		gOutput[i] = gInputA[i];
	}
}
