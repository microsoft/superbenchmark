// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

StructuredBuffer<float> gInputA : register(t0);
RWStructuredBuffer<float> gOutput : register(u0);

cbuffer ParamBuffer : register(b0) {
	int numLoop;
	uint3 numThreads;
	uint3 numDispatch;
};

[numthreads(numThreads.x, numThreads.y , numThreads.z)]
void Read(uint threadID : SV_GroupIndex, uint3 groupID : SV_GroupID, uint3 dispatchId : SV_DispatchThreadID)
{
	uint idStart = dispatchId.x +
		dispatchId.y * numDispatch.x * numThread.x +
		dispatchId.z * numDispatch.x * numThread.x * numDispatch.y * numThread.y;

	uint i = idStart;
	uint start = idStart * numLoop;
	uint end = start + numLoop;
	i = start;
	for (; i < end; i++)
	{
		float c = gOutput[i].v1;
		if (c == -1)
		{
			// This condition should never access since gOutput init as zero.
			// It is for avoid compile optimization.
			gOutput[i].v1 = 0;
		}
	}
}

[numthreads(numThreads.x, numThreads.y , numThreads.z)]
void Write(uint threadID : SV_GroupIndex, uint3 groupID : SV_GroupID, uint3 dispatchId : SV_DispatchThreadID)
{
	uint idStart = dispatchId.x +
		dispatchId.y * numDispatch.x * numThread.x +
		dispatchId.z * numDispatch.x * numThread.x * numDispatch.y * numThread.y;
	uint i = idStart;

	uint start = idStart * numLoop;
	uint end = start + numLoop;
	i = start;
	for (; i < end; i++)
	{
		gOutput[i].v1 =  i % 256;
	}
}

[numthreads(numThreads.x, numThreads.y , numThreads.z)]
void ReadWrite(uint threadID : SV_GroupIndex, uint3 groupID : SV_GroupID, uint3 dispatchId : SV_DispatchThreadID)
{
	uint idStart = dispatchId.x +
		dispatchId.y * numDispatch.x * numThread.x +
		dispatchId.z * numDispatch.x * numThread.x * numDispatch.y * numThread.y;
	uint i = idStart;

	uint start = idStart * numLoop;
	uint end = start + numLoop;
	i = start;
	for (; i < end; i++)
	{
		float c = gInputA[i].v1;
		gOutput[i].v1 = c;
	}
}
