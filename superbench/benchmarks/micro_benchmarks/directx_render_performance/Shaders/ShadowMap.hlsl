
cbuffer ObjectConstantBuffer : register(b0)
{
    float4x4 gWorld;
    float4x4 gViewProj;
};

struct VertexIn
{
    float3 PosL : POSITION;
    float2 TexC : TEXCOORD0;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
};

void SetShadowDepthOutputs(float4 WorldPosition, float4x4 gViewProj, out float4 OutPosition, out float ShadowDepth)
{
    // Transform the vertex position from world to view
    OutPosition = mul(WorldPosition, gViewProj);

    float DepthBias = 0.01;
    float InvMaxSubjectDepth = 0.001;

    // Output linear, normalized depth
    ShadowDepth = OutPosition.z * InvMaxSubjectDepth + DepthBias;
    OutPosition.z = ShadowDepth * OutPosition.w;
}

// Generate depth info from the view of light.
VertexOut VS(VertexIn vin)
{
    VertexOut vout;

    // Transform the vertex position from object / local space to world space
    float4 WorldPos = mul(float4(vin.PosL, 1.0), gWorld);

    float dummy; // Corrected the variable name

    SetShadowDepthOutputs(
        WorldPos,
        gViewProj,
        vout.PosH,
        dummy
    );

    return vout;
}

void PS(VertexOut pin)
{
    // Pixel shader implementation goes here
}
