// Scene cube.
TextureCube gCubeMap : register(t0);

// An array of textures, which is only supported in shader model 5.1+.  Unlike Texture2DArray, the textures
// in this array can be different sizes and formats, making it more flexible than texture arrays.
Texture2D gTextureMaps[TEXTURECOUNT] : register(t1);


SamplerState gsamAnisotropicWrap : register(s0);


// Constant data that varies per frame.
cbuffer D3DObjectConstantBuffer : register(b0)
{
    float4x4 gWorld;
    float4x4 gTexTransform;
    uint gMaterialIndex;
};

// Constant data that varies per material.
cbuffer PassConstantBuffer : register(b1)
{
    float4x4 gView;
    float4x4 gViewProj;
    float3 gEyePosW;
};

cbuffer MaterialDataConstantBuffer : register(b2)
{
    float4 DiffuseAlbedo;
    float3 FresnelR0;
    float Roughness;
    float4x4 MatTransform;
    uint DiffuseMapIndex;
    uint NormalMapIndex;
};


//---------------------------------------------------------------------------------------
// Transforms a normal map sample to world space.
//---------------------------------------------------------------------------------------
float3 NormalSampleToWorldSpace(float3 normalMapSample, float3 unitNormalW, float3 tangentW)
{
	// Transform from [0,1] to [-1,1].
    float3 normalT = 2.0f * normalMapSample - 1.0f;

    float3 N = unitNormalW;
    float3 T = normalize(tangentW - dot(tangentW, N) * N);
    float3 B = cross(N, T);

    float3x3 TBN = float3x3(T, B, N);

	// Trans to world space.
    float3 bumpedNormalW = mul(normalT, TBN);

    return bumpedNormalW;
}


struct VertexIn
{
	float3 PosL    : POSITION;
    float3 NormalL : NORMAL;
	float2 TexC    : TEXCOORD;
	float3 TangentU : TANGENT;
};

struct VertexOut
{
	float4 PosH    : SV_POSITION;
    float3 PosW    : POSITION;
    float3 NormalW : NORMAL;
	float3 TangentW : TANGENT;
	float2 TexC    : TEXCOORD;
};

struct PixelOut
{
    float4 position : SV_Target0;
    float4 normal : SV_Target1;
    float4 color : SV_Target2;
};

float3 SchlickFresnel(float3 R0, float3 normal, float3 lightVec)
{
    float cosIncidentAngle = saturate(dot(normal, lightVec));

    float f0 = 1.0f - cosIncidentAngle;
    float3 reflectPercent = R0 + (1.0f - R0) * (f0 * f0 * f0 * f0 * f0);

    return reflectPercent;
}

VertexOut VS(VertexIn vin)
{
    VertexOut vout = (VertexOut)0.0f;

    float4 posW = mul(float4(vin.PosL, 1.0f), gWorld);
    vout.PosW = posW.xyz;
    vout.PosH = mul(posW, gViewProj);
    vout.NormalW = mul(vin.NormalL, (float3x3) gWorld);
    vout.TangentW = mul(vin.TangentU, (float3x3) gWorld);
    float4 texC = mul(float4(vin.TexC, 0.0f, 1.0f), gTexTransform);
    vout.TexC = mul(texC, MatTransform).xy;

    return vout;
}

PixelOut PS(VertexOut pin)
{
    // Normalize normap map.
    pin.NormalW = normalize(pin.NormalW);
    
    float4 normalSample = gTextureMaps[NormalMapIndex].Sample(gsamAnisotropicWrap, pin.TexC);
    float3 bumpedNormalW = NormalSampleToWorldSpace(normalSample.xyz, pin.NormalW, pin.TangentW);
    float4 diffuseAlbedo = DiffuseAlbedo * 
    gTextureMaps[DiffuseMapIndex].Sample(gsamAnisotropicWrap, pin.TexC);
    
    const float shininess = (1.0f - Roughness) * normalSample.a;

    PixelOut pout;
    pout.position = float4(pin.PosW, FresnelR0.x);
    pout.normal = float4(bumpedNormalW, shininess);
    
    float3 toEyeW = normalize(gEyePosW - pin.PosW);
    float3 ref = reflect(-toEyeW, bumpedNormalW);
    float4 reflectColor = gCubeMap.Sample(gsamAnisotropicWrap, ref);
    float3 fresnelFactor = SchlickFresnel(FresnelR0, bumpedNormalW, ref);
    pout.color = float4(diffuseAlbedo.xyz + shininess * fresnelFactor * reflectColor.xyz, 1.0f);
    
    return pout;
}


