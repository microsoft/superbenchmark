

// Constant buffers
cbuffer ViewConstantBuffer : register(b1)  
{  
    float4 View_InvDeviceZToWorldZTransform;  
    float4 View_TemporalAAParams;  
    float4 View_BufferSizeAndInvSize;  
    float4 View_DiffuseOverrideParameter;  
    float4 View_SpecularOverrideParameter;  
    float4x4 View_ClipToView;  
    float4x4 View_ViewToClip;  
    float4x4 View_ScreenToWorld;  
    float3 View_WorldCameraOrigin;
    float Padding0;
    float3 View_PreViewTranslation;  
    float Padding1;
    float4 View_ScreenPositionScaleBias;  
    float4x4 View_TranslatedWorldToClip;  
    uint View_StateFrameIndexMod8;  
    float3 Padding; // Add padding to maintain 16-byte alignment  
};  



cbuffer DeferredLightUniformsConstantBuffer : register(b2)  
{  
    float4 DeferredLightUniforms_ShadowMapChannelMask;  
    float2 DeferredLightUniforms_DistanceFadeMAD;  
    float DeferredLightUniforms_ContactShadowLength;  
    float DeferredLightUniforms_VolumetricScatteringIntensity;  
    uint DeferredLightUniforms_ShadowedBits;  
    uint DeferredLightUniforms_LightingChannelMask;  
    float3 DeferredLightUniforms_Position;  
    float DeferredLightUniforms_InvRadius;  
    float3 DeferredLightUniforms_Color;  
    float DeferredLightUniforms_FalloffExponent;  
    float3 DeferredLightUniforms_Direction;  
    float DeferredLightUniforms_SpecularScale;  
    float3 DeferredLightUniforms_Tangent;  
    float DeferredLightUniforms_SourceRadius;  
    float2 DeferredLightUniforms_SpotAngles;  
    float DeferredLightUniforms_SoftSourceRadius;  
    float DeferredLightUniforms_SourceLength;  
    float DeferredLightUniforms_RectLightBarnCosAngle;  
    float DeferredLightUniforms_RectLightBarnLength;  
};  

cbuffer ShadowConstantBuffer : register(b3)
{
    float4 LightPositionAndInvRadius;
    float4 PointLightDepthBiasAndProjParameters;
    float4x4 ShadowViewProjectionMatrices[6]; 
    float ShadowSharpen;
    float ShadowFadeFraction;
    float InvShadowmapResolution;
}



// Texture declarations  
Texture2D<float4> SceneTexturesStruct_SceneDepthTexture: register(t0);
Texture2D<float4> SceneTexturesStruct_GBufferATexture : register(t1);
Texture2D<float4> SceneTexturesStruct_GBufferBTexture : register(t2);
Texture2D<float4> SceneTexturesStruct_GBufferCTexture : register(t3);
Texture2D<float4> SceneTexturesStruct_GBufferDTexture : register(t4);
Texture2D<float4> SceneTexturesStruct_GBufferETexture : register(t5);
Texture2D<float4> SceneTexturesStruct_ScreenSpaceAOTexture : register(t6);
Texture2D<float4> LightAttenuationTexture: register(t7);
Texture2D<float4> SceneTexturesStruct_CustomDepthTexture : register(t8);
Texture2D<float4> ShadowDepthCubeTexture : register(t9);



// Sampler declarations (assuming sampler registers are in the same register space)  
SamplerState SceneTexturesStruct_SceneDepthTextureSampler : register(s0);
SamplerState SceneTexturesStruct_GBufferATextureSampler : register(s1);
SamplerState SceneTexturesStruct_GBufferBTextureSampler : register(s2);
SamplerState SceneTexturesStruct_GBufferCTextureSampler : register(s3);
SamplerState SceneTexturesStruct_GBufferDTextureSampler : register(s4);
SamplerState SceneTexturesStruct_GBufferETextureSampler : register(s5);
SamplerState SceneTexturesStruct_ScreenSpaceAOTextureSampler : register(s6);
SamplerState LightAttenuationTextureSampler : register(s7);
SamplerState SceneTexturesStruct_CustomDepthTextureSampler : register(s8);
SamplerComparisonState ShadowDepthCubeTextureSampler : register(s9);

const static float PI = 3.1415926535897932f;
const static float MaxHalfFloat = 65504.0f;

struct FLightAccumulator {
	float3 Diffuse;
	float3 Specular;
	float3 Transmission;
	float EstimatedCost;
	float3 TotalLight;
};

struct FGBufferData {
	float3 WorldNormal;
	float PerObjectGBufferData;
	float Metallic;
	float Specular;
	float Roughness;
	uint ShadingModelID;
	uint SelectiveOutputMask;
	float3 BaseColor;
	float GBufferAO;
	float IndirectIrradiance;
	float4 CustomData;
	float4 PrecomputedShadowFactors;
	float CustomDepth;
	uint CustomStencil;
	float Depth;
	float3 StoredBaseColor;
	float StoredMetallic;
	float StoredSpecular;
	float3 SpecularColor;
	float3 DiffuseColor;
	float4 Velocity;
};


struct FDeferredLightData {
	float3 Position;
	float InvRadius;
	float3 Color;
	float FalloffExponent;
	float3 Direction;
	float3 Tangent;
	float2 SpotAngles;
	float SourceRadius;
	float SourceLength;
	float SoftSourceRadius;
	float SpecularScale;
	float ContactShadowLength;
	bool ContactShadowLengthInWS;
	float2 DistanceFadeMAD;
	float4 ShadowMapChannelMask;
	uint ShadowedBits;
	bool bInverseSquared;
	bool bRadialLight;
	bool bSpotLight;
	bool bRectLight;
	float RectLightBarnCosAngle;
	float RectLightBarnLength;
};


struct FShadowTerms {
	float SurfaceShadow;
	float TransmissionShadow;
	float TransmissionThickness;
};

struct FDirectLighting {
	float3 Diffuse;
	float3 Specular;
	float3 Transmission;
};

struct FRectTexture {
	float Dummy;
};

struct FCapsuleLight {
	float Length;
	float Radius;
	float SoftRadius;
	float DistBiasSqr;
	float3 LightPos[2];
};

struct FRect {
	float Dummy;
};


struct FAreaLight
{
	float SphereSinAlpha;
	float SphereSinAlphaSoft;
	float LineCosSubtended;
	float FalloffColor;
	FRect Rect; // Assuming FRect is a custom struct representing a rectangle  
	bool bIsRect;
	FRectTexture Texture;
};


struct BxDFContext {
	float NoL;  // Normal dot Light
	float NoV;  // Normal dot View
	float VoL;  // View dot Light
	float NoH;  // Normal dot Half
	float VoH;  // View dot Half
};

struct FScreenSpaceData {
	FGBufferData GBuffer;
	float AmbientOcclusion;
};

Texture2D DummyRectLightTextureForCapsuleCompilerWarning;
Texture2D DeferredLightUniforms_SourceTexture;

FLightAccumulator LightAccumulator_Init() {
	FLightAccumulator acc;
	acc.TotalLight = float3(0, 0, 0);
	acc.EstimatedCost = 0;
	return acc;
}

FRectTexture InitRectTexture(Texture2D SourceTexture) {
	FRectTexture Output;
	Output.Dummy = 0;

	return Output;
}

float4 Texture2DSampleLevel(Texture2D Tex, SamplerState Sampler, float2 UV,
	float Mip) {
	return Tex.SampleLevel(Sampler, UV, Mip);
}

float ConvertFromDeviceZ(float DeviceZ) {

	return DeviceZ * View_InvDeviceZToWorldZTransform[0] +
		View_InvDeviceZToWorldZTransform[1] +
		1.0f / (DeviceZ * View_InvDeviceZToWorldZTransform[2] -
			View_InvDeviceZToWorldZTransform[3]);
}

float CalcSceneDepth(float2 ScreenUV) {

	return ConvertFromDeviceZ(
		Texture2DSampleLevel(SceneTexturesStruct_SceneDepthTexture,
			SceneTexturesStruct_SceneDepthTextureSampler,
			ScreenUV, 0)
		.r);
}


bool CheckerFromPixelPos(uint2 PixelPos) {

	uint TemporalAASampleIndex = View_TemporalAAParams.x;

	return (PixelPos.x + PixelPos.y + TemporalAASampleIndex) % 2;
}

bool UseSubsurfaceProfile(int ShadingModel) {
	return ShadingModel == 5 || ShadingModel == 9;
}

bool CheckerFromSceneColorUV(float2 UVSceneColor) {

	uint2 PixelPos = uint2(UVSceneColor * View_BufferSizeAndInvSize.xy);

	return CheckerFromPixelPos(PixelPos);
}

float3 DecodeNormal(float3 N) { return N * 2 - 1; }

uint DecodeShadingModelId(float InPackedChannel) {
	return ((uint)round(InPackedChannel * (float)0xFF)) & 0xF;
}

uint DecodeSelectiveOutputMask(float InPackedChannel) {
	return ((uint)round(InPackedChannel * (float)0xFF)) & ~0xF;
}

float3 DecodeBaseColor(float3 BaseColor) { return BaseColor; }

float DecodeIndirectIrradiance(float IndirectIrradiance) {

	const float OneOverPreExposure = 1.f;

	float LogL = IndirectIrradiance;
	const float LogBlackPoint = 0.00390625;
	return OneOverPreExposure * (exp2(LogL * 16 - 8) - LogBlackPoint);
}

float DielectricSpecularToF0(float Specular) { return 0.08f * Specular; }

float Lerp(float a, float b, float t) { return a + (b - a) * t; }

float3 ComputeF0(float Specular, float3 BaseColor, float Metallic) {
	float4 F0 = DielectricSpecularToF0(Specular);
	return lerp(F0.xxx, BaseColor, Metallic.xxx);
}

FGBufferData DecodeGBufferData(float4 InGBufferA, float4 InGBufferB,
	float4 InGBufferC, float4 InGBufferD,
	float4 InGBufferE, float4 InGBufferVelocity,
	float CustomNativeDepth, uint CustomStencil,
	float SceneDepth, bool bGetNormalizedNormal,
	bool bChecker) {
	FGBufferData GBuffer;

	GBuffer.WorldNormal = DecodeNormal(InGBufferA.xyz);
	if (bGetNormalizedNormal) {
		GBuffer.WorldNormal = normalize(GBuffer.WorldNormal);
	}

	GBuffer.PerObjectGBufferData = InGBufferA.a;
	GBuffer.Metallic = InGBufferB.r;
	GBuffer.Specular = InGBufferB.g;
	GBuffer.Roughness = InGBufferB.b;

	GBuffer.ShadingModelID = DecodeShadingModelId(InGBufferB.a);
	GBuffer.SelectiveOutputMask = DecodeSelectiveOutputMask(InGBufferB.a);

	GBuffer.BaseColor = DecodeBaseColor(InGBufferC.rgb);

	GBuffer.GBufferAO = 1;
	GBuffer.IndirectIrradiance = DecodeIndirectIrradiance(InGBufferC.a);

	GBuffer.CustomData =
		!(GBuffer.SelectiveOutputMask & (1 << 4)) ? InGBufferD : 0;

	GBuffer.PrecomputedShadowFactors =
		!(GBuffer.SelectiveOutputMask & (1 << 5))
		? InGBufferE
		: ((GBuffer.SelectiveOutputMask & (1 << 6)) ? 0 : 1);
	GBuffer.CustomDepth = ConvertFromDeviceZ(CustomNativeDepth);
	GBuffer.CustomStencil = CustomStencil;
	GBuffer.Depth = SceneDepth;

	GBuffer.StoredBaseColor = GBuffer.BaseColor;
	GBuffer.StoredMetallic = GBuffer.Metallic;
	GBuffer.StoredSpecular = GBuffer.Specular;

	[flatten] if (GBuffer.ShadingModelID == 9) { GBuffer.Metallic = 0.0; }

	{
		GBuffer.SpecularColor =
			ComputeF0(GBuffer.Specular, GBuffer.BaseColor, GBuffer.Metallic);

		GBuffer.DiffuseColor =
			GBuffer.BaseColor - GBuffer.BaseColor * GBuffer.Metallic;

		{

			GBuffer.DiffuseColor =
				GBuffer.DiffuseColor * View_DiffuseOverrideParameter.www +
				View_DiffuseOverrideParameter.xyz;
			GBuffer.SpecularColor =
				GBuffer.SpecularColor * View_SpecularOverrideParameter.w +
				View_SpecularOverrideParameter.xyz;
		}
	}

	GBuffer.Velocity =
		!(GBuffer.SelectiveOutputMask & (1 << 7)) ? InGBufferVelocity : 0;

	return GBuffer;
}

FGBufferData GetGBufferData(float2 UV, bool bGetNormalizedNormal = true) {
	float4 GBufferA =
		Texture2DSampleLevel(SceneTexturesStruct_GBufferATexture,
			SceneTexturesStruct_GBufferATextureSampler, UV, 0);
	float4 GBufferB =
		Texture2DSampleLevel(SceneTexturesStruct_GBufferBTexture,
			SceneTexturesStruct_GBufferBTextureSampler, UV, 0);
	float4 GBufferC =
		Texture2DSampleLevel(SceneTexturesStruct_GBufferCTexture,
			SceneTexturesStruct_GBufferCTextureSampler, UV, 0);
	float4 GBufferD =
		Texture2DSampleLevel(SceneTexturesStruct_GBufferDTexture,
			SceneTexturesStruct_GBufferDTextureSampler, UV, 0);
	float CustomNativeDepth =
		Texture2DSampleLevel(SceneTexturesStruct_CustomDepthTexture,
			SceneTexturesStruct_CustomDepthTextureSampler, UV, 0)
		.r;
    uint CustomStencil;

	float4 GBufferE =
		Texture2DSampleLevel(SceneTexturesStruct_GBufferETexture,
			SceneTexturesStruct_GBufferETextureSampler, UV, 0);

	float4 GBufferVelocity = 0;

	float SceneDepth = CalcSceneDepth(UV);

	return DecodeGBufferData(GBufferA, GBufferB, GBufferC, GBufferD, GBufferE,
		GBufferVelocity, CustomNativeDepth, CustomStencil,
		SceneDepth, bGetNormalizedNormal,
		CheckerFromSceneColorUV(UV));
}


FScreenSpaceData GetScreenSpaceData(float2 UV,
	bool bGetNormalizedNormal = true) {
	FScreenSpaceData Out;

	Out.GBuffer = GetGBufferData(UV, bGetNormalizedNormal);
	float4 ScreenSpaceAO = Texture2DSampleLevel(
		SceneTexturesStruct_ScreenSpaceAOTexture,
		SceneTexturesStruct_ScreenSpaceAOTextureSampler, UV, 0);

	Out.AmbientOcclusion = ScreenSpaceAO.r;

	return Out;
}

FDeferredLightData SetupLightDataForStandardDeferred() {

	FDeferredLightData LightData;
	LightData.Position = DeferredLightUniforms_Position;
	LightData.InvRadius = DeferredLightUniforms_InvRadius;
	LightData.Color = DeferredLightUniforms_Color;
	LightData.FalloffExponent = DeferredLightUniforms_FalloffExponent;
	LightData.Direction = DeferredLightUniforms_Direction;
	LightData.Tangent = DeferredLightUniforms_Tangent;
	LightData.SpotAngles = DeferredLightUniforms_SpotAngles;
	LightData.SourceRadius = DeferredLightUniforms_SourceRadius,
		LightData.SourceLength = DeferredLightUniforms_SourceLength;
	LightData.SoftSourceRadius = DeferredLightUniforms_SoftSourceRadius;
	LightData.SpecularScale = DeferredLightUniforms_SpecularScale;
	LightData.ContactShadowLength =
		abs(DeferredLightUniforms_ContactShadowLength);
	LightData.ContactShadowLengthInWS =
		DeferredLightUniforms_ContactShadowLength < 0.0f;
	LightData.DistanceFadeMAD = DeferredLightUniforms_DistanceFadeMAD;
	LightData.ShadowMapChannelMask = DeferredLightUniforms_ShadowMapChannelMask;
	LightData.ShadowedBits = DeferredLightUniforms_ShadowedBits;

	LightData.bInverseSquared = 0;
	LightData.bRadialLight = 1 > 0;

	LightData.bSpotLight = 1 > 0;
	LightData.bRectLight = 1 == 2;

	LightData.RectLightBarnCosAngle = DeferredLightUniforms_RectLightBarnCosAngle;
	LightData.RectLightBarnLength = DeferredLightUniforms_RectLightBarnLength;

	return LightData;
}


float InterleavedGradientNoise(float2 uv, float FrameId) {

	uv += FrameId * (float2(47, 17) * 0.695f);

	const float3 magic = float3(0.06711056f, 0.00583715f, 52.9829189f);
	return frac(magic.z * frac(dot(uv, magic.xy)));
}

float4 Square(float4 x) {
	return x * x;
}

float Square(float x) {
	return x * x;
}

float2 Square(float2 x) {
	return x * x;
}

float3 Square(float3 x) {
	return x * x;
}

float4 GetPerPixelLightAttenuation(float2 UV) {
	return Square(Texture2DSampleLevel(LightAttenuationTexture,
		LightAttenuationTextureSampler, UV, 0));
}

float RadialAttenuation(float3 WorldLightVector, float FalloffExponent) {
	float NormalizeDistanceSquared = dot(WorldLightVector, WorldLightVector);

	return pow(1.0f - saturate(NormalizeDistanceSquared), FalloffExponent);
}

float SpotAttenuation(float3 L, float3 SpotDirection, float2 SpotAngles) {
	float ConeAngleFalloff =
		Square(saturate((dot(L, -SpotDirection) - SpotAngles.x) * SpotAngles.y));
	return ConeAngleFalloff;
}


float GetLocalLightAttenuation(float3 WorldPosition,
	FDeferredLightData LightData,
	inout float3 ToLight, inout float3 L) {
	ToLight = LightData.Position - WorldPosition;

	float DistanceSqr = dot(ToLight, ToLight);
	L = ToLight * rsqrt(DistanceSqr);

	float LightMask;
	if (LightData.bInverseSquared) {
		LightMask =
			Square(saturate(1 - Square(DistanceSqr * Square(LightData.InvRadius))));
	}
	else {
		LightMask = RadialAttenuation(ToLight * LightData.InvRadius,
			LightData.FalloffExponent);
	}

	if (LightData.bSpotLight) {
		LightMask *= SpotAttenuation(L, -LightData.Direction, LightData.SpotAngles);
	}

	if (LightData.bRectLight) {

		LightMask = dot(LightData.Direction, L) < 0 ? 0 : LightMask;
	}

	return LightMask;
}


// Example implementation of DistanceFromCameraFade() function  
float DistanceFromCameraFade(float depth, FDeferredLightData lightData, float3 worldPosition, float3 cameraOrigin)
{
	// Calculate the distance between the world position and the camera origin  
	float distance = length(worldPosition - cameraOrigin);

	// Apply a fade function based on the distance  
	float fadeFactor = saturate(1.0f - distance / depth);

	return fadeFactor;
}


// Main function  
void GetShadowTerms(FGBufferData GBuffer, FDeferredLightData LightData,
	float3 WorldPosition, float3 L, float4 LightAttenuation,
	float Dither, inout FShadowTerms Shadow) {
	float ContactShadowLength = 0.0f;
	const float ContactShadowLengthScreenScale =
		View_ClipToView[1][1] * GBuffer.Depth;

	if (LightData.ShadowedBits) {

		float UsesStaticShadowMap =
			dot(LightData.ShadowMapChannelMask, float4(1, 1, 1, 1));
		float StaticShadowing = lerp(
			1,
			dot(GBuffer.PrecomputedShadowFactors, LightData.ShadowMapChannelMask),
			UsesStaticShadowMap);

		if (LightData.bRadialLight) {

			Shadow.SurfaceShadow = LightAttenuation.z * StaticShadowing;

			Shadow.TransmissionShadow = LightAttenuation.w * StaticShadowing;

			Shadow.TransmissionThickness = LightAttenuation.w;
		}
		else {

			float DynamicShadowFraction = DistanceFromCameraFade(
				GBuffer.Depth, LightData, WorldPosition, View_WorldCameraOrigin);

			Shadow.SurfaceShadow =
				lerp(LightAttenuation.x, StaticShadowing, DynamicShadowFraction);

			Shadow.TransmissionShadow =
				min(lerp(LightAttenuation.y, StaticShadowing, DynamicShadowFraction),
					LightAttenuation.w);

			Shadow.SurfaceShadow *= LightAttenuation.z;
			Shadow.TransmissionShadow *= LightAttenuation.z;

			Shadow.TransmissionThickness =
				min(LightAttenuation.y, LightAttenuation.w);
		}

		if (LightData.ShadowedBits > 1 &&
			LightData.ContactShadowLength > 0) {
			ContactShadowLength =
				LightData.ContactShadowLength *
				(LightData.ContactShadowLengthInWS ? 1.0f
					: ContactShadowLengthScreenScale);
		}
	}
}


void Init(inout BxDFContext Context, float3 N, float3 V, float3 L) {
	Context.NoL = dot(N, L);
	Context.NoV = dot(N, V);
	Context.VoL = dot(V, L);
	float InvLenH = rsqrt(2 + 2 * Context.VoL);
	Context.NoH = saturate((Context.NoL + Context.NoV) * InvLenH);
	Context.VoH = saturate(InvLenH + InvLenH * Context.VoL);
}

float3 Diffuse_Lambert(float3 DiffuseColor) {
    return DiffuseColor * (1 / PI);
}

float3 SpecularGGX(float Roughness, float3 SpecularColor, BxDFContext Context, float NoL, FAreaLight AreaLight) {
	// Calculation of GGX Specular term is complex, and involves Fresnel, Geometric, and Distribution functions.
	// Here, we'll simplify it with a placeholder function. In actual practice, this function should compute the complete microfacet specular BRDF.
	float D = max(0.0, Context.NoH); // Placeholder distribution term (D)
	float G = min(1.0, Context.NoV * Context.NoL); // Placeholder geometric term (G)
	float3 F = SpecularColor; // Placeholder Fresnel term (F)

	// Combine all the terms
	return (D * G * F) / (4 * NoL * Context.NoV); // Microfacet specular BRDF
}

FDirectLighting DefaultLitBxDF(FGBufferData GBuffer, float3 N, float3 V,
	float3 L, float Falloff, float NoL,
	FAreaLight AreaLight, FShadowTerms Shadow) {
	BxDFContext Context;
	Init(Context, N, V, L);
	Context.NoV = saturate(abs(Context.NoV) + 1e-5);

	FDirectLighting Lighting;
	Lighting.Diffuse = AreaLight.FalloffColor * (Falloff * NoL) *
		Diffuse_Lambert(GBuffer.DiffuseColor);


	Lighting.Specular = AreaLight.FalloffColor * (Falloff * NoL) *
		SpecularGGX(GBuffer.Roughness, GBuffer.SpecularColor,
			Context, NoL, AreaLight);

	Lighting.Transmission = 0;
	return Lighting;
}

float Pow2(float x) {
	return x * x;
}

FDirectLighting IntegrateBxDF(FGBufferData GBuffer, float3 N, float3 V, FCapsuleLight Capsule, FShadowTerms Shadow, bool bInverseSquared) {
	float NoL;
	float Falloff;
	float LineCosSubtended = 1;

	float DistSqr = dot(Capsule.LightPos[0], Capsule.LightPos[0]);
	Falloff = rcp(DistSqr + Capsule.DistBiasSqr);

	float3 L = Capsule.LightPos[0] * rsqrt(DistSqr);
	NoL = dot(N, L);

	NoL = saturate(NoL);
	Falloff = bInverseSquared ? Falloff : 1;

	float3 ToLight = Capsule.LightPos[0];

	DistSqr = dot(ToLight, ToLight);
	float InvDist = rsqrt(DistSqr);
	L = ToLight * InvDist;

	GBuffer.Roughness = max(GBuffer.Roughness, 0.02);
	float a = Pow2(GBuffer.Roughness);

	FAreaLight AreaLight;
	AreaLight.SphereSinAlpha = saturate(Capsule.Radius * InvDist * (1 - a));
	AreaLight.SphereSinAlphaSoft = saturate(Capsule.SoftRadius * InvDist);
	AreaLight.LineCosSubtended = LineCosSubtended;
	AreaLight.FalloffColor = 1;
	AreaLight.Rect = (FRect)0;
	AreaLight.bIsRect = false;
	AreaLight.Texture = InitRectTexture(DummyRectLightTextureForCapsuleCompilerWarning);

	return DefaultLitBxDF(GBuffer, N, V, L, Falloff, NoL, AreaLight, Shadow);
}

FLightAccumulator LightAccumulator_Add(
	FLightAccumulator In, float3 TotalLight, float3 ScatterableLight,
	float3 CommonMultiplier,
	const bool bNeedsSeparateSubsurfaceLightAccumulation) {

	In.TotalLight += TotalLight * CommonMultiplier;
	return In;
}

float4 LightAccumulator_GetResult(FLightAccumulator In) {
	float4 Ret;

	Ret = float4(In.TotalLight, 0);
	return Ret;
}

FCapsuleLight GetCapsule(float3 ToLight, FDeferredLightData LightData) {
	FCapsuleLight Capsule;
	Capsule.Length = LightData.SourceLength;
	Capsule.Radius = LightData.SourceRadius;
	Capsule.SoftRadius = LightData.SoftSourceRadius;
	Capsule.DistBiasSqr = 1.0f;
	Capsule.LightPos[0] = ToLight - 0.5 * Capsule.Length * LightData.Tangent;
	Capsule.LightPos[1] = ToLight + 0.5 * Capsule.Length * LightData.Tangent;
	return Capsule;
}


float4 GetDynamicLighting(
	float3 WorldPosition,
	float3 CameraVector,
	FGBufferData GBuffer,
	float AmbientOcclusion,
	uint ShadingModelID,
	FDeferredLightData LightData,
	float4 LightAttenuation,
	float Dither,
	uint2 SVPos,
	FRectTexture SourceTexture
) {
	FLightAccumulator LightAccumulator = LightAccumulator_Init();
	LightAccumulator.EstimatedCost += 0.3f;

	float3 V = -CameraVector;
	float3 N = GBuffer.WorldNormal;

	float3 L = LightData.Direction;
	float3 ToLight = L;

	float LightMask = 1;
	if (LightData.bRadialLight) {
		LightMask = GetLocalLightAttenuation(WorldPosition, LightData, ToLight, L);
	}

	if (LightMask > 0) {
		FShadowTerms Shadow;
		Shadow.SurfaceShadow = AmbientOcclusion;
		Shadow.TransmissionShadow = 1;
		Shadow.TransmissionThickness = 1;
		GetShadowTerms(GBuffer, LightData, WorldPosition, L, LightAttenuation, Dither, Shadow);

		LightAccumulator.EstimatedCost += 0.3f;

		if (Shadow.SurfaceShadow + Shadow.TransmissionShadow > 0) {
			bool bNeedsSeparateSubsurfaceLightAccumulation = UseSubsurfaceProfile(GBuffer.ShadingModelID);
			float3 LightColor = LightData.Color;

			FDirectLighting Lighting;

			FCapsuleLight Capsule = GetCapsule(ToLight, LightData);
			Lighting = IntegrateBxDF(GBuffer, N, V, Capsule, Shadow, LightData.bInverseSquared);

			Lighting.Specular *= LightData.SpecularScale;

			LightAccumulator = LightAccumulator_Add(
				LightAccumulator,
				Lighting.Diffuse + Lighting.Specular,
				Lighting.Diffuse,
				LightColor * LightMask * Shadow.SurfaceShadow,
				bNeedsSeparateSubsurfaceLightAccumulation
			);
			LightAccumulator = LightAccumulator_Add(
				LightAccumulator,
				Lighting.Transmission,
				Lighting.Transmission,
				LightColor * LightMask * Shadow.TransmissionShadow,
				bNeedsSeparateSubsurfaceLightAccumulation
			);

			LightAccumulator.EstimatedCost += 0.4f;
		}
	}

	return LightAccumulator_GetResult(LightAccumulator);
}


float ComputeLightProfileMultiplier(
	float3 WorldPosition,
	float3 LightPosition,
	float3 LightDirection,
	float3 LightTangent
) {
	return 1.0f;
}


struct VertexOutput
{
    float4 OutScreenPosition : TEXCOORD0;
    float4 OutPosition : SV_POSITION;
};


float4 DeferredLightPixelMain(VertexOutput vout) : SV_TARGET0
{

	//printf("DeferredLightPixelMain\n");

	float4 InScreenPosition = vout.OutScreenPosition;
	float4 SVPos = vout.OutPosition;
	float4 OutColor = 0;

	float2 ScreenUV = InScreenPosition.xy / InScreenPosition.w * View_ScreenPositionScaleBias.xy + View_ScreenPositionScaleBias.wz;

	FScreenSpaceData ScreenSpaceData = GetScreenSpaceData(ScreenUV);

	if (ScreenSpaceData.GBuffer.ShadingModelID > 0)
	{
		float SceneDepth = CalcSceneDepth(ScreenUV);

		float2 ClipPosition = InScreenPosition.xy / InScreenPosition.w * (View_ViewToClip[3][3] < 1.0f ? SceneDepth : 1.0f);
		float4 position = mul(float4(ClipPosition, SceneDepth, 1), View_ScreenToWorld);
		float3 WorldPosition = position.xyz;
		float3 CameraVector = normalize(WorldPosition - View_WorldCameraOrigin);

		FDeferredLightData LightData = SetupLightDataForStandardDeferred();

		float Dither = InterleavedGradientNoise(SVPos.xy, View_StateFrameIndexMod8);

		FRectTexture RectTexture = InitRectTexture(DeferredLightUniforms_SourceTexture);
		OutColor = GetDynamicLighting(WorldPosition, CameraVector, ScreenSpaceData.GBuffer, ScreenSpaceData.AmbientOcclusion, ScreenSpaceData.GBuffer.ShadingModelID, LightData, GetPerPixelLightAttenuation(ScreenUV), Dither, uint2(SVPos.xy), RectTexture);
		OutColor *= ComputeLightProfileMultiplier(WorldPosition, DeferredLightUniforms_Position, -DeferredLightUniforms_Direction, DeferredLightUniforms_Tangent);
	}

	return OutColor;
}



static const float2 DiscSamples29[]=
{
	float2(0.000000, 2.500000),
	float2(1.016842, 2.283864),
	float2(1.857862, 1.672826),
	float2(2.377641, 0.772542),
	float2(2.486305, -0.261321),
	float2(2.165063, -1.250000),
	float2(1.469463, -2.022543),
	float2(0.519779, -2.445369),
	float2(-0.519779, -2.445369),
	float2(-1.469463, -2.022542),
	float2(-2.165064, -1.250000),
	float2(-2.486305, -0.261321),
	float2(-2.377641, 0.772543),
	float2(-1.857862, 1.672827),
	float2(-1.016841, 2.283864),
	float2(0.091021, -0.642186),
	float2(0.698035, 0.100940),
	float2(0.959731, -1.169393),
	float2(-1.053880, 1.180380),
	float2(-1.479156, -0.606937),
	float2(-0.839488, -1.320002),
	float2(1.438566, 0.705359),
	float2(0.067064, -1.605197),
	float2(0.728706, 1.344722),
	float2(1.521424, -0.380184),
	float2(-0.199515, 1.590091),
	float2(-1.524323, 0.364010),
	float2(-0.692694, -0.086749),
	float2(-0.082476, 0.654088),
};


float CubemapHardwarePCF(float3 WorldPosition, float3 LightPosition, float LightInvRadius, float DepthBias)
{
	float Shadow = 1;
	float3 LightVector = LightPosition - WorldPosition.xyz;
	float Distance = length(LightVector);
	[branch]
	if (Distance * LightInvRadius < 1.0f)
	{
		float3 NormalizedLightVector = LightVector / Distance;
		float3 SideVector = normalize(cross(NormalizedLightVector, float3(0, 0, 1)));
		float3 UpVector = cross(SideVector, NormalizedLightVector);
		SideVector *= InvShadowmapResolution;
		UpVector *= InvShadowmapResolution;
		float3 AbsLightVector = abs(LightVector);
		float MaxCoordinate = max(AbsLightVector.x, max(AbsLightVector.y, AbsLightVector.z));
		int CubeFaceIndex = 0;
		if (MaxCoordinate == AbsLightVector.x)
		{
			CubeFaceIndex = AbsLightVector.x == LightVector.x ? 0 : 1;
		}
		else if (MaxCoordinate == AbsLightVector.y)
		{
			CubeFaceIndex = AbsLightVector.y == LightVector.y ? 2 : 3;
		}
		else
		{
			CubeFaceIndex = AbsLightVector.z == LightVector.z ? 4 : 5;
		}
		float4 ShadowPosition = mul(float4(WorldPosition.xyz, 1), ShadowViewProjectionMatrices[CubeFaceIndex]);
		float CompareDistance = ShadowPosition.z / ShadowPosition.w;
		float ShadowDepthBias = - DepthBias / ShadowPosition.w;
		Shadow = 0;
		[unroll]  for(int i = 0; i < 29; ++i)
		{
			float3 SamplePos = NormalizedLightVector + SideVector * DiscSamples29[i].x + UpVector * DiscSamples29[i].y;
			Shadow += ShadowDepthCubeTexture.SampleCmpLevelZero(
				ShadowDepthCubeTextureSampler,
				SamplePos.xy,
				CompareDistance + ShadowDepthBias * length(DiscSamples29[i])).r;
		}
		Shadow /= 29;
	}
	return Shadow;
}


float  EncodeLightAttenuation( float  InColor)
{
	return sqrt(InColor);
}


float4 MainOnePassPointLightShadowPS(
	VertexOutput vout
	): SV_TARGET0
{
	float4 OutColor;
	float2 ScreenUV = float2( vout.OutPosition.xy * View_BufferSizeAndInvSize.zw );
	float SceneW = CalcSceneDepth( ScreenUV );
	float2 ScreenPosition = ( ScreenUV.xy - View_ScreenPositionScaleBias.wz ) / View_ScreenPositionScaleBias.xy;
	float4 position = mul(float4(ScreenPosition.xy * SceneW, SceneW, 1), View_ScreenToWorld);
	float3 WorldPosition = position.xyz;
	float3 LightVector = LightPositionAndInvRadius.xyz - WorldPosition.xyz;
 	float Shadow = CubemapHardwarePCF(WorldPosition, LightPositionAndInvRadius.xyz, LightPositionAndInvRadius.w, PointLightDepthBiasAndProjParameters.x);
	Shadow = saturate( (Shadow - 0.5) * ShadowSharpen + 0.5 );
	float FadedShadow = lerp(1.0f, Square(Shadow), ShadowFadeFraction);
	OutColor.b = EncodeLightAttenuation(FadedShadow);
	OutColor.rga = 1;
	OutColor.a = OutColor.b;
	return OutColor;
}