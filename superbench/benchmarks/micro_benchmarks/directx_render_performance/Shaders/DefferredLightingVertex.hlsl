

cbuffer StencilingParametersConstantBuffer : register(b0)
{
    float4 StencilingGeometryPosAndScale;
    float4 StencilingConeParameters;
    float4x4 StencilingConeTransform;
    float3 StencilingPreViewTranslation;
    float padding;
};


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


struct VertexInput  
{  
    float3 InPosition : POSITION;  
};  


struct VertexOutput
{
    float4 OutScreenPosition : TEXCOORD0;
    float4 OutPosition : SV_POSITION;
};

const static float PI = 3.1415926535897932f;
const static float MaxHalfFloat = 65504.0f;

VertexOutput RadialVertexMain1(VertexInput input, uint InVertexId : SV_VertexID)
{
    //printf ("RadialVertexMain\n");
    VertexOutput output;

    float3 WorldPosition;
    uint NumSides = StencilingConeParameters.x;

    if (NumSides != 0)
    {
        float SphereRadius = StencilingConeParameters.w;
        float ConeAngle = StencilingConeParameters.z;

        const float InvCosRadiansPerSide = 1.0f / cos(PI / (float)NumSides);

        const float ZRadius = SphereRadius * cos(ConeAngle);
        const float TanConeAngle = tan(ConeAngle);

        uint NumSlices = StencilingConeParameters.y;
        uint CapIndexStart = NumSides * NumSlices;

        if (InVertexId < CapIndexStart)
        {
            uint SliceIndex = InVertexId / NumSides;
            uint SideIndex = InVertexId % NumSides;

            const float CurrentAngle = SideIndex * 2 * PI / (float)NumSides;
            const float DistanceDownConeDirection =
                ZRadius * SliceIndex / (float)(NumSlices - 1);

            const float SliceRadius =
                DistanceDownConeDirection * TanConeAngle * InvCosRadiansPerSide;

            const float3 LocalPosition = float3(
                ZRadius * SliceIndex / (float)(NumSlices - 1),
                SliceRadius * sin(CurrentAngle), SliceRadius * cos(CurrentAngle));

            WorldPosition =
                mul(float4(LocalPosition, 1), StencilingConeTransform).xyz +
                StencilingPreViewTranslation;
        }
        else
        {
            const float CapRadius = ZRadius * tan(ConeAngle);

            uint VertexId = InVertexId - CapIndexStart;
            uint SliceIndex = VertexId / NumSides;
            uint SideIndex = VertexId % NumSides;

            const float UnadjustedSliceRadius =
                CapRadius * SliceIndex / (float)(NumSlices - 1);

            const float SliceRadius = UnadjustedSliceRadius * InvCosRadiansPerSide;

            const float ZDistance =
                sqrt(SphereRadius * SphereRadius -
                    UnadjustedSliceRadius * UnadjustedSliceRadius);

            const float CurrentAngle = SideIndex * 2 * PI / (float)NumSides;
            const float3 LocalPosition =
                float3(ZDistance, SliceRadius * sin(CurrentAngle),
                    SliceRadius * cos(CurrentAngle));
            WorldPosition =
                mul(float4(LocalPosition, 1), StencilingConeTransform).xyz +
                StencilingPreViewTranslation;
        }
    }
    else
    {
        WorldPosition = input.InPosition * StencilingGeometryPosAndScale.w +
            StencilingGeometryPosAndScale.xyz;
    }

    output.OutScreenPosition = output.OutPosition =
        mul(float4(WorldPosition, 1), View_TranslatedWorldToClip);

    return output;
}

VertexOutput RadialVertexMain(VertexInput input, uint InVertexId : SV_VertexID)
{
    VertexOutput   output;
    float3 WorldPosition = {0,0, 0};
    uint NumSides = StencilingConeParameters.x;
  
    if (NumSides != 0)
    {
        float SphereRadius = StencilingConeParameters.w;
        float ConeAngle = StencilingConeParameters.z;
        const float InvCosRadiansPerSide = 1.0f / cos(PI / (float)NumSides);

        const float ZRadius = SphereRadius * cos(ConeAngle);
        const float TanConeAngle = tan(ConeAngle);

        uint NumSlices = StencilingConeParameters.y;
        uint CapIndexStart = NumSides * NumSlices;
        if (InVertexId < CapIndexStart)
        {
            uint SliceIndex = InVertexId / NumSides;
            uint SideIndex = InVertexId % NumSides;

            const float CurrentAngle = SideIndex * 2 * PI / (float)NumSides;
            const float DistanceDownConeDirection =
                ZRadius * SliceIndex / (float)(NumSlices - 1);

            const float SliceRadius =
                DistanceDownConeDirection * TanConeAngle * InvCosRadiansPerSide;

            const float3 LocalPosition = float3(
                ZRadius * SliceIndex / (float)(NumSlices - 1),
                SliceRadius * sin(CurrentAngle), SliceRadius * cos(CurrentAngle));
            float4 position = mul(float4(LocalPosition, 1), StencilingConeTransform);
            WorldPosition = position.xyz + StencilingPreViewTranslation;
        }
        else
        {
            const float CapRadius = ZRadius * tan(ConeAngle);

            uint VertexId = InVertexId - CapIndexStart;
            uint SliceIndex = VertexId / NumSides;
            uint SideIndex = VertexId % NumSides;

            const float UnadjustedSliceRadius =
                CapRadius * SliceIndex / (float)(NumSlices - 1);

            const float SliceRadius = UnadjustedSliceRadius * InvCosRadiansPerSide;

            const float ZDistance =
                sqrt(SphereRadius * SphereRadius -
                    UnadjustedSliceRadius * UnadjustedSliceRadius);

            const float CurrentAngle = SideIndex * 2 * PI / (float)NumSides;
            const float3 LocalPosition =
                float3(ZDistance, SliceRadius * sin(CurrentAngle),
                    SliceRadius * cos(CurrentAngle));
            float4 position = mul(float4(LocalPosition, 1), StencilingConeTransform);
            WorldPosition = position.xyz + StencilingPreViewTranslation;
        }
    }
    else
    {
        WorldPosition = input.InPosition * StencilingGeometryPosAndScale.w +
            StencilingGeometryPosAndScale.xyz;
    }


    output.OutScreenPosition = output.OutPosition =
        mul(float4(WorldPosition, 1), View_TranslatedWorldToClip);
    
    return output;
}
