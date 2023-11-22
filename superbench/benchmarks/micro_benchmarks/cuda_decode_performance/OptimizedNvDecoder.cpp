// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cmath>

#include "OptimizedNvDecoder.h"

int OptimizedNvDecoder::Decode(const uint8_t *pData, int nSize, int nFlags, int64_t nTimestamp) {
    m_nDecodedFrame = 0;
    m_nDecodedFrameReturned = 0;
    CUVIDSOURCEDATAPACKET packet = {0};
    packet.payload = pData;
    packet.payload_size = nSize;
    packet.flags = nFlags | CUVID_PKT_TIMESTAMP;
    packet.timestamp = nTimestamp;
    if (!pData || nSize == 0) {
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
    }
    auto start = std::chrono::high_resolution_clock::now();
    NVDEC_API_CALL(cuvidParseVideoData(m_hParser, &packet));
    int64_t elapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count();
    frameLatency.push_back(std::make_tuple(m_nDecodedFrame, elapsedTime / 1000.0f / 1000.0f));
    return m_nDecodedFrame;
}

OptimizedNvDecoder::OptimizedNvDecoder(CUcontext &cuContext, bool bUseDeviceFrame, cudaVideoCodec eCodec,
                                       CUVIDDECODECAPS decodecaps, bool bLowLatency, bool bDeviceFramePitched,
                                       const Rect *pCropRect, const Dim *pResizeDim, bool extract_user_SEI_Message,
                                       int maxWidth, int maxHeight, unsigned int clkRate, bool force_zero_latency) {
    m_cuContext = cuContext;
    m_bUseDeviceFrame = bUseDeviceFrame;
    m_eCodec = eCodec;
    m_bDeviceFramePitched = bDeviceFramePitched;
    m_bExtractSEIMessage = extract_user_SEI_Message;
    m_nMaxWidth = maxWidth;
    m_nMaxHeight = maxHeight;
    m_bForce_zero_latency = force_zero_latency;
    if (pCropRect)
        m_cropRect = *pCropRect;
    if (pResizeDim)
        m_resizeDim = *pResizeDim;

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    NVDEC_API_CALL(cuvidCtxLockCreate(&m_ctxLock, cuContext));

    ck(cuStreamCreate(&m_cuvidStream, CU_STREAM_DEFAULT));

    decoderSessionID = 0;

    if (m_bExtractSEIMessage) {
        m_fpSEI = fopen("sei_message.txt", "wb");
        m_pCurrSEIMessage = new CUVIDSEIMESSAGEINFO;
        memset(&m_SEIMessagesDisplayOrder, 0, sizeof(m_SEIMessagesDisplayOrder));
    }
    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = eCodec;
    videoParserParameters.ulMaxNumDecodeSurfaces = 1;
    videoParserParameters.ulClockRate = clkRate;
    videoParserParameters.ulMaxDisplayDelay = bLowLatency ? 0 : 1;
    videoParserParameters.pUserData = this;
    videoParserParameters.pfnSequenceCallback = HandleVideoSequenceProc;
    videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc;
    videoParserParameters.pfnDisplayPicture = m_bForce_zero_latency ? NULL : HandlePictureDisplayProc;
    videoParserParameters.pfnGetOperatingPoint = HandleOperatingPointProc;
    videoParserParameters.pfnGetSEIMsg = m_bExtractSEIMessage ? HandleSEIMessagesProc : NULL;
    NVDEC_API_CALL(cuvidCreateVideoParser(&m_hParser, &videoParserParameters));
    // reuse the decodecaps queried before
    m_decodecaps = decodecaps;
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
}

int OptimizedNvDecoder::HandleVideoSequence(CUVIDEOFORMAT *pVideoFormat) {
    START_TIMER
    m_videoInfo.str("");
    m_videoInfo.clear();
    m_videoInfo << "Video Input Information" << std::endl
                << "\tCodec        : " << GetVideoCodecString(pVideoFormat->codec) << std::endl
                << "\tFrame rate   : " << pVideoFormat->frame_rate.numerator << "/"
                << pVideoFormat->frame_rate.denominator << " = "
                << 1.0 * pVideoFormat->frame_rate.numerator / pVideoFormat->frame_rate.denominator << " fps"
                << std::endl
                << "\tSequence     : " << (pVideoFormat->progressive_sequence ? "Progressive" : "Interlaced")
                << std::endl
                << "\tCoded size   : [" << pVideoFormat->coded_width << ", " << pVideoFormat->coded_height << "]"
                << std::endl
                << "\tDisplay area : [" << pVideoFormat->display_area.left << ", " << pVideoFormat->display_area.top
                << ", " << pVideoFormat->display_area.right << ", " << pVideoFormat->display_area.bottom << "]"
                << std::endl
                << "\tChroma       : " << GetVideoChromaFormatString(pVideoFormat->chroma_format) << std::endl
                << "\tBit depth    : " << pVideoFormat->bit_depth_luma_minus8 + 8;
    m_videoInfo << std::endl;

    int nDecodeSurface = pVideoFormat->min_num_decode_surfaces;

    // re-call the cuvidGetDecoderCaps when the video codeoc and format change
    if (m_decodecaps.eCodecType != pVideoFormat->codec || m_decodecaps.eChromaFormat != pVideoFormat->chroma_format ||
        m_decodecaps.nBitDepthMinus8 != pVideoFormat->bit_depth_luma_minus8) {
        m_decodecaps.eCodecType = pVideoFormat->codec;
        m_decodecaps.eChromaFormat = pVideoFormat->chroma_format;
        m_decodecaps.nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;

        CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
        NVDEC_API_CALL(cuvidGetDecoderCaps(&m_decodecaps));
        CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    }

    if (!m_decodecaps.bIsSupported) {
        NVDEC_THROW_ERROR("Codec not supported on this GPU", CUDA_ERROR_NOT_SUPPORTED);
        return nDecodeSurface;
    }

    if ((pVideoFormat->coded_width > m_decodecaps.nMaxWidth) ||
        (pVideoFormat->coded_height > m_decodecaps.nMaxHeight)) {

        std::ostringstream errorString;
        errorString << std::endl
                    << "Resolution          : " << pVideoFormat->coded_width << "x" << pVideoFormat->coded_height
                    << std::endl
                    << "Max Supported (wxh) : " << m_decodecaps.nMaxWidth << "x" << m_decodecaps.nMaxHeight << std::endl
                    << "Resolution not supported on this GPU";

        const std::string cErr = errorString.str();
        NVDEC_THROW_ERROR(cErr, CUDA_ERROR_NOT_SUPPORTED);
        return nDecodeSurface;
    }
    if ((pVideoFormat->coded_width >> 4) * (pVideoFormat->coded_height >> 4) > m_decodecaps.nMaxMBCount) {

        std::ostringstream errorString;
        errorString << std::endl
                    << "MBCount             : " << (pVideoFormat->coded_width >> 4) * (pVideoFormat->coded_height >> 4)
                    << std::endl
                    << "Max Supported mbcnt : " << m_decodecaps.nMaxMBCount << std::endl
                    << "MBCount not supported on this GPU";
        NVDEC_THROW_ERROR(errorString.str(), CUDA_ERROR_NOT_SUPPORTED);
        return nDecodeSurface;
    }

    if (m_nWidth && m_nLumaHeight && m_nChromaHeight) {

        // cuvidCreateDecoder() has been called before, and now there's possible config change
        return ReconfigureDecoder(pVideoFormat);
    }

    // eCodec has been set in the constructor (for parser). Here it's set again for potential correction
    m_eCodec = pVideoFormat->codec;
    m_eChromaFormat = pVideoFormat->chroma_format;
    m_nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    m_nBPP = m_nBitDepthMinus8 > 0 ? 2 : 1;

    // Set the output surface format same as chroma format
    if (m_eChromaFormat == cudaVideoChromaFormat_420 || cudaVideoChromaFormat_Monochrome)
        m_eOutputFormat =
            pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    else if (m_eChromaFormat == cudaVideoChromaFormat_444)
        m_eOutputFormat =
            pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
    else if (m_eChromaFormat == cudaVideoChromaFormat_422)
        m_eOutputFormat = cudaVideoSurfaceFormat_NV12; // no 4:2:2 output format supported yet so make 420 default

    // Check if output format supported. If not, check falback options
    if (!(m_decodecaps.nOutputFormatMask & (1 << m_eOutputFormat))) {
        if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))
            m_eOutputFormat = cudaVideoSurfaceFormat_NV12;
        else if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016))
            m_eOutputFormat = cudaVideoSurfaceFormat_P016;
        else if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444))
            m_eOutputFormat = cudaVideoSurfaceFormat_YUV444;
        else if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444_16Bit))
            m_eOutputFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
        else
            NVDEC_THROW_ERROR("No supported output format found", CUDA_ERROR_NOT_SUPPORTED);
    }
    m_videoFormat = *pVideoFormat;

    CUVIDDECODECREATEINFO videoDecodeCreateInfo = {0};
    videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
    videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
    videoDecodeCreateInfo.OutputFormat = m_eOutputFormat;
    videoDecodeCreateInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    if (pVideoFormat->progressive_sequence)
        videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    else
        videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
    videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
    // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware
    videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    videoDecodeCreateInfo.ulNumDecodeSurfaces = nDecodeSurface;
    videoDecodeCreateInfo.vidLock = m_ctxLock;
    videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
    videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;
    // AV1 has max width/height of sequence in sequence header
    if (pVideoFormat->codec == cudaVideoCodec_AV1 && pVideoFormat->seqhdr_data_length > 0) {
        CUVIDEOFORMATEX *vidFormatEx = (CUVIDEOFORMATEX *)pVideoFormat;
        if (m_nMaxWidth < pVideoFormat->coded_width) {
            m_nMaxWidth = vidFormatEx->av1.max_width;
        }
        if (m_nMaxHeight < pVideoFormat->coded_height) {
            m_nMaxHeight = vidFormatEx->av1.max_height;
        }
    }
    if (m_nMaxWidth < (int)pVideoFormat->coded_width)
        m_nMaxWidth = pVideoFormat->coded_width;
    if (m_nMaxHeight < (int)pVideoFormat->coded_height)
        m_nMaxHeight = pVideoFormat->coded_height;
    videoDecodeCreateInfo.ulMaxWidth = m_nMaxWidth;
    videoDecodeCreateInfo.ulMaxHeight = m_nMaxHeight;

    if (!(m_cropRect.r && m_cropRect.b) && !(m_resizeDim.w && m_resizeDim.h)) {
        m_nWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
        m_nLumaHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
        videoDecodeCreateInfo.ulTargetWidth = pVideoFormat->coded_width;
        videoDecodeCreateInfo.ulTargetHeight = pVideoFormat->coded_height;
    } else {
        if (m_resizeDim.w && m_resizeDim.h) {
            videoDecodeCreateInfo.display_area.left = pVideoFormat->display_area.left;
            videoDecodeCreateInfo.display_area.top = pVideoFormat->display_area.top;
            videoDecodeCreateInfo.display_area.right = pVideoFormat->display_area.right;
            videoDecodeCreateInfo.display_area.bottom = pVideoFormat->display_area.bottom;
            m_nWidth = m_resizeDim.w;
            m_nLumaHeight = m_resizeDim.h;
        }

        if (m_cropRect.r && m_cropRect.b) {
            videoDecodeCreateInfo.display_area.left = m_cropRect.l;
            videoDecodeCreateInfo.display_area.top = m_cropRect.t;
            videoDecodeCreateInfo.display_area.right = m_cropRect.r;
            videoDecodeCreateInfo.display_area.bottom = m_cropRect.b;
            m_nWidth = m_cropRect.r - m_cropRect.l;
            m_nLumaHeight = m_cropRect.b - m_cropRect.t;
        }
        videoDecodeCreateInfo.ulTargetWidth = m_nWidth;
        videoDecodeCreateInfo.ulTargetHeight = m_nLumaHeight;
    }

    m_nChromaHeight = (int)(ceil(m_nLumaHeight * GetChromaHeightFactor(m_eOutputFormat)));
    m_nNumChromaPlanes = GetChromaPlaneCount(m_eOutputFormat);
    m_nSurfaceHeight = videoDecodeCreateInfo.ulTargetHeight;
    m_nSurfaceWidth = videoDecodeCreateInfo.ulTargetWidth;
    m_displayRect.b = videoDecodeCreateInfo.display_area.bottom;
    m_displayRect.t = videoDecodeCreateInfo.display_area.top;
    m_displayRect.l = videoDecodeCreateInfo.display_area.left;
    m_displayRect.r = videoDecodeCreateInfo.display_area.right;

    m_videoInfo << "Video Decoding Params:" << std::endl
                << "\tNum Surfaces : " << videoDecodeCreateInfo.ulNumDecodeSurfaces << std::endl
                << "\tCrop         : [" << videoDecodeCreateInfo.display_area.left << ", "
                << videoDecodeCreateInfo.display_area.top << ", " << videoDecodeCreateInfo.display_area.right << ", "
                << videoDecodeCreateInfo.display_area.bottom << "]" << std::endl
                << "\tResize       : " << videoDecodeCreateInfo.ulTargetWidth << "x"
                << videoDecodeCreateInfo.ulTargetHeight << std::endl
                << "\tDeinterlace  : "
                << std::vector<const char *>{"Weave", "Bob", "Adaptive"}[videoDecodeCreateInfo.DeinterlaceMode];
    m_videoInfo << std::endl;

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    NVDEC_API_CALL(cuvidCreateDecoder(&m_hDecoder, &videoDecodeCreateInfo));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    STOP_TIMER("Session Initialization Time: ");
    NvDecoder::addDecoderSessionOverHead(getDecoderSessionID(), elapsedTime);
    return nDecodeSurface;
}
