// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#include "NvDecoder/NvDecoder.h"

// This class is derived from NvDecoder class and is used to optimize the cuvidGetDecoderCaps overhead
class OptimizedNvDecoder : public NvDecoder {

  public:
    OptimizedNvDecoder() {}
    /**
     *  @brief This function is used to initialize the decoder session.
     *  Application must call this function to initialize the decoder, before
     *  starting to decode any frames.
     *  The only difference from the original function is to add a new member m_decodecaps.
     *  Other part is the same as the original function, refer to NvDecoder.cpp in NVIDIA Video Codec SDK.
     */
    OptimizedNvDecoder(CUcontext &cuContext, bool bUseDeviceFrame, cudaVideoCodec eCodec, CUVIDDECODECAPS decodecaps,
                       bool bLowLatency = false, bool bDeviceFramePitched = false, const Rect *pCropRect = NULL,
                       const Dim *pResizeDim = NULL, bool extract_user_SEI_Message = false, int maxWidth = 0,
                       int maxHeight = 0, unsigned int clkRate = 1000, bool force_zero_latency = false);

    /**
     * @brief This function is to overwrite the origin Decode function to record the latency on frame level.
     */
    int Decode(const uint8_t *pData, int nSize, int nFlags = 0, int64_t nTimestamp = 0);
    /**
     * @brief This function is used to Get the frameLatency vector
     */
    std::vector<std::tuple<int, double>> &GetFrameLatency() { return frameLatency; }

  protected:
    /**
     *   @brief  Callback function to be registered for getting a callback when decoding of sequence starts
     */
    static int CUDAAPI HandleVideoSequenceProc(void *pUserData, CUVIDEOFORMAT *pVideoFormat) {
        if (pUserData == nullptr) {
            throw std::runtime_error("pUserData is nullptr");
        }
        return ((OptimizedNvDecoder *)pUserData)->HandleVideoSequence(pVideoFormat);
    }
    /**
     *   @brief  Define the new handler when decoding of sequence starts.
     *           The only change is to re-query decoder caps when the video codec or format change
     *           Other part is the same as the original function, refer to NvDecoder.cpp in NVIDIA Video Codec SDK.
     */
    int HandleVideoSequence(CUVIDEOFORMAT *pVideoFormat);

    CUVIDDECODECAPS m_decodecaps;

    std::vector<std::tuple<int, double>> frameLatency;
};
