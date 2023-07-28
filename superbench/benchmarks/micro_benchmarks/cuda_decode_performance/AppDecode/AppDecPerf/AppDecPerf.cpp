/*
 * Copyright 2017-2023 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//---------------------------------------------------------------------------
//! \file AppDecPerf.cpp
//! \brief Source file for AppDecPerf sample
//!
//!  This sample application measures decoding performance in FPS.
//!  The application creates multiple host threads and runs a different decoding session on each thread.
//!  The number of threads can be controlled by the CLI option "-thread".
//!  The application creates 2 host threads, each with a separate decode session, by default.
//!  The application supports measuring the decode performance only (keeping decoded
//!  frames in device memory as well as measuring the decode performance including transfer
//!  of frames to the host memory.
//---------------------------------------------------------------------------

#include <cuda.h>
#include <cudaProfiler.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <string.h>
#include <memory>
#include <fstream>
#include <string>
#include <numeric>
#include <chrono>
#include <algorithm>

#include "NvDecoder/NvDecoder.h"
#include "../Utils/NvCodecUtils.h"
#include "../Utils/FFmpegDemuxer.h"
#include "../Utils/ThreadPoolUtils.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

/**
 *   @brief  Function to decode media file using NvDecoder interface
 *   @param  pDec    - Handle to NvDecoder
 *   @param  demuxer - Pointer to an FFmpegDemuxer instance
 *   @param  pnFrame - Variable to record the number of frames decoded
 *   @param  ex      - Stores current exception in case of failure
 */
// void DecProc(NvDecoder *pDec, FFmpegDemuxer *demuxer, int *pnFrame, std::exception_ptr &ex)
void DecProc(NvDecoder *pDec, const char *szInFilePath, int *pnFrame, std::exception_ptr &ex)
{
    try
    {
        std::unique_ptr<FFmpegDemuxer> demuxer(new FFmpegDemuxer(szInFilePath));
        int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
        uint8_t *pVideo = NULL, *pFrame = NULL;

        do
        {
            demuxer->Demux(&pVideo, &nVideoBytes);
            nFrameReturned = pDec->Decode(pVideo, nVideoBytes);
            if (!nFrame && nFrameReturned)
                LOG(INFO) << pDec->GetVideoInfo();

            nFrame += nFrameReturned;
        } while (nVideoBytes);
        *pnFrame = nFrame;
    }
    catch (std::exception &)
    {
        ex = std::current_exception();
    }
}

void ShowHelpAndExit(const char *szBadOption = NULL)
{
    std::ostringstream oss;
    bool bThrowError = false;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "-i           Input file path" << std::endl
        << "-gpu         Ordinal of GPU to use" << std::endl
        << "-thread      Number of decoding thread" << std::endl
        << "-total       Number of total videos to test" << std::endl
        << "-single      (No value) Use single context (this may result in suboptimal performance; default is multiple contexts)" << std::endl
        << "-host        (No value) Copy frame to host memory (this may result in suboptimal performance; default is device memory)" << std::endl
        << "-multi_input  Multiple Input file list path" << std::endl;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
    else
    {
        std::cout << oss.str();
        exit(0);
    }
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, int &iGpu, int &nThread, int &nTotalVideo, bool &bSingle, bool &bHost, std::string &inputFilesListPath)
{
    for (int i = 1; i < argc; i++)
    {
        if (!_stricmp(argv[i], "-h"))
        {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-gpu"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-thread"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-thread");
            }
            nThread = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-total"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-total");
            }
            nTotalVideo = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-multi_input"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-multi_input");
            }
            inputFilesListPath = std::string(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-single"))
        {
            bSingle = true;
            continue;
        }
        if (!_stricmp(argv[i], "-host"))
        {
            bHost = true;
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

struct NvDecPerfData
{
    uint8_t *pBuf;
    std::vector<uint8_t *> *pvpPacketData;
    std::vector<int> *pvpPacketDataSize;
};

int CUDAAPI HandleVideoData(void *pUserData, CUVIDSOURCEDATAPACKET *pPacket)
{
    NvDecPerfData *p = (NvDecPerfData *)pUserData;
    memcpy(p->pBuf, pPacket->payload, pPacket->payload_size);
    p->pvpPacketData->push_back(p->pBuf);
    p->pvpPacketDataSize->push_back(pPacket->payload_size);
    p->pBuf += pPacket->payload_size;
    return 1;
}

NvDecoder *InitNvDecoder(int i, const CUdevice &cuDevice, CUcontext &cuContext, bool bSingle, bool bHost, cudaVideoCodec codec, CUVIDDECODECAPS decodecaps)
{
    if (!bSingle)
    {
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
    }
    NvDecoder *sessionObject = new NvDecoder(cuContext, !bHost, codec, decodecaps);
    sessionObject->setDecoderSessionID(i);
    return sessionObject;
}

float DecodeVideo(size_t i, const std::vector<NvDecoder *> &vDec, const char *szInFilePath, int *pnFrame, std::exception_ptr &ex)
{
    NvDecoder *pDec = vDec[i];
    auto start = std::chrono::high_resolution_clock::now();
    DecProc(pDec, szInFilePath, pnFrame, ex);
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    float decodeDuration = (elapsedTime - NvDecoder::getDecoderSessionOverHead(i)) / 1000.0f;
    NvDecoder::resetDecoderSessionOverHead(i);
    return decodeDuration;
}

std::vector<std::string> ReadMultipleVideoFiles(std::string filepath)
{
    std::ifstream file(filepath);

    if (!file)
    {
        std::cerr << "Error opening the file." << std::endl;
        exit(1);
    }

    std::string line;
    std::vector<std::string> tokens;
    while (std::getline(file, line))
    {
        tokens.push_back(line);
    }

    file.close();
    return tokens;
}

void InitializeContext(std::vector<NvDecoder *> &vDec, int iGpu, int nThread, bool bSingle, bool bHost)
{
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu)
    {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        exit(1);
    }
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;

    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    auto codec = cudaVideoCodec_H264;
    CUVIDDECODECAPS decodecaps;
    memset(&decodecaps, 0, sizeof(decodecaps));
    decodecaps.eCodecType = codec;
    decodecaps.eChromaFormat = cudaVideoChromaFormat_420;
    decodecaps.nBitDepthMinus8 = 0;
    NVDEC_API_CALL(cuvidGetDecoderCaps(&decodecaps));

    ThreadPool threadPool(nThread);
    std::vector<std::future<NvDecoder *>> futures;
    for (int i = 0; i < nThread; i++)
    {
        futures.push_back(threadPool.enqueue(InitNvDecoder, cuDevice, cuContext, bSingle, bHost, codec, decodecaps));
    }

    for (auto &future : futures)
    {
        vDec.push_back(future.get()); // Retrieve the results from each task
    }
}

std::tuple<double, double, double, double, double, double, double, double> CalLatencyMetrics(std::vector<double> data){
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();
    double min = *std::min_element(data.begin(), data.end());
    double max = *std::max_element(data.begin(), data.end());
    std::sort(data.begin(), data.end());
    double p50 = data[data.size() / 2];                         
    double p90 = data[static_cast<size_t>(data.size() * 0.9)];  
    double p95 = data[static_cast<size_t>(data.size() * 0.95)]; 
    double p99 = data[static_cast<size_t>(data.size() * 0.99)];
    return std::make_tuple(sum, mean, min, max, p50, p90, p95, p99);
}

int main(int argc, char **argv)
{
    char szInFilePath[256] = "";
    int iGpu = 0;
    int nThread = 5;
    int nTotalVideo = 100;
    bool bSingle = false;
    bool bHost = false;
    std::string inputFilesListPath = "";
    
    try
    {
        // Parse the command line arguments
        ParseCommandLine(argc, argv, szInFilePath, iGpu, nThread, nTotalVideo, bSingle, bHost, inputFilesListPath);

        std::vector<std::string> files;
        if (inputFilesListPath.size() != 0)
        {
            auto videofiles = ReadMultipleVideoFiles(inputFilesListPath);
            int smallerSize = videofiles.size();

            if (nTotalVideo > smallerSize)
            {
                // files.resize(nTotalVideo);
                int numIterations = nTotalVideo / smallerSize;

                for (int i = 0; i < numIterations; i++)
                {
                    files.insert(files.end(), videofiles.begin(), videofiles.end());
                }

                int remainingElements = nTotalVideo - (numIterations * smallerSize);
                files.insert(files.end(), videofiles.begin(), videofiles.begin() + remainingElements);
            }
            else
            {
                files = videofiles;
            }

            std::cout << "Multifile mode - " << nTotalVideo << "videos will be decoded" << std::endl;
        }
        else
        {
            for (int i = 0; i < nTotalVideo; i++)
            {
                files.push_back(std::string(szInFilePath));
            }
        }
        // Initialize the thread pool and decoder context
        std::vector<NvDecoder *> vDec;
        InitializeContext(vDec, iGpu, nThread, bSingle, bHost);
        std::vector<int> vnFrame;
        std::vector<std::exception_ptr> vExceptionPtrs;
        vnFrame.resize(nTotalVideo, 0);
        vExceptionPtrs.resize(nTotalVideo);
        std::vector<std::future<float>> decodeLatencyFutures;
        ThreadPool threadPool(nThread);
        // Enqueue the video decoding task into thread pool
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < nTotalVideo; i++)
        {
            auto filePath = files[i].c_str();
            CheckInputFile(filePath);
            decodeLatencyFutures.push_back(threadPool.enqueue(DecodeVideo, vDec, filePath, &vnFrame[i], std::ref(vExceptionPtrs[i])));
        }
        // Wait until decoding tasks finished
        int nTotalFrames = 0;
        std::vector<double> latencies;
        for (int i = 0; i < nTotalVideo; i++)
        {
            auto decodeLatency = decodeLatencyFutures[i].get();
            latencies.push_back(decodeLatency);
            nTotalFrames += vnFrame[i];
        }
        // Calculated the metrics
        auto elapsedTime = (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count())/ 1000.0f;
        double sum, mean, min, max, p50, p90, p95, p99;
        std::tie(sum, mean, min, max, p50, p90, p95, p99) = CalLatencyMetrics(latencies);
        std::cout << "Total Frames Decoded=" << nTotalFrames << " FPS=" << nTotalFrames / elapsedTime << " LatencyPerFrame=" << elapsedTime / nTotalFrames * 1000 << " Mean Latency for each video=" << mean * 1000 << " P50 Latency=" << p50 * 1000 << " P90 Latency=" << p90 * 1000 << " P95 Latency=" << p95 * 1000 << " P99 Latency=" << p99 * 1000 << "ms" << std::endl;
        // Deinitialization
        for (int i = 0; i < nThread; i++)
        {
            delete (vDec[i]);
        }
        ck(cuProfilerStop());

        for (int i = 0; i < nThread; i++)
        {
            if (vExceptionPtrs[i])
            {
                std::rethrow_exception(vExceptionPtrs[i]);
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what();
        exit(1);
    }
    return 0;
}
