// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <cudaProfiler.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdio.h>
#include <string.h>
#include <string>
#include <thread>

#include "../Utils/FFmpegDemuxer.h"
#include "../Utils/NvCodecUtils.h"
#include "OptimizedNvDecoder.h"
#include "ThreadPoolUtils.h"

// Define logger which need in third party utils
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

// Define the codec map
std::map<std::string, cudaVideoCodec_enum> codecMap = {
    {"mpeg1", cudaVideoCodec_MPEG1},       {"mpeg2", cudaVideoCodec_MPEG2},       {"mpeg4", cudaVideoCodec_MPEG4},
    {"vc1", cudaVideoCodec_VC1},           {"h264", cudaVideoCodec_H264},         {"jpeg", cudaVideoCodec_JPEG},
    {"h264_svc", cudaVideoCodec_H264_SVC}, {"h264_mvc", cudaVideoCodec_H264_MVC}, {"hevc", cudaVideoCodec_HEVC},
    {"vp8", cudaVideoCodec_VP8},           {"vp9", cudaVideoCodec_VP9},           {"av1", cudaVideoCodec_AV1}};

/**
 *   @brief  Function to decode media file using OptimizedNvDecoder interface
 *   @param  pDec    - Handle to OptimizedNvDecoder
 *   @param  demuxer - Pointer to an FFmpegDemuxer instance
 *   @param  pnFrame - Variable to record the number of frames decoded
 *   @param  ex      - Stores current exception in case of failure
 */
void DecProc(OptimizedNvDecoder *pDec, const char *szInFilePath, int *pnFrame, std::exception_ptr &ex) {
    try {
        std::unique_ptr<FFmpegDemuxer> demuxer(new FFmpegDemuxer(szInFilePath));
        int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
        uint8_t *pVideo = NULL, *pFrame = NULL;

        do {
            demuxer->Demux(&pVideo, &nVideoBytes);
            nFrameReturned = pDec->Decode(pVideo, nVideoBytes);
            if (!nFrame && nFrameReturned)
                LOG(INFO) << pDec->GetVideoInfo();

            nFrame += nFrameReturned;
        } while (nVideoBytes);
        *pnFrame = nFrame;
    } catch (std::exception &) {
        ex = std::current_exception();
    }
}

void ShowHelpAndExit(const char *szBadOption = NULL) {
    std::ostringstream oss;
    bool bThrowError = false;
    if (szBadOption) {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "-i           Input file path" << std::endl
        << "-o           Output file path" << std::endl
        << "-gpu         Ordinal of GPU to use" << std::endl
        << "-thread      Number of decoding thread" << std::endl
        << "-total       Number of total video to test" << std::endl
        << "-single      (No value) Use single context (this may result in suboptimal performance; default is multiple "
           "contexts)"
        << std::endl
        << "-host        (No value) Copy frame to host memory (this may result in suboptimal performance; default is "
           "device memory)"
        << std::endl
        << "-multi_input Multiple Input file list path" << std::endl
        << "-codec       The codecc of video to test" << std::endl;
    if (bThrowError) {
        throw std::invalid_argument(oss.str());
    } else {
        std::cout << oss.str();
        exit(0);
    }
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, int &iGpu, int &nThread, int &nTotalVideo,
                      bool &bSingle, bool &bHost, std::string &inputFilesListPath, std::string &outputFile,
                      cudaVideoCodec &codec) {
    for (int i = 1; i < argc; i++) {
        if (!_stricmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            outputFile = std::string(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-gpu")) {
            if (++i == argc) {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-thread")) {
            if (++i == argc) {
                ShowHelpAndExit("-thread");
            }
            nThread = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-total")) {
            if (++i == argc) {
                ShowHelpAndExit("-total");
            }
            nTotalVideo = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-multi_input")) {
            if (++i == argc) {
                ShowHelpAndExit("-multi_input");
            }
            inputFilesListPath = std::string(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-single")) {
            bSingle = true;
            continue;
        }
        if (!_stricmp(argv[i], "-host")) {
            bHost = true;
            continue;
        }
        if (!_stricmp(argv[i], "-codec")) {
            if (++i == argc) {
                ShowHelpAndExit("-codec");
            }
            std::string codecName = std::string(argv[i]);
            std::transform(codecName.begin(), codecName.end(), codecName.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (codecMap.find(codecName) != codecMap.end()) {
                codec = codecMap[codecName];
            } else {
                std::cout << "Codec name not found in the map." << std::endl;
                exit(1);
            }
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

OptimizedNvDecoder *InitOptimizedNvDecoder(int i, const CUdevice &cuDevice, CUcontext &cuContext, bool bSingle,
                                           bool bHost, cudaVideoCodec codec, CUVIDDECODECAPS decodecaps) {
    if (!bSingle) {
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
    }
    OptimizedNvDecoder *sessionObject = new OptimizedNvDecoder(cuContext, !bHost, codec, decodecaps);
    sessionObject->setDecoderSessionID(i);
    return sessionObject;
}

std::string GetTime(const std::chrono::_V2::system_clock::time_point &now) {
    // Convert the time_point to a time_t
    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    // Convert the time_t to a human-readable format
    std::tm *now_tm = std::localtime(&now_time_t);
    char time_cstr[100];
    strftime(time_cstr, sizeof(time_cstr), "%Y-%m-%d %H:%M:%S", now_tm);
    std::string time_str(time_str);
    return time_str;
}

double DecodeVideo(size_t i, const std::vector<OptimizedNvDecoder *> &vDec, const char *szInFilePath, int *pnFrame,
                   std::exception_ptr &ex) {
    OptimizedNvDecoder *pDec = vDec[i];
    auto start = std::chrono::high_resolution_clock::now();
    DecProc(pDec, szInFilePath, pnFrame, ex);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Decode finished -- start:" << GetTime(start) << " end:" << GetTime(end) << " duration:" << elapsedTime
              << " frames:" << *pnFrame << std::endl;
    return elapsedTime / 1000.0f;
}

std::vector<std::string> ReadMultipleVideoFiles(std::string filepath) {
    std::ifstream file(filepath);
    if (!file) {
        std::cerr << "Error opening the file." << std::endl;
        exit(1);
    }
    std::string line;
    std::vector<std::string> tokens;
    while (std::getline(file, line)) {
        tokens.push_back(line);
    }
    file.close();
    return tokens;
}

void GetDefaultDecoderCaps(CUVIDDECODECAPS &decodecaps, cudaVideoCodec codec) {
    memset(&decodecaps, 0, sizeof(decodecaps));
    decodecaps.eCodecType = codec;
    decodecaps.eChromaFormat = cudaVideoChromaFormat_420;
    decodecaps.nBitDepthMinus8 = 0;
    NVDEC_API_CALL(cuvidGetDecoderCaps(&decodecaps));
}

void InitializeContext(std::vector<OptimizedNvDecoder *> &vDec, int iGpu, int nThread, bool bSingle, bool bHost,
                       cudaVideoCodec codec) {
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
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

    CUVIDDECODECAPS decodecaps;
    GetDefaultDecoderCaps(decodecaps, codec);

    ThreadPool threadPool(nThread);
    std::vector<std::future<OptimizedNvDecoder *>> futures;
    for (int i = 0; i < nThread; i++) {
        futures.push_back(
            threadPool.enqueue(InitOptimizedNvDecoder, cuDevice, cuContext, bSingle, bHost, codec, decodecaps));
    }
    for (auto &future : futures) {
        vDec.push_back(future.get()); // Retrieve the results from each task
    }
}

void WriteRawData(const std::vector<double> &data, std::vector<int> &frames, std::string filename) {
    // Open the output file stream
    std::ofstream outputFile(filename);
    outputFile << "latency" << std::endl;
    for (int i = 0; i < data.size(); i++) {
        outputFile << data[i] << std::endl;
    }
    outputFile << "FPS" << std::endl;
    for (int i = 0; i < data.size(); i++) {
        outputFile << frames[i] / data[i] << std::endl;
    }
    // Close the file stream
    outputFile.close();
}

std::tuple<double, double, double, double, double, double, double, double>
CalLatencyMetrics(const std::vector<double> &originData) {
    std::vector<double> data = originData;
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

std::vector<std::string> GenerateTotalFileList(std::string inputFilesListPath, int nTotalVideo,
                                               const char *szInFilePath) {
    std::vector<std::string> files;
    if (inputFilesListPath.size() != 0) {
        auto videofiles = ReadMultipleVideoFiles(inputFilesListPath);
        int smallerSize = videofiles.size();

        if (nTotalVideo > smallerSize) {
            int numIterations = nTotalVideo / smallerSize;

            for (int i = 0; i < numIterations; i++) {
                files.insert(files.end(), videofiles.begin(), videofiles.end());
            }

            int remainingElements = nTotalVideo - (numIterations * smallerSize);
            files.insert(files.end(), videofiles.begin(), videofiles.begin() + remainingElements);
        } else {
            files = videofiles;
        }

        std::cout << "Multifile mode - " << nTotalVideo << "videos will be decoded" << std::endl;
    } else {
        for (int i = 0; i < nTotalVideo; i++) {
            files.push_back(std::string(szInFilePath));
        }
    }
    return files;
}

float run(std::vector<OptimizedNvDecoder *> &vDec, int nThread, std::vector<std::string> &files,
          std::vector<int> &vnFrame, std::vector<std::exception_ptr> &vExceptionPtrs, int *nTotalFrames,
          std::vector<double> &vnLatency) {
    std::vector<std::future<double>> decodeLatencyFutures;
    ThreadPool threadPool(nThread);
    // Enqueue the video decoding task into thread pool
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < files.size(); i++) {
        auto filePath = files[i].c_str();
        CheckInputFile(filePath);
        decodeLatencyFutures.push_back(
            threadPool.enqueue(DecodeVideo, vDec, filePath, &vnFrame[i], std::ref(vExceptionPtrs[i])));
    }
    // Wait until decoding tasks finished
    for (int i = 0; i < files.size(); i++) {
        auto decodeLatency = decodeLatencyFutures[i].get();
        vnLatency.push_back(decodeLatency);
        nTotalFrames += vnFrame[i];
    }
    // Calculated the metrics
    return (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
                .count()) /
           1000.0f;
}

int main(int argc, char **argv) {
    char szInFilePath[256] = "";
    int iGpu = 0;
    int nThread = 5;
    int nTotalVideo = 100;
    bool bSingle = false;
    bool bHost = false;
    std::string inputFilesListPath = "";
    std::string outputFilePath = "";
    std::vector<std::exception_ptr> vExceptionPtrs(nTotalVideo);
    cudaVideoCodec codec = cudaVideoCodec_H264;
    try {
        // Parse the command line arguments
        ParseCommandLine(argc, argv, szInFilePath, iGpu, nThread, nTotalVideo, bSingle, bHost, inputFilesListPath,
                         outputFilePath, codec);
        auto files = GenerateTotalFileList(inputFilesListPath, nTotalVideo, szInFilePath);

        // Initialize and prepare the decoder context for each thread
        std::vector<OptimizedNvDecoder *> vDec;
        InitializeContext(vDec, iGpu, nThread, bSingle, bHost, codec);

        // Decode all video with thread pool
        std::vector<int> vnFrame(nTotalVideo);
        int nTotalFrames = 0;
        std::vector<double> vnLatency;

        auto elapsedTime = run(vDec, nThread, files, vnFrame, vExceptionPtrs, &nTotalFrames, vnLatency);

        // Calculate and output the raw data into file and metrics into stdout
        double sum, mean, min, max, p50, p90, p95, p99;
        std::tie(sum, mean, min, max, p50, p90, p95, p99) = CalLatencyMetrics(vnLatency);
        if (outputFilePath.size() != 0) {
            WriteRawData(vnLatency, vnFrame, outputFilePath);
        }
        std::cout << "Total Frames Decoded=" << nTotalFrames << " FPS=" << nTotalFrames / elapsedTime
                  << " LatencyPerFrame=" << elapsedTime / nTotalFrames * 1000
                  << " Mean Latency for each video=" << mean * 1000 << " P50 Latency=" << p50 * 1000
                  << " P90 Latency=" << p90 * 1000 << " P95 Latency=" << p95 * 1000 << " P99 Latency=" << p99 * 1000
                  << "ms" << std::endl;

        // Deinitialization
        for (int i = 0; i < nThread; i++) {
            delete (vDec[i]);
        }
        for (int i = 0; i < nThread; i++) {
            if (vExceptionPtrs[i]) {
                std::rethrow_exception(vExceptionPtrs[i]);
            }
        }
    } catch (const std::exception &ex) {
        std::cout << ex.what();
        exit(1);
    }
    return 0;
}
