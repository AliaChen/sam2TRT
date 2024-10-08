#pragma once

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include "trt_infer.h"

namespace TRT {

struct EncoderOutput {
    std::vector<float*> data;
    size_t ori_height;
    size_t ori_width;
    cv::Mat ori_mat;
};

struct DecoderOutput {
    std::vector<float*> data;
    cv::Mat ori_mat;
    size_t outscore_size;
    size_t outmask_size;
        size_t ori_height;
    size_t ori_width;
};

class EncoderDecoderPipeline {
public:
    EncoderDecoderPipeline(const std::string& encoderPath, const std::string& decoderPath);
    ~EncoderDecoderPipeline();

    void processImages(const std::string& rootDir);
    void stop();

private:
    void initializeEncoder();
    void initializeDecoder();
    void runEncoderTask(const cv::Mat& image);
    void runDecoderTask();

    TRTLogger logger;  // 添加 logger 成员
    std::unique_ptr<TRTEngine> encoderEngine;
    std::unique_ptr<TRTEngine> decoderEngine;
    std::unique_ptr<Encoder> encoder;
    std::unique_ptr<Decoder> decoder;
    std::unique_ptr<ImageProcessor> imageProcessor;

    std::mutex mtx;
    std::condition_variable cv;
    std::queue<EncoderOutput> encoderOutputQueue;
    std::queue<DecoderOutput> decoderOutputQueue;
    bool done = false;

    std::future<void> decoderFuture;
};

} // namespace TRT