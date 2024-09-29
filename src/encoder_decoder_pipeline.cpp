#include "encoder_decoder_pipeline.h"
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <chrono>

namespace TRT {

EncoderDecoderPipeline::EncoderDecoderPipeline(const std::string& encoderPath, const std::string& decoderPath)
    : logger(),  // 初始化 logger
      encoderEngine(std::make_unique<TRTEngine>(encoderPath, logger)),
      decoderEngine(std::make_unique<TRTEngine>(decoderPath, logger)),
      imageProcessor(std::make_unique<ImageProcessor>()) {
    initializeEncoder();
    initializeDecoder();
    decoderFuture = std::async(std::launch::async, &EncoderDecoderPipeline::runDecoderTask, this);
}

EncoderDecoderPipeline::~EncoderDecoderPipeline() {
    stop();
    if (decoderFuture.valid()) {
        decoderFuture.wait();
    }
}

void EncoderDecoderPipeline::initializeEncoder() {
    encoder = std::make_unique<Encoder>(3);
    encoder->initialize(encoderEngine->getEngine());
}

void EncoderDecoderPipeline::initializeDecoder() {
    decoder = std::make_unique<Decoder>(7, 2);
    decoder->initialize(decoderEngine->getEngine());
}

void EncoderDecoderPipeline::runEncoderTask(const cv::Mat& image) {
    size_t ori_height = image.rows;
    size_t ori_width = image.cols;

    float* input_data_device = imageProcessor->preprocess_ima(const_cast<cv::Mat&>(image));
    
    float* bindings[] = {input_data_device, encoder->output_data_device[0], 
                         encoder->output_data_device[1], encoder->output_data_device[2]};
    
    auto st_time = std::chrono::high_resolution_clock::now();
    bool status = encoder->runEncoderInfer(bindings, encoderEngine->getStream(), encoderEngine->getContext());
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - st_time);
    //std::cout << "Encoder consume time: " << duration.count() << " ms" << std::endl;
    if (status) {
        //std::cout << " encoder infer done." <<endl;
        EncoderOutput output;
        output.data = {encoder->output_data_device[0], 
                       encoder->output_data_device[1], 
                       encoder->output_data_device[2]};
        output.ori_height = ori_height;
        output.ori_width = ori_width;
        {
            std::lock_guard<std::mutex> lock(mtx);
            encoderOutputQueue.push(std::move(output));
        }
        cv.notify_one();
    }
}

void EncoderDecoderPipeline::runDecoderTask() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return !encoderOutputQueue.empty() || done; });
        
        if (done && encoderOutputQueue.empty()) {
            break;
        }

        auto encoderOutput = std::move(encoderOutputQueue.front());
        encoderOutputQueue.pop();
        std::cout << "encoderOutputQueue size: " << encoderOutputQueue.size() << std::endl;
        //float point_coords[2] = {800, 440}; // 示例坐标
        //float labels = 1.0f;
        size_t image_size[2] = {encoderOutput.ori_height, encoderOutput.ori_width};
        //decoder->prepare_inputs(decoder->input_data_device, point_coords, image_size, labels, decoderEngine->getStream());
        std::vector<std::array<float,2>> point_coords_list = { {800, 440}, {200,50}};
        std::vector<float> labels = { 1.0f, 1.0f};

        for (int p = 0; p < point_coords_list.size(); ++p){
            decoder->prepare_inputs(decoder->input_data_device, point_coords_list[p].data(), image_size, labels[p], decoderEngine->getStream());
              
            // 更新绑定数组以包含所有7个输入和2个输出
            float* bindings[] = {
                encoderOutput.data[0], encoderOutput.data[2], encoderOutput.data[1],
                decoder->input_data_device[0], decoder->input_data_device[1],
                decoder->input_data_device[2], decoder->input_data_device[3],
                decoder->output_data_device[0], decoder->output_data_device[1]

                
            };

            bool status = decoder->runDecoderInfer(bindings, decoderEngine->getStream(), decoderEngine->getContext());

        

            if (!status) {
                std::cerr << "Decoder inference failed" << std::endl;
            } else {
                // 将输出从设备内存复制到主机内存
                for (int i = 0; i < decoder->outputDataList.size(); ++i) {
                    checkRuntime(cudaMemcpyAsync(decoder->output_data_host[i], 
                                                decoder->output_data_device[i],
                                                decoder->output_size_list[i] * sizeof(float),
                                                cudaMemcpyDeviceToHost,
                                                decoderEngine->getStream()));
                }
                checkRuntime(cudaStreamSynchronize(decoderEngine->getStream()));

            }

            for (int n = 0; n < 3; ++n) {
                std::cout << p  <<"   Score: " << decoder->output_data_host[0][n] << std::endl;
            }

        
        }
        lock.unlock();

        // // 更新绑定数组以包含所有7个输入和2个输出
        // float* bindings[] = {
        //     encoderOutput.data[0], encoderOutput.data[2], encoderOutput.data[1],
        //     decoder->input_data_device[0], decoder->input_data_device[1],
        //     decoder->input_data_device[2], decoder->input_data_device[3],
        //     decoder->output_data_device[0], decoder->output_data_device[1]

            
        // };

        
   
        // 处理输出...
  
        // cudaError_t error = cudaGetLastError();
        // if (error != cudaSuccess) {
        //     std::cerr << "CUDA error before inference: " << cudaGetErrorString(error) << std::endl;
        // }

        // bool status = decoder->runDecoderInfer(bindings, decoderEngine->getStream(), decoderEngine->getContext());

        // error = cudaGetLastError();
        // if (error != cudaSuccess) {
        //     std::cerr << "CUDA error after inference: " << cudaGetErrorString(error) << std::endl;
        // }

        // if (!status) {
        //     std::cerr << "Decoder inference failed" << std::endl;
        // } else {
        //     // 将输出从设备内存复制到主机内存
        //     for (int i = 0; i < decoder->outputDataList.size(); ++i) {
        //         checkRuntime(cudaMemcpyAsync(decoder->output_data_host[i], 
        //                                      decoder->output_data_device[i],
        //                                      decoder->output_size_list[i] * sizeof(float),
        //                                      cudaMemcpyDeviceToHost,
        //                                      decoderEngine->getStream()));
        //     }
        //     checkRuntime(cudaStreamSynchronize(decoderEngine->getStream()));

        // }

        // for (int n = 0; n < 3; ++n) {
        //     std::cout << "Score: " << decoder->output_data_host[0][n] << std::endl;
        // }



    }
}

void EncoderDecoderPipeline::processImages(const std::string& rootDir) {
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(rootDir.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.find(".jpg") != std::string::npos || filename.find(".png") != std::string::npos) {
                std::string imagePath = rootDir + "/" + filename;
                cv::Mat image = cv::imread(imagePath);
                if (!image.empty()) {
                    std::cout << "Processing file: " << filename << std::endl;
                    runEncoderTask(image);
                }
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Could not open directory: " << rootDir << std::endl;
    }
}

void EncoderDecoderPipeline::stop() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        done = true;
    }
    cv.notify_all();
}

} // namespace TRT