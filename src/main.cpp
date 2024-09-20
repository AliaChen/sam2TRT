#include "encoder_decoder_pipeline.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <encoder_model_path> <decoder_model_path>" << std::endl;
        return 1;
    }

    std::string encoderPath = argv[1];
    std::string decoderPath = argv[2];

    try {
        TRT::EncoderDecoderPipeline pipeline(encoderPath, decoderPath);
        
        // 假设我们要处理的图像在 "test_images" 目录中
        pipeline.processImages("test_images");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}