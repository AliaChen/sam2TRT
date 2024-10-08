#include "encoder_decoder_pipeline.h"
#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <points_json>" << std::endl;
        return 1;
    }

    std::string pointsJson = argv[1];
    json points = json::parse(pointsJson);

    // 处理点信息
    std::vector<std::pair<float, float>> pointData;
    for (const auto& point : points) {
        pointData.emplace_back(point["x"], point["y"]);
    }

    // 初始化模型
    std::string encoderPath = "path/to/encoder_model";
    std::string decoderPath = "path/to/decoder_model";
    try {
        TRT::EncoderDecoderPipeline pipeline(encoderPath, decoderPath);
        // 使用点信息进行模型操作
        std::string resultImagePath = pipeline.processPoints(pointData);
        std::cout << resultImagePath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}