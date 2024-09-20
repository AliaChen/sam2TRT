#include "trt_infer.h"
#include <fstream>
#include <opencv2/opencv.hpp>

namespace TRT {

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
    if (code != cudaSuccess) {
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d %s failed. \n code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

TRTEngine::TRTEngine(const std::string& modelPath, TRTLogger& logger) {
    auto model_data = load_model(modelPath);
    runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
    engine = make_nvshared(runtime->deserializeCudaEngine(model_data.data(), model_data.size()));
    context = make_nvshared(engine->createExecutionContext());
    checkRuntime(cudaStreamCreate(&stream));
}

TRTEngine::~TRTEngine() {
    checkRuntime(cudaStreamDestroy(stream));
}

vector<unsigned char> TRTEngine::load_model(const std::string& modelPath) {
    std::ifstream in(modelPath, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        return {};
    }

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

mixMemory::mixMemory() {}

mixMemory::mixMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size) {
    reference_data(cpu, cpu_size, gpu, gpu_size);
}

mixMemory::~mixMemory() {
    release_all();
}

void mixMemory::reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size) {
    release_all();
    if (cpu == nullptr || cpu_size == 0) {
        cpu = nullptr;
        cpu_size = 0;
    }

    if (gpu == nullptr || gpu_size == 0) {
        gpu = nullptr;
        gpu_size = 0;
    }

    this->cpu_data = cpu;
    this->cpu_size_ = cpu_size;
    this->gpu_data = gpu;
    this->gpu_size_= gpu_size;
}

void mixMemory::release_all() {
    release_gpu();
    release_cpu();
}

void mixMemory::release_gpu() {
    if (gpu_data) {
        checkRuntime(cudaFree(gpu_data));
    }
    gpu_data = nullptr;
}

void mixMemory::release_cpu() {
    if (cpu_data) {
        checkRuntime(cudaFreeHost(cpu_data));
    }
    cpu_data = nullptr;
}

void* mixMemory::get_gpu(size_t size) {
    std::cout << " gpu _size_" <<gpu_size_ <<endl;
    if (gpu_size_ < size) {
        release_gpu();
        gpu_size_ = size;
        checkRuntime(cudaMalloc(&gpu_data, size));
        checkRuntime(cudaMemset(gpu_data, 0, size));
    }
    return gpu_data;
}

void* mixMemory::get_cpu(size_t size) {
    if (cpu_size_ < size) {
        release_cpu();
        cpu_size_ = size;
        checkRuntime(cudaMallocHost(&cpu_data, size));
        memset(cpu_data, 0, size);
    }
    return cpu_data;
}

ImageProcessor::ImageProcessor() {}

float* ImageProcessor::preprocess_ima(cv::Mat &image) {
    float mean_value[3] = { 0.485f, 0.456f , 0.406f };
    float std_value[3] = { 0.229f, 0.224f, 0.225f };

    cv::Mat input_image(encoder_input_w, encoder_input_h, CV_8UC3);
    cv::resize(image, input_image, cv::Size(encoder_input_w, encoder_input_h));
    int image_area = input_image.cols * input_image.rows;

    float* input_data_host = encoder_input_data.get_cpu<float>(encoder_input_numel);

    unsigned char* pimage = input_image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;

    for (int i = 0; i < image_area; ++i, pimage += 3) {
        *phost_r++ = (pimage[0] / 255.0f - mean_value[0]) / std_value[0];
        *phost_g++ = (pimage[1] / 255.0f - mean_value[1]) / std_value[1];
        *phost_b++ = (pimage[2] / 255.0f - mean_value[2]) / std_value[2];
    }

    float* input_data_device = encoder_input_data.get_gpu<float>(encoder_input_numel);
    checkRuntime(cudaMemcpy(input_data_device, input_data_host, encoder_input_numel * sizeof(float), cudaMemcpyHostToDevice));

    return input_data_device;
}

Infer::Infer() {}

bool Infer::runEncoder(float* bindings[], cudaStream_t stream, nvinfer1::IExecutionContext* context) {
    return context->enqueueV2((void**)bindings, stream, nullptr);
}

Encoder::Encoder(int num_output) {
    outputDataList.resize(num_output);
    output_data_device.resize(num_output);
    output_data_host.resize(num_output);
    output_size_list.resize(num_output);
}

bool Encoder::runEncoderInfer(float* bindings[], cudaStream_t stream, nvinfer1::IExecutionContext* context) {
    return context->enqueueV2((void**)bindings, stream, nullptr);
}

Decoder::Decoder(int num_input, int num_output) {
    inputDataList.resize(num_input);
    input_data_device.resize(num_input);
    input_size_list.resize(num_input);

    outputDataList.resize(num_output);
    output_data_device.resize(num_output);
    output_data_host.resize(num_output);
    output_size_list.resize(num_output);
}

void Decoder::preare_ponits(int ori_height, int ori_width, float* points) {
    points[0] = points[0] / ori_width * 1024;
    points[1] = points[1] / ori_height * 1024;
}

void Decoder::prepare_inputs(std::vector<float*>& input_data_device, float* points, size_t* image_size, float& labels, cudaStream_t stream) {
    std::vector<float> mask_input(1 * 1024 / scale_factor * 1024 / scale_factor, 0);
    size_t has_mask_input = 0;
    preare_ponits(image_size[0], image_size[1], points);
    checkRuntime(cudaMemcpyAsync(input_data_device[0], points, 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(input_data_device[1], &labels, 1 * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(input_data_device[2], mask_input.data(), 1 * 1024 / scale_factor * 1024 / scale_factor * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(input_data_device[3], &has_mask_input, 1 * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
}

bool Decoder::runDecoderInfer(float* bindings[], cudaStream_t stream, nvinfer1::IExecutionContext* context) {
    return context->enqueueV2((void**)bindings, stream, nullptr);
}

void Decoder::initialize(nvinfer1::ICudaEngine* engine) {
    for (int i = 0; i < inputDataList.size(); ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        inputDataList[i] = std::make_unique<mixMemory>();
        input_data_device[i] = inputDataList[i]->get_gpu<float>(size);
        input_size_list[i] = size;
    }

    for (int i = 0; i < outputDataList.size(); ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i + inputDataList.size());
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        outputDataList[i] = std::make_unique<mixMemory>();
        output_data_device[i] = outputDataList[i]->get_gpu<float>(size);
        output_data_host[i] = outputDataList[i]->get_cpu<float>(size);
        output_size_list[i] = size;
    }
}

void Encoder::initialize(nvinfer1::ICudaEngine* engine) {
    int numBindings = engine->getNbBindings();
    int outputIndex = 0;
    for (int i = 0; i < numBindings; ++i) {
        if (!engine->bindingIsInput(i)) {
            nvinfer1::Dims dims = engine->getBindingDimensions(i);
            size_t size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                size *= dims.d[j];
            }
            outputDataList[outputIndex] = std::make_unique<mixMemory>();
            output_data_device[outputIndex] = outputDataList[outputIndex]->get_gpu<float>(size);
            output_data_host[outputIndex] = outputDataList[outputIndex]->get_cpu<float>(size);
            output_size_list[outputIndex] = size;
            outputIndex++;
        }
    }
}

} // namespace TRT