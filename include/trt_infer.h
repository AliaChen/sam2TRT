#pragma once
#ifndef  TRT_INFER_H
#define  TRT_INFER_H
//tensorrt 
#include <NvInferRuntime.h>

//opencv 
#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <iostream>
#include<vector>
#include <fstream>
#include<memory>
#include<future>

namespace TRT {

	using namespace std;
	bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

	#define checkRuntime(op) __check_cuda_runtime(op, #op, __FILE__, __LINE__)

	class TRTLogger : public nvinfer1::ILogger {
	public:

		inline const char* severity_string(nvinfer1::ILogger::Severity t) {
			switch (t) {
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
			case nvinfer1::ILogger::Severity::kERROR:   return "error";
			case nvinfer1::ILogger::Severity::kWARNING: return "warning";
			case nvinfer1::ILogger::Severity::kINFO:    return "info";
			case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
			default: return "unknow";
			}
		};

		virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
			if (severity <= Severity::kWARNING) {
				// ӡɫַʽ£
							   // printf("\033[47;33mӡı\033[0m");
							   // \033[ ʼ
							   //      47    Ǳɫ
							   //      ;     ָ
							   //      33    ɫ
							   //      m     ʼǽ
							   //      \033[0m ֹ
							   // бɫɫɲд
							   // ɫ https://blog.csdn.net/ericbar/article/details/79652086
				if (severity == Severity::kWARNING) {
					printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
				}
				else if (severity <= Severity::kERROR) {
					printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
				}
				else {
					printf("%s: %s\n", severity_string(severity), msg);
				}
			}
		}
	};


	class TRTEngine {

	public:

		//ָģ庯
		template<typename _T>
		shared_ptr<_T> make_nvshared(_T* ptr) {
			return shared_ptr<_T>(ptr, [](_T* p) {p->destroy(); });}

		// Engine
		TRTEngine(const std::string& modelPath, TRTLogger& logger);
		~TRTEngine();

		//load binaty model data
		vector<unsigned char> load_model(const string& modelPath);

		// ȡengine context stream
		nvinfer1::ICudaEngine* getEngine() { return engine.get(); };
		nvinfer1::IExecutionContext* getContext() { return context.get(); };
		cudaStream_t getStream() { return stream; }
	private:
		shared_ptr<nvinfer1::IRuntime> runtime;
		shared_ptr<nvinfer1::ICudaEngine> engine;
		shared_ptr<nvinfer1::IExecutionContext> context;
		cudaStream_t stream;

	

	};

	class mixMemory {
	public:
		mixMemory();
		mixMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size);
		virtual ~mixMemory();
		void* get_gpu(size_t size);
		void* get_cpu(size_t size);
		void release_gpu();
		void release_cpu();
		void release_all();

		void* get_gpu() { return gpu_data; };
		void* get_cpu() { return cpu_data; };

		template<typename _T>
		_T* get_gpu(size_t size) { return (_T*)get_gpu(size * sizeof(_T)); }

		template<typename _T>
		_T* get_cpu(size_t size) { return (_T*)get_cpu(size * sizeof(_T)); };

		void reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size);
	private:
		void* gpu_data = nullptr;
		void* cpu_data = nullptr;
		size_t gpu_size_ = 0;
		size_t cpu_size_ = 0;

	};

	class ImageProcessor {
	public:
		ImageProcessor();

		float* preprocess_ima(cv::Mat &image);
		//cv::Mat post_process_output(void* data);

	private:
		int encoder_input_w = 1024;
		int encoder_input_h = 1024;
		int channels = 3;
		int encoder_input_numel = encoder_input_w * encoder_input_h * channels ;
		mixMemory encoder_input_data;

	};


	class Infer {
	public:
		Infer();
		bool runEncoder(float* bindings[], cudaStream_t stream, nvinfer1::IExecutionContext* context);

	public:
		std::vector<std::unique_ptr<mixMemory>> outputDataList;
		std::vector<float*> output_data_device;
		std::vector<float*> output_data_host;
		std::vector<size_t> output_size_list;

	};

	class Encoder {
	public:
		Encoder(int num_output);
		void initialize(nvinfer1::ICudaEngine* engine);
		bool runEncoderInfer(float* bindings[], cudaStream_t stream, nvinfer1::IExecutionContext* context);

	public:
		std::vector<std::unique_ptr<mixMemory>> outputDataList;
		std::vector<float*> output_data_device;
		std::vector<float*> output_data_host;
		std::vector<size_t> output_size_list;
	};

	class Decoder {

	// decoder Ľṹ
	public:
	
		//struct Params {

		//	float point_coords[2];
		//	float point_label;
		//	std::vector<float> mask_input;
		//	float hast_mask_input;

		//};

	public:
		Decoder(int num_input, int num_output);
		void initialize(nvinfer1::ICudaEngine* engine);
		bool runDecoderInfer(float* bindings[], cudaStream_t stream, nvinfer1::IExecutionContext* context);
		void preare_ponits(int ori_height, int ori_width, float* points);
		void prepare_inputs(std::vector<float*>&input_data_device, float* points, size_t* image_size, float &labels, cudaStream_t stream);

	public:
		std::vector<std::unique_ptr<mixMemory>>inputDataList;
		std::vector<std::unique_ptr<mixMemory>>outputDataList;
		std::vector<float*> input_data_device;
		std::vector<float*> output_data_device;
		std::vector<float*> output_data_host;
		std::vector<size_t> input_size_list;
		std::vector<size_t> output_size_list;
		//Params params;
		float scale_factor = 4;


	};
};


#endif //  TRT_INFER.H

