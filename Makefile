# 编译器
CXX= g++

#编译选项
CXXFLAGS = -std=c++14 -Wall -Wextra -O2

#Include 目录
INCLUDES = -I./include -I./3rdpart/opencv4/include -I/mnt/d/Chen/wsl_dir/3rdparts/cuda-11.8/targets/x86_64-linux/include\
		   -I/mnt/d/Chen/wsl_dir/3rdparts/TensorRT-8.6.0.12/targets/x86_64-linux-gnu/include

#Lib 目录
LIBDIRS := -L./3rdpart/opencv4/lib -L/mnt/d/Chen/wsl_dir/3rdparts/cuda-11.8/targets/x86_64-linux/lib\
		   -L/mnt/d/Chen/wsl_dir/3rdparts/TensorRT-8.6.0.12/targets/x86_64-linux-gnu/lib

#拼接路径
library_paths := ./3rdpart/opencv4/lib /mnt/d/Chen/wsl_dir/3rdparts/cuda-11.8/targets/x86_64-linux/lib /mnt/d/Chen/wsl_dir/3rdparts/TensorRT-8.6.0.12/targets/x86_64-linux-gnu/lib
empty :=
LIBRARY_PATH_EXPORT := $(subst $(empty) $(empty),:,$(library_paths))

#依赖库
LIBS = -lnvinfer -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

# 源文件
SRCS := src/main.cpp src/trt_infer.cpp src/encoder_decoder_pipeline.cpp

#.o 文件
OBJS := $(SRCS:.cpp=.o)

#可执行程序
TARGET := sam2_exe

#Default target
all: $(TARGET)

run: $(TARGET)
	./$(TARGET)
#Linking
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LIBDIRS) $^ -o $@ $(LIBS)

# Compilacation
%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

#clean
clean:
	rm -rf $(OBJS) $(TARGET)
show: 
	@echo $(LIBRARY_PATH_EXPORT)

export LD_LIBRARY_PATH:=$(LIBRARY_PATH_EXPORT)