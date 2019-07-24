# 按惯例，cmake的版本
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
# 项目名称
PROJECT(TestCuda)
# cmake寻找cuda，这个要现在系统里面装好cuda，设置好cuda的环境参数啥的
FIND_PACKAGE(CUDA REQUIRED)


find_package(ITK REQUIRED)
include(${ITK_USE_FILE})


# C++和CUDA的编译参数，可选。
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
# 头文件路径，按需
INCLUDE_DIRECTORIES(
	"C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1/7_CUDALibraries/common/UtilNPP"
	"C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1/common/inc"
	"C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1/7_CUDALibraries/common/FreeImage/include"
)
# 库文件路径，按需
LINK_DIRECTORIES(
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64"
	"C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1/7_CUDALibraries/common/FreeImage/lib/x64"
)
# 主要就是这个，教cmake去找nvcc来编译这些东西
CUDA_ADD_EXECUTABLE(TestCuda
    main.cu
)
# 链接外部库，按需
TARGET_LINK_LIBRARIES(TestCuda
 ${CUDA_cusolver_LIBRARY} ${ITK_LIBRARIES} nppc.lib nppisu.lib nppif.lib FreeImage.lib
)