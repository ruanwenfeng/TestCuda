//# include < time.h >
//# include <stdio.h >
//# include <stdlib.h>
//# include <cuda_runtime.h>
//# include <cusolverDn.h>
//#include <Windows.h>
// #include "itkImage.h"
// #include "itksys/SystemTools.hxx"
// #include <iostream>
// #include <fstream>
// #include "itkImageFileReader.h"
// #include "itkHessianRecursiveGaussianImageFilter.h"
// #include "itkTimeProbe.h"
//#include "itkImageRegionConstIterator.h"
////
////
////LARGE_INTEGER
////getFILETIMEoffset()
////{
////	SYSTEMTIME s;
////	FILETIME f;
////	LARGE_INTEGER t;
////
////	s.wYear = 1970;
////	s.wMonth = 1;
////	s.wDay = 1;
////	s.wHour = 0;
////	s.wMinute = 0;
////	s.wSecond = 0;
////	s.wMilliseconds = 0;
////	SystemTimeToFileTime(&s, &f);
////	t.QuadPart = f.dwHighDateTime;
////	t.QuadPart <<= 32;
////	t.QuadPart |= f.dwLowDateTime;
////	return (t);
////}
////
////int
////clock_gettime(int, struct timeval *tv)
////{
////	LARGE_INTEGER           t;
////	FILETIME            f;
////	double                  microseconds;
////	static LARGE_INTEGER    offset;
////	static double           frequencyToMicroseconds;
////	static int              initialized = 0;
////	static BOOL             usePerformanceCounter = 0;
////
////	if (!initialized) {
////		LARGE_INTEGER performanceFrequency;
////		initialized = 1;
////		usePerformanceCounter = QueryPerformanceFrequency(&performanceFrequency);
////		if (usePerformanceCounter) {
////			QueryPerformanceCounter(&offset);
////			frequencyToMicroseconds = (double)performanceFrequency.QuadPart / 1000000.;
////		}
////		else {
////			offset = getFILETIMEoffset();
////			frequencyToMicroseconds = 10.;
////		}
////	}
////	if (usePerformanceCounter) QueryPerformanceCounter(&t);
////	else {
////		GetSystemTimeAsFileTime(&f);
////		t.QuadPart = f.dwHighDateTime;
////		t.QuadPart <<= 32;
////		t.QuadPart |= f.dwLowDateTime;
////	}
////
////	t.QuadPart -= offset.QuadPart;
////	microseconds = (double)t.QuadPart / frequencyToMicroseconds;
////	t.QuadPart = microseconds;
////	tv->tv_sec = t.QuadPart / 1000000;
////	tv->tv_usec = t.QuadPart % 1000000;
////	return (0);
////}
////typedef itk::Image<float, 3>  ImageType;
////typedef itk::SymmetricSecondRankTensor< ImageType::PixelType, ImageType::ImageDimension > HessianPixelType;
////typedef itk::Image< HessianPixelType, ImageType::ImageDimension >  HessianImageType;
////typedef itk::FixedArray< ImageType::PixelType, ImageType::ImageDimension > EigenValueArrayType;
////typedef itk::Matrix< ImageType::PixelType, ImageType::ImageDimension > EigenVectorMatrixType;
////typedef itk::SymmetricEigenAnalysis< HessianPixelType, EigenValueArrayType, EigenVectorMatrixType >   CalculatorType;
////# define MILLION 1000000L ;
////
////int calc2(const itk::SymmetricSecondRankTensor< ImageType::PixelType, ImageType::ImageDimension > *image) {
////	struct timeval start, stop; // variables for timing
////	double accum; // elapsed time variable
////	cusolverDnHandle_t cusolverH;
////	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
////	cudaError_t cudaStat = cudaSuccess;
////	const int m = 2048; // number of rows and columns of A
////	const int lda = m; // leading dimension of A
////	float *A; // mxm matrix
////	float *V; // mxm matrix of eigenvectors
////	float *W; // m- vector of eigenvalues
////	// prepare memory on the host
////	A = (float *)malloc(lda*m * sizeof(float));
////	V = (float *)malloc(lda*m * sizeof(float));
////	W = (float *)malloc(m * sizeof(float));
////	// define random A
////	for (int i = 0; i < lda*m; i++) A[i] = rand() / (float)RAND_MAX;
////	/*A[0] = *(image->Begin());
////	A[1] = *(image->Begin() + 1);
////	A[2] = *(image->Begin() + 2);
////	A[3] = A[1];
////	A[4] = *(image->Begin() + 3);
////	A[5] = *(image->Begin() + 4);
////	A[6] = A[2];
////	A[7] = A[5];
////	A[8] = *(image->Begin() + 5);*/
////	// declare arrays on the device
////	float *d_A; // mxm matrix A on the device
////	float *d_W; // m- vector of eigenvalues on the device
////	int * devInfo; // info on the device
////	float * d_work; // workspace on the device
////	int lwork = 0; // workspace size
////	int info_gpu = 0; // info copied from device to host
////	// create cusolver handle
////	cusolver_status = cusolverDnCreate(&cusolverH);
////	// prepare memory on the device
////	cudaStat = cudaMalloc((void **)& d_A, sizeof(float)* lda*m);
////	cudaStat = cudaMalloc((void **)& d_W, sizeof(float)*m);
////	cudaStat = cudaMalloc((void **)& devInfo, sizeof(int));
////	cudaStat = cudaMemcpy(d_A, A, sizeof(float)* lda*m,
////		cudaMemcpyHostToDevice); // copy A- >d_A
////		// compute eigenvalues and eigenvectors
////	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
////	// use lower left triangle of the matrix
////	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
////	// compute buffer size and prepare workspace
////	cusolver_status = cusolverDnSsyevd_bufferSize(cusolverH,
////		jobz, uplo, m, d_A, lda, d_W, &lwork);
////	cudaStat = cudaMalloc((void **)& d_work, sizeof(float)* lwork);
////	clock_gettime(0, &start); // start timer
////	itk::TimeProbe time;
////	time.Start();
////	// compute the eigenvalues and eigenvectors for a symmetric ,
////	// real mxm matrix ( only the lower left triangle af A is used )
////	cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, m,
////		d_A, lda, d_W, d_work, lwork, devInfo);
////	cudaStat = cudaDeviceSynchronize();
////	time.Stop();
////	std::cout << " eigen value test takes: " << time.GetMean() << " seconds" << std::endl;
////
////	clock_gettime(0, &stop); // stop timer
////
////	accum = (stop.tv_sec - start.tv_sec) + // elapsed time
////		(stop.tv_usec - start.tv_usec) / (double)MILLION;
////	printf(" Ssyevd time : %lf sec .\n", accum); // print elapsed time
////	cudaStat = cudaMemcpy(W, d_W, sizeof(float)*m,
////		cudaMemcpyDeviceToHost); // copy d_W - >W
////	cudaStat = cudaMemcpy(V, d_A, sizeof(float)* lda*m,
////		cudaMemcpyDeviceToHost); // copy d_A - >V
////	cudaStat = cudaMemcpy(&info_gpu, devInfo, sizeof(int),
////		cudaMemcpyDeviceToHost); // copy devInfo - > info_gpu
////	printf(" after syevd : info_gpu = %d\n", info_gpu);
////	printf(" eigenvalues :\n"); // print first eigenvalues
////	for (int i = 0; i < 3; i++) {
////		printf("W[%d] = %E\n", i + 1, W[i]);
////	}
////	// free memory
////	cudaFree(d_A);
////	cudaFree(d_W);
////	cudaFree(devInfo);
////	cudaFree(d_work);
////	cusolverDnDestroy(cusolverH);
////	cudaDeviceReset();
////	return 0;
////
////}
////
////int calc(const itk::SymmetricSecondRankTensor< ImageType::PixelType, ImageType::ImageDimension > *image)
////{
////	/*
////	CUBLAS_FILL_MODE_LOWER) or upper
////(CUBLAS_FILL_MODE_UPPER)
////	*/
////	struct timeval start, stop; // variables for timing
////	double accum; // elapsed time variable
////	cusolverDnHandle_t cusolverH;
////	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
////	cudaError_t cudaStat = cudaSuccess;
////	const int m = 3; // number of rows and columns of A
////	const int lda = m; // leading dimension of A
////	float *A; // mxm matrix
////	float *V; // mxm matrix of eigenvectors
////	float *W; // m- vector of eigenvalues
////	// prepare memory on the host
////	A = (float *)malloc(lda*m * sizeof(float));
////	V = (float *)malloc(lda*m * sizeof(float));
////	W = (float *)malloc(m * sizeof(float));
////	// define random A
////	A[0] = *(image->Begin());
////	A[1] = *(image->Begin() + 1);
////	A[2] = *(image->Begin() + 2);
////	A[3] = A[1];
////	A[4] = *(image->Begin() + 3);
////	A[5] = *(image->Begin() + 4);
////	A[6] = A[2];
////	A[7] = A[5];
////	A[8] = *(image->Begin() + 5);
////	/*A[0] = 1;
////	A[1] = 2;
////	A[2] = 3;
////	A[3] = 2;
////	A[4] = 1;
////	A[5] = 4;
////	A[6] = 3;
////	A[7] = 4;
////	A[8] = 1;*/
////	//for (int i = 0; i < lda*m; i++) A[i] = i / (float)RAND_MAX;
////	 //declare arrays on the device
////	float *d_A; // mxm matrix A on the device
////	float *d_W; // m- vector of eigenvalues on the device
////	int * devInfo ; // info on the device
////	float * d_work; // workspace on the device
////	int lwork = 0; // workspace size
////	int info_gpu = 0; // info copied from device to host
////	// create cusolver handle
////	cusolver_status = cusolverDnCreate(&cusolverH);
////	// prepare memory on the device
////	cudaStat = cudaMalloc((void **)& d_A, sizeof(float)* lda*m);
////	cudaStat = cudaMalloc((void **)& d_W, sizeof(float)*m);
////	cudaStat = cudaMalloc((void **)& devInfo, sizeof(int));
////	cudaStat = cudaMemcpy(d_A, A, sizeof(float)* lda*m,
////		cudaMemcpyHostToDevice); // copy A- >d_A
////		// compute eigenvalues and eigenvectors
////	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
////	// use lower left triangle of the matrix
////	cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
////	// compute buffer size and prepare workspace
////	cusolver_status = cusolverDnSsyevd_bufferSize(cusolverH,
////		jobz, uplo, m, d_A, lda, d_W, &lwork);
////	cudaStat = cudaMalloc((void **)& d_work, sizeof(float)* lwork);
//////	clock_gettime(0, &start); // start timer
////	// compute the eigenvalues and eigenvectors for a symmetric ,
////	// real mxm matrix ( only the lower left triangle af A is used )
////	cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, m,
////		d_A, lda, d_W, d_work, lwork, devInfo);
////	cudaStat = cudaDeviceSynchronize();
////	//clock_gettime(0, &stop); // stop timer
////	//accum = (stop.tv_sec - start.tv_sec) + // elapsed time
////		//(stop.tv_usec - start.tv_usec) / (double)MILLION;
////	//printf(" Ssyevd time : %lf sec .\n", accum); // print elapsed time
////	cudaStat = cudaMemcpy(W, d_W, sizeof(float)*m,
////		cudaMemcpyDeviceToHost); // copy d_W - >W
////	cudaStat = cudaMemcpy(V, d_A, sizeof(float)* lda*m,
////		cudaMemcpyDeviceToHost); // copy d_A - >V
////	cudaStat = cudaMemcpy(&info_gpu, devInfo, sizeof(int),
////		cudaMemcpyDeviceToHost); // copy devInfo - > info_gpu
////	printf(" after syevd : info_gpu = %d\n", info_gpu);
////	printf(" eigenvalues :\n"); // print first eigenvalues
////	/*for (int i = 0; i < m; i++) {
////		printf("W[%d] = %E\n", i + 1, W[i]);
////	}
////
////	printf("\n\n");
////	
////	for (size_t i = 0; i < m; i++)
////	{
////		for (size_t j = 0; j < m; j++)
////		{
////			printf("V[%d] = %E\n", i*m + j, V[i*m+j]);
////		}
////	}
////	printf("\n\n");
////
////	for (size_t i = 0; i < m; i++)
////	{
////		for (size_t j = 0; j < m; j++)
////		{
////			printf("V[%d] = %E\n", j*m + i, V[j*m + i]);
////		}
////	}*/
////	// free memory
////	cudaFree(d_A);
////	cudaFree(d_W);
////	cudaFree(devInfo);
////	cudaFree(d_work);
////	cusolverDnDestroy(cusolverH);
////	cudaDeviceReset();
////	return 0;
////}
////
////
////void CancelLesion()
////{
////
////	std::string outputFolder = "F:/TestCuda/testByfurcationOutput/testEigenValues";
////	itksys::SystemTools::MakeDirectory(outputFolder);
////	std::string valueFile = outputFolder + "/value.txt";
////	std::string vectorFile = outputFolder + "/vector.txt";
////	std::ofstream valueWriter;
////	std::ofstream vectorWriter;
////	valueWriter.open(valueFile.c_str());
////	vectorWriter.open(vectorFile.c_str());
////
////
////	std::string filePath = "F:/TestCuda/preprocessedLiver.nii.gz";
////	//WriteNiftiImage<ImageType>(duplicator->GetOutput(), outputFolderTemplate, "templateImageJXY.nii.gz");
////	typedef itk::ImageFileReader<ImageType> ReaderType;
////	ReaderType::Pointer imageReader = ReaderType::New();
////	imageReader->SetFileName(filePath);
////	try
////	{
////		imageReader->Update();
////	}
////	catch (itk::ExceptionObject &exception)
////	{
////		//std::cerr << "Exception caught during reading file/truth: " << t2FileName << std::endl;
////		std::cerr << exception << std::endl;
////		return;
////	}
////
////	ImageType::Pointer inputImage = imageReader->GetOutput();
////	inputImage->DisconnectPipeline();
////
////	
////	CalculatorType eigenCalculator(ImageType::ImageDimension);
////
////	std::cout << "begin 000" << std::endl;
////	typedef itk::HessianRecursiveGaussianImageFilter<ImageType, HessianImageType> HessianFilterType;
////	typename HessianFilterType::Pointer h_filter = HessianFilterType::New();
////	h_filter->SetInput(inputImage);
////	h_filter->SetSigma(2.0);
////	h_filter->SetNormalizeAcrossScale(true);
////	h_filter->Update();
////
////	std::cout << "begin 001" << std::endl;
////	itk::TimeProbe time;
////	time.Start();
////
////	itk::ImageRegionConstIterator< ImageType > iIt(inputImage, inputImage->GetLargestPossibleRegion());
////	itk::ImageRegionConstIterator< HessianImageType > hessianIt(h_filter->GetOutput(), h_filter->GetOutput()->GetLargestPossibleRegion());
////	float * A = new float[9];
////	float *V = new float[3*3 * sizeof(float)]; // mxm matrix of eigenvectors
////	float *W = new float[3 * sizeof(float)];
////	//V = (float *)malloc(lda*m * sizeof(float));
//////W = (float *)malloc(m * sizeof(float));
////	for (hessianIt.GoToBegin(), iIt.GoToBegin(); !hessianIt.IsAtEnd(); ++hessianIt, ++iIt)
////	{
////		if (iIt.Get() < 50) continue;
////
////		/*for (auto it = vector.Begin(); it != vector.End(); ++it, ++i)
////		{
////			std::cout << *it << " ";
////		}
////		std::cout << std::endl;
////		for (size_t i = 0; i < 9; i++)
////		{
////			std::cout << A[i] << " ";
////		}
////		std::cout << std::endl;*/
////		calc2(&hessianIt.Get());
////		break;
////
////
////		//save eigenValues		
////		//{
////		//	valueWriter << hessianIt.GetIndex()[0] << "," << hessianIt.GetIndex()[1] << "," << hessianIt.GetIndex()[2] << ": ";
////		//	for (int j = 0; j < 3; j++)
////		//	{
////		//		valueWriter << eigenValues[j] << " ";
////		//	}
////		//	valueWriter << std::endl;
////		//}
////
////		////save eigenvectors		
////		//{
////		//	vectorWriter << hessianIt.GetIndex()[0] << "," << hessianIt.GetIndex()[1] << "," << hessianIt.GetIndex()[2] << ": ";
////		//	for (int i = 0; i < ImageType::ImageDimension; i++)
////		//	{
////		//		for (int j = 0; j < ImageType::ImageDimension; j++)
////		//		{
////		//			vectorWriter << eigenvectors[i][j] << " ";
////		//		}
////		//		vectorWriter << "; ";
////		//	}
////		//	vectorWriter << std::endl;
////		//}
////	}
////
////	time.Stop();
////	std::cout << std::setprecision(3) << " eigen value test takes: " << time.GetMean() << " seconds" << std::endl;
////
////	valueWriter.close();
////	vectorWriter.close();
////
////	return;
////}
//void CancelLesion3() {
//	std::cout << "0" << std::endl;
//
//	itk::TimeProbe time;
//	time.Start();
//	std::cout << "0" << std::endl;
//
//	typedef itk::Image<float, 1905>  ImageType;
//	typedef itk::SymmetricSecondRankTensor< ImageType::PixelType, ImageType::ImageDimension > HessianPixelType;
//	typedef itk::Image< HessianPixelType, ImageType::ImageDimension >  HessianImageType;
//	typedef itk::FixedArray< ImageType::PixelType, ImageType::ImageDimension > EigenValueArrayType;
//	typedef itk::Matrix< ImageType::PixelType, ImageType::ImageDimension, ImageType::ImageDimension > EigenVectorMatrixType;
//	typedef itk::SymmetricEigenAnalysis< HessianPixelType, EigenValueArrayType, EigenVectorMatrixType >   CalculatorType;
//	std::cout << "0" << std::endl;
//
//	CalculatorType eigenCalculator(ImageType::ImageDimension);
//
//	std::cout << "1" << std::endl;
//
//	EigenValueArrayType *eigenValues=new EigenValueArrayType;
//	EigenVectorMatrixType *eigenvectors=new EigenVectorMatrixType;
//	eigenCalculator.SetOrderEigenMagnitudes(true);
//	HessianPixelType *hessianIt =new HessianPixelType;
//	std::cout << "2" << std::endl;
//	for (int i = 0; i < hessianIt->Size(); i++)
//	{
//		//std::cout <<"index:\t"<< i << std::endl;
//		//hessianIt.SetElement(i, i + 1);
//		(*hessianIt)[i] = i+1;
// 	}
//	std::cout << "3" << std::endl;
//	eigenCalculator.ComputeEigenValuesAndVectors(*hessianIt, *eigenValues, *eigenvectors);
//	std::cout << "4" << std::endl;
//	time.Stop();
//	std::cout << std::setprecision(3) << " eigen value test takes: " << time.GetMean() << " seconds" << std::endl;
//}
////void CancelLesion2()
////{
////	typedef itk::Image<float, 3>  ImageType;
////
////	std::string outputFolder = "F:/TestCuda/testByfurcationOutput/testEigenValues";
////	itksys::SystemTools::MakeDirectory(outputFolder);
////	std::string valueFile = outputFolder + "/value.txt";
////	std::string vectorFile = outputFolder + "/vector.txt";
////	std::ofstream valueWriter;
////	std::ofstream vectorWriter;
////	valueWriter.open(valueFile.c_str());
////	vectorWriter.open(vectorFile.c_str());
////
////	std::string filePath = "F:/TestCuda/preprocessedLiver.nii.gz";
////	//WriteNiftiImage<ImageType>(duplicator->GetOutput(), outputFolderTemplate, "templateImageJXY.nii.gz");
////	typedef itk::ImageFileReader<ImageType> ReaderType;
////	ReaderType::Pointer imageReader = ReaderType::New();
////	imageReader->SetFileName(filePath);
////	try
////	{
////		imageReader->Update();
////	}
////	catch (itk::ExceptionObject &exception)
////	{
////		//std::cerr << "Exception caught during reading file/truth: " << t2FileName << std::endl;
////		std::cerr << exception << std::endl;
////		return;
////	}
////
////	ImageType::Pointer inputImage = imageReader->GetOutput();
////	inputImage->DisconnectPipeline();
////
////	typedef itk::SymmetricSecondRankTensor< ImageType::PixelType, ImageType::ImageDimension > HessianPixelType;
////	typedef itk::Image< HessianPixelType, ImageType::ImageDimension >  HessianImageType;
////	typedef itk::FixedArray< ImageType::PixelType, ImageType::ImageDimension > EigenValueArrayType;
////	typedef itk::Matrix< ImageType::PixelType, ImageType::ImageDimension > EigenVectorMatrixType;
////	typedef itk::SymmetricEigenAnalysis< HessianPixelType, EigenValueArrayType, EigenVectorMatrixType >   CalculatorType;
////	CalculatorType eigenCalculator(ImageType::ImageDimension);
////
////	std::cout << "begin 000" << std::endl;
////	typedef itk::HessianRecursiveGaussianImageFilter<ImageType, HessianImageType> HessianFilterType;
////	typename HessianFilterType::Pointer h_filter = HessianFilterType::New();
////	h_filter->SetInput(inputImage);
////	h_filter->SetSigma(2.0);
////	h_filter->SetNormalizeAcrossScale(true);
////	h_filter->Update();
////
////	std::cout << "begin 001" << std::endl;
////	itk::TimeProbe time;
////	time.Start();
////
////	itk::ImageRegionConstIterator< ImageType > iIt(inputImage, inputImage->GetLargestPossibleRegion());
////	itk::ImageRegionConstIterator< HessianImageType > hessianIt(h_filter->GetOutput(), h_filter->GetOutput()->GetLargestPossibleRegion());
////	for (hessianIt.GoToBegin(), iIt.GoToBegin(); !hessianIt.IsAtEnd(); ++hessianIt, ++iIt)
////	{
////		if (iIt.Get() < 50) continue;
////
////		EigenValueArrayType eigenValues;
////		EigenVectorMatrixType eigenvectors;
////		eigenCalculator.SetOrderEigenMagnitudes(true);
////		eigenCalculator.ComputeEigenValuesAndVectors(hessianIt.Get(), eigenValues, eigenvectors);
////		break;
////		////save eigenValues		
////		//{
////		//	valueWriter << hessianIt.GetIndex()[0] << "," << hessianIt.GetIndex()[1] << "," << hessianIt.GetIndex()[2] << ": ";
////		//	for (int j = 0; j < 3; j++)
////		//	{
////		//		valueWriter << eigenValues[j] << " ";
////		//	}
////		//	valueWriter << std::endl;
////		//}
////
////		////save eigenvectors		
////		//{
////		//	vectorWriter << hessianIt.GetIndex()[0] << "," << hessianIt.GetIndex()[1] << "," << hessianIt.GetIndex()[2] << ": ";
////		//	for (int i = 0; i < ImageType::ImageDimension; i++)
////		//	{
////		//		for (int j = 0; j < ImageType::ImageDimension; j++)
////		//		{
////		//			vectorWriter << eigenvectors[i][j] << " ";
////		//		}
////		//		vectorWriter << "; ";
////		//	}
////		//	vectorWriter << std::endl;
////		//}
////	}
////
////	time.Stop();
////	std::cout << std::setprecision(3) << " eigen value test takes: " << time.GetMean() << " seconds" << std::endl;
////
////	valueWriter.close();
////	vectorWriter.close();
////
////	return;
////}
//int main(int argc, char*argv[]) {
//
//	//calc2(nullptr);
//
//	//CancelLesion2();
//
//	CancelLesion3();
//	//CancelLesion();
//}

#if 0
/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

inline int cudaDeviceInit(int argc, const char **argv)
{
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));

	if (deviceCount == 0)
	{
		std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
		exit(EXIT_FAILURE);
	}

	int dev = findCudaDevice(argc, argv);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

	checkCudaErrors(cudaSetDevice(dev));

	return dev;
}

bool printfNPPinfo(int argc, char *argv[])
{
	const NppLibraryVersion *libVer = nppGetLibVersion();

	printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

	int driverVersion, runtimeVersion;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
	printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

	// Min spec is SM 1.0 devices
	bool bVal = checkCudaCapabilities(1, 0);
	return bVal;
}

int main(int argc, char *argv[])
{
	printf("%s Starting...\n\n", argv[0]);

	try
	{
		std::string sFilename;
		char *filePath;

		cudaDeviceInit(argc, (const char **)argv);

		if (printfNPPinfo(argc, argv) == false)
		{
			exit(EXIT_SUCCESS);
		}

		if (checkCmdLineFlag(argc, (const char **)argv, "input"))
		{
			getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
		}
		else
		{
			filePath = sdkFindFilePath("Lena.pgm", argv[0]);
		}

		if (filePath)
		{
			sFilename = filePath;
		}
		else
		{
			sFilename = "Lena.pgm";
		}

		// if we specify the filename at the command line, then we only test sFilename[0].
		int file_errors = 0;
		std::ifstream infile(sFilename.data(), std::ifstream::in);

		if (infile.good())
		{
			std::cout << "cannyEdgeDetectionNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
			file_errors = 0;
			infile.close();
		}
		else
		{
			std::cout << "cannyEdgeDetectionNPP unable to open: <" << sFilename.data() << ">" << std::endl;
			file_errors++;
			infile.close();
		}

		if (file_errors > 0)
		{
			exit(EXIT_FAILURE);
		}

		std::string sResultFilename = sFilename;

		std::string::size_type dot = sResultFilename.rfind('.');

		if (dot != std::string::npos)
		{
			sResultFilename = sResultFilename.substr(0, dot);
		}

		sResultFilename += "_cannyEdgeDetection.pgm";

		if (checkCmdLineFlag(argc, (const char **)argv, "output"))
		{
			char *outputFilePath;
			getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFilePath);
			sResultFilename = outputFilePath;
		}

		// declare a host image object for an 8-bit grayscale image
		npp::ImageCPU_8u_C1 oHostSrc;
		// load gray-scale image from disk
		npp::loadImage(sFilename, oHostSrc);
		// declare a device image and copy construct from the host image,
		// i.e. upload host to device
		npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

		NppiSize oSrcSize = { (int)oDeviceSrc.width(), (int)oDeviceSrc.height() };
		NppiPoint oSrcOffset = { 0, 0 };

		// create struct with ROI size
		NppiSize oSizeROI = { (int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
		// allocate device image of appropriately reduced size
		npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

		int nBufferSize = 0;
		Npp8u * pScratchBufferNPP = 0;

		// get necessary scratch buffer size and allocate that much device memory
		NPP_CHECK_NPP(
			nppiFilterCannyBorderGetBufferSize(oSizeROI, &nBufferSize));

		cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

		// now run the canny edge detection filter
		// Using nppiNormL2 will produce larger magnitude values allowing for finer control of threshold values 
		// while nppiNormL1 will be slightly faster. Also, selecting the sobel gradient filter allows up to a 5x5 kernel size
		// which can produce more precise results but is a bit slower. Commonly nppiNormL2 and sobel gradient filter size of
		// 3x3 are used. Canny recommends that the high threshold value should be about 3 times the low threshold value.
		// The threshold range will depend on the range of magnitude values that the sobel gradient filter generates for a particular image.

		Npp16s nLowThreshold = 72;
		Npp16s nHighThreshold = 256;

		if ((nBufferSize > 0) && (pScratchBufferNPP != 0))
		{

			NPP_CHECK_NPP(
				nppiFilterCannyBorder_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
					oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI,
					NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, nLowThreshold, nHighThreshold,
					nppiNormL2, NPP_BORDER_REPLICATE, pScratchBufferNPP));
		}

		// free scratch buffer memory
		cudaFree(pScratchBufferNPP);

		// declare a host image for the result
		npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
		// and copy the device result data into it
		oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

		saveImage(sResultFilename, oHostDst);
		std::cout << "Saved image: " << sResultFilename << std::endl;

		nppiFree(oDeviceSrc.data());
		nppiFree(oDeviceDst.data());

		exit(EXIT_SUCCESS);
	}
	catch (npp::Exception &rException)
	{
		std::cerr << "Program error! The following exception occurred: \n";
		std::cerr << rException << std::endl;
		std::cerr << "Aborting." << std::endl;

		exit(EXIT_FAILURE);
	}
	catch (...)
	{
		std::cerr << "Program error! An unknow type of exception occurred. \n";
		std::cerr << "Aborting." << std::endl;

		exit(EXIT_FAILURE);
		return -1;
	}

	return 0;
}
#endif




#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
using namespace std;
struct V3x3 {
	float data[6] = { 0 };
	int mode = 0; //upper lower
};
//#define V3x3 float
void MatrixPrint(V3x3 *mat, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//cout << setw(2) << mat[i*cols + j].mode << " ";
		}
		cout << endl;
	}
	cout << endl;
}



// 3x3
void eigenvectors(V3x3 *a) {
//	a->mode += 1;
}

__global__ void addone(V3x3 *a,int size) {

    int tix = threadIdx.x;
    int tiy = threadIdx.y;
    
	int bdx = blockDim.x;
    int bdy = blockDim.y;

	int bix = blockIdx.x;
    int biy = blockIdx.y;

	int gdx = gridDim.x;
    int gdy = gridDim.y;
	for(int j=bix;j<size;j+=gdx){
		for(int i=tix;i<size;i+=bdx) {
			//a->mode += 1;
			a[j*size+i].mode += 1;
			//a[j*size+i] += 1;
		}
	}
}

/*

a b c
d e f
g h i
a[0] a[1] a[2]
a[3] a[4] a[5]
a[6] a[7] a[8]

三次项：1 
二次项：-(a+e+i)
一次项：ae+ei+ai-cg-bd-fh
常数项：ceg+bdi+afh-aei-bfg-cdh

*/
int main()
{
	int size = 5000;
	V3x3 *a = (V3x3*)malloc(sizeof(V3x3)*size*size);
	memset(a,0,sizeof(V3x3)*size*size);
	/*
	
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			a[i*size + j] = 1.0f;
		}
	}
	*/
	//MatrixPrint(a, size, size);
	V3x3 *a_cuda;
	cudaMalloc((void**)&a_cuda, sizeof(V3x3)*size*size);
	cudaMemcpy(a_cuda, a, sizeof(V3x3)*size*size, cudaMemcpyHostToDevice);

	dim3 grid(50, 1, 1), block(32, 33, 1);

//	itk::TimeProbe time;
//	time.Start();
	std::cout << " start" << std::endl;
	addone << <grid, block >> > (a_cuda,size);
//	time.Stop();
//	std::cout << " eigen value test takes: " << time.GetMean() << " seconds" << std::endl;
	cudaMemcpy(a, a_cuda, sizeof(V3x3)*size*size, cudaMemcpyDeviceToHost);
	//MatrixPrint(a, size, size);

	float  vec[] = {
		4, 2, -5,
		6, 4, -9,
		5, 3, -7
	};
	std::cout << "三次项: " << 1 << std::endl;
	std::cout << "二次项: " << -1 * (vec[0] + vec[4] + vec[8]) << std::endl;
	std::cout << "一次项: " << (vec[0] * vec[4] + vec[4] * vec[8] + vec[0] * vec[8] - vec[2] * vec[6] - vec[1] * vec[3] - vec[5] * vec[7]) <<std::endl;
	std::cout << "常数项: " << (vec[2] * vec[4] * vec[6] + vec[1] * vec[3] * vec[8] + vec[0] * vec[5] * vec[7] - vec[0] * vec[4] * vec[8] - vec[1] * vec[5] * vec[6] - vec[2] * vec[3] * vec[7]) << std::endl;
	return 0;
}