
#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "cuda.h"
#include "cuda_runtime.h"

#include "voxelGridTree.hpp"
#include "PinnedCPUResource.hpp"
#include "GPUResource.hpp"

namespace kernel {

void castKernel(unsigned long dataSize, float *float_data, unsigned char *char_data);

extern void computeMeanKernel(unsigned char *grid, 
		unsigned int *hit_counter, 
		unsigned int *sum, 
		unsigned long long int nData, 
		cudaStream_t stream);

extern void HoleFillingKernel(unsigned char interpolationRadius,
			unsigned int gridIdx, unsigned int gridIdy, unsigned int gridIdz,
			unsigned int subgridWidth, unsigned int subgridHeight, unsigned int subgridLength,
			unsigned long long int nData,
			unsigned char *srcGrid, 
			unsigned char *dstGrid, 
			cudaStream_t stream);

void VNNKernel(
		int nImages, int imgWidth, int imgHeight, 
		float deltaGrid, float deltaX, float deltaY,
		float xMin, float yMin, float zMin,
		unsigned int gridIdx, unsigned int gridIdy, unsigned int gridIdz,
		unsigned int voxelGridWidth, unsigned int voxelGridHeight, unsigned int voxelGridLength,
		float **offsets_d,
		float **rotations_d,
		unsigned char *char_image_data, 
		unsigned char *voxel_data, 
		unsigned int *mean_grid,
		unsigned int *hit_counter, 
		cudaStream_t stream);

enum NormalType { NORMAL_PER_QUAD, NORMAL_PER_VERTEX};
enum ColorType { COLOR_PER_QUAD, COLOR_PER_VERTEX};

unsigned int computeQuads(float **h_quads, float **h_colors, 
		VoxelGridTree<unsigned char, PinnedCPUResource, GPUResource> *cpuGrid,
		float alpha, 
		unsigned char threshold, 
		NormalType nt=NORMAL_PER_QUAD, ColorType ct=COLOR_PER_QUAD);
}

#endif



