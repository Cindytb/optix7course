#pragma once

#include "SoundItem.h"
#include "Microphone.h"
#include "../LaunchParams.h"
#include "modifiedLaunchParams.h"
#include "../kernels.cuh"
#include "../debug.cuh"
#include "constants.h"
#include "optix.h"
//#include "optix7.h"


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/*CUDA Includes*/
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <glm/glm.hpp>


class SoundSource:SoundItem {
public:
    SoundSource();
    SoundSource(float3 pos);
    SoundSource(float3 pos, float3 orientation);
    ~SoundSource();
    static int freq_bands, time_bins, num_rays;
    static float hist_bin_size, time_thres, dist_thres, energy_thres, c;
    static OptixShaderBindingTable sbt;
    static OptixTraversableHandle traversable;
    static OptixPipeline pipeline;
    void add_mic(Microphone mic);
    void trace();
    void compute_IRs();
    void convolve();
    osc::LaunchParams *local_histogram;
private:
    osc::LaunchParams *d_local_histogram;
    float* m_histogram;
    std::vector<Microphone> m_microphones;
    cudaStream_t m_stream;
    
};