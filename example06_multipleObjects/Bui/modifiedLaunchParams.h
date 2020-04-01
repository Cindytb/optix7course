#pragma once

#include "gdt/math/vec.h"
#include <cuda_runtime.h>
//struct ClosestHitRecord
//{
//    vec3f *vertex;
//    vec3i *index;
//    bool isMic;
//    float absorption;
//    int micID;
//};
namespace bui {
    struct LaunchParams
    {
        float* d_histogram;
        float* d_transmitted;
        float3 pos, orientation;
        int freq_bands, time_bins, hist_res, num_mics, hist_bin_size;
        float dist_thres;
        OptixTraversableHandle traversable;
    };

}


