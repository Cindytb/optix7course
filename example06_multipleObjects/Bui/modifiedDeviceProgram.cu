// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>

#include "modifiedLaunchParams.h"

extern "C" __constant__ const double pi = 3.14159265358979323846;
extern "C" __constant__ const double pi_2 = 1.57079632679489661923;

/*! launch parameters in constant memory, filled in by optix upon
            optixLaunch (this gets filled in from the buffer we pass to
            optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

static __forceinline__ __device__ int next_pow_2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
// for this simple example, we have a single ray type
enum
{
    SURFACE_RAY_TYPE = 0,
    RAY_TYPE_COUNT
};

static __forceinline__ __device__ void *unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void *ptr = reinterpret_cast<void *>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T *getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__raytrace()
{
    const ClosestHitRecord &sbtData = *(const ClosestHitRecord *)optixGetSbtDataPointer();

    float *&transmitted = *(float **)getPRD<float *>(1);
    float &distance = *(float *)getPRD<float>(2);
    const float ray_leg = optixGetRayTmax();
    distance = ray_leg + distance;
    if (sbtData.isMic)
    {
        int mic_no = sbtData.micID;
        int STRIDE = optixLaunchParams.time_bins * optixLaunchParams.freq_bands;
        int time_bin = distance / optixLaunchParams.hist_bin_size;
        for (int i = 0; i < optixLaunchParams.freq_bands; i++)
        {
            int idx = mic_no * STRIDE + time_bin * optixLaunchParams.freq_bands + i;
            optixLaunchParams.d_histogram[idx] = transmitted[i] / (distance * distance);
        }
    }
    else
    {
        for (int i = 0; i < optixLaunchParams.freq_bands; i++)
        {
            transmitted[i] /= sbtData.absorption;
        }
    }
    if (distance < optixLaunchParams.dist_thres)
    {
        
        const int primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const vec3f &A = sbtData.vertex[index.x];
        const vec3f &B = sbtData.vertex[index.y];
        const vec3f &C = sbtData.vertex[index.z];
        const vec3f Ng = normalize(cross(B - A, C - A));

        const vec3f rayDir = optixGetWorldRayDirection();
        //TODO: Use LA to compute the direction orthogonal to the place where it hit
        vec3f new_ray_dir = rayDir;

        optixTrace(optixLaunchParams.traversable,
                   camera.position,
                   new_ray_dir,
                   0.f,   // tmin
                   1e20f, // tmax
                   0.0f,  // rayTime
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
                   SURFACE_RAY_TYPE,              // SBT offset
                   RAY_TYPE_COUNT,                // SBT stride
                   SURFACE_RAY_TYPE,              // missSBTIndex
                   optixGetPayload_0(), optixGetPayload_1(), optixGetPayload_2(), optixGetPayload_3());
    }
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{ /* Unused */
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__beginRay()
{
    const int x = optixGetLaunchIndex.x;
    const int n_rays = optixGetLaunchDimensions().x;
    const float energy_0 = 2.f / n_rays;
    const float offset = 2.f / n_rays;

    float increment = pi * (3.f - sqrt(5.f)); // phi increment

    const float z = (x * offset - 1) + offset / 2.f;
    const float rho = sqrt(1.f - z * z);

    const float phi = ix * increment;

    const float x = cos(phi) * rho;
    const float y = sin(phi) * rho;
    const float azimuth = atan2(y, x);
    const float colatitude = atan2(sqrt(x * x + y * y), z);
    const int STRIDE = optixLaunchParams.time_bins * optixLaunchParams.freq_bands;
    const int STRIDE_POW_2 = next_pow_2(STRIDE);

    float *transmitted = optixLaunchParams.d_transmitted + n_rays * optixLaunchParams.freq_bands;
    for (int i = 0; i < optixLaunchParams.freq_bands; i++)
    {
        transmitted[i] = energy_0;
    }
    float distance = 0;
    unsigned int u0, u1, u2, u3;
    packPointer(transmitted, u0, u1);
    packPointer(&distance, u2, u3);
    float3 ray_direction = {
        sin(colatitude) * cos(azimuth),
        sin(colatitude) * sin(azimuth),
        cos(colatitude)};

    optixTrace(launchData.traversable,
               launchData.pos,
               ray_direction,
               0.f,  // tmin
               1e20, // tmax
               0.0f, // rayTime
               OptixVisibilitymask(255),
               OPTIX_RAY_FLAG_DISASBLE_ANYHIT
                   SURFACE_RAY_TYPE, //SBT offset
               RAY_TYPE_COUNT,       //SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               u0, u1, u2, u3);
}