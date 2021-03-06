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
#include "debug.cuh"
#include "prd.h"
#include "LaunchParams.h"

using namespace osc;

namespace osc
{
	extern "C" __constant__ const double pi = 3.14159265358979323846;
	extern "C" __constant__ const double pi_2 = 1.57079632679489661923;
	/*! launch parameters in constant memory, filled in by optix upon
			optixLaunch (this gets filled in from the buffer we pass to
			optixLaunch) */
	extern "C" __constant__ LaunchParams optixLaunchParams;
	//extern "C" __constant__ bui::LaunchParams launchData;

	// for this simple example, we have a single ray type
	enum
	{
		SURFACE_RAY_TYPE = 0,
		RAY_TYPE_COUNT
	};
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
	static __forceinline__ __device__ void DEVICE_DEBUG(int line)
	{
		if (FULL_CUDA_DEBUG)
			printf("devicePrograms.cu: %d\n", line);
	}
	static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1)
	{
		const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
		void* ptr = reinterpret_cast<void*>(uptr);
		return ptr;
	}

	static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
	{
		const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
		i0 = uptr >> 32;
		i1 = uptr & 0x00000000ffffffff;
	}

	template <typename T>
	static __forceinline__ __device__ T* getPRD(int no)
	{
		if (no == 0)
		{
			const uint32_t u0 = optixGetPayload_0();
			const uint32_t u1 = optixGetPayload_1();
			return reinterpret_cast<T*>(unpackPointer(u0, u1));
		}
		else if (no == 1)
		{
			const uint32_t u2 = optixGetPayload_2();
			const uint32_t u3 = optixGetPayload_3();
			return reinterpret_cast<T*>(unpackPointer(u2, u3));
		}
		const uint32_t u4 = optixGetPayload_4();
		const uint32_t u5 = optixGetPayload_5();
		return reinterpret_cast<T*>(unpackPointer(u4, u5));
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

	extern "C" __global__ void __closesthit__radiance()
	{

		/*GRAPHICS ONLY*/
		const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

		const int primID = optixGetPrimitiveIndex();
		const vec3i index = sbtData.index[primID];
		const vec3f& A = sbtData.vertex[index.x];
		const vec3f& B = sbtData.vertex[index.y];
		const vec3f& C = sbtData.vertex[index.z];
		vec3f Ng = normalize(cross(B - A, C - A));
		vec3f rayDir = optixGetWorldRayDirection();
		if (optixLaunchParams.frame.size.x) {
			const float cosDN = 0.2f + .8f * fabsf(dot(rayDir, Ng));
			vec3f& prd = *(vec3f*)getPRD<vec3f>(0);
			prd = cosDN * sbtData.color;
		}
		else {
			PRD& ray_data = *(PRD*)getPRD<PRD>(0);
			float u = optixGetTriangleBarycentrics().x;
			float v = optixGetTriangleBarycentrics().y;
			float ray_leg = optixGetRayTmax();
			ray_data.distance += ray_leg;
			

			/*
				Resource on understanding Barycentric coordinates in computer graphics:
				https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
				Given a triangle with vertices A, B, and C:
				P = u * A + v * B + w * C
				such that u + v + w = 1
				and point P is a point somewhere inside the triangle
			*/
			vec3f P = (1.f - u - v) * A + u * B + v * C;
			
			
			//printf("P: %f %f %f\t ray_leg: %f\n", P.x, P.y, P.z, ray_leg); 
			if (sbtData.isMic)
			{	
				if (sbtData.pos != ray_data.previous_intersection) {
					vec3f radius = sbtData.pos - P;
					float norm = abs(sqrt(dot(radius, radius)));
					ray_data.distance += norm;
					//printf("distance: %f \n", ray_data.distance);
					int time_bin = ray_data.distance / (optixLaunchParams.hist_bin_size * optixLaunchParams.c);
					int ray_no = optixGetLaunchIndex().x;
					printf("Ray no: %i\tTransmitted[0]: %.10f\n", ray_no, ray_data.transmitted[0]);
					printf("Ray no %i\tEnergy[0]: %.10f\n", ray_no, ray_data.transmitted[0] / (ray_data.distance * ray_data.distance));
					int STRIDE = optixLaunchParams.time_bins * optixLaunchParams.freq_bands;
					for (int i = 0; i < optixLaunchParams.freq_bands; i++)
					{
						int idx = sbtData.micID * STRIDE + time_bin * optixLaunchParams.freq_bands + i;
						atomicAdd(optixLaunchParams.d_histogram + idx, ray_data.transmitted[i] / (ray_data.distance * ray_data.distance));
						
					}
					P = sbtData.pos;
					
				}
				ray_data.previous_intersection = sbtData.pos;
			}
			else {
				for (int i = 0; i < optixLaunchParams.freq_bands; i++)
				{
					ray_data.transmitted[i] *= 1 - sbtData.absorption;
				}
				//printf("transmitted[0]: %f\n", ray_data.transmitted[0]);
				/*
					Resource on understanding pure specular reflections:
					https://mathworld.wolfram.com/Reflection.html
				*/
				vec3f specularDir = rayDir - 2.0f * (rayDir * Ng) * Ng;
				ray_data.direction = specularDir;
				ray_data.previous_intersection = P;
				
			}
			ray_data.position = P + (1e-7f * ray_data.direction);
			ray_data.recursion_depth++;
		}
	}


	extern "C" __global__ void __anyhit__radiance()
	{ /* Empty*/
	}
	//------------------------------------------------------------------------------
	// miss program that gets called for any ray that did not have a
	// valid intersection
	//
	// as with the anyhit/closest hit programs, in this example we only
	// need to have _some_ dummy function to set up a valid SBT
	// ------------------------------------------------------------------------------

	extern "C" __global__ void __miss__radiance()
	{
		//printf("miss\n");
		vec3f& prd = *(vec3f*)getPRD<vec3f>(0);
		// set to constant white as background color
		prd = vec3f(1.f);
	}

	//------------------------------------------------------------------------------
	// ray gen program - the actual rendering happens in here
	//------------------------------------------------------------------------------
	extern "C" __global__ void __raygen__renderFrame()
	{
		// compute a test pattern based on pixel ID
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;
		unsigned int u0, u1; //payload values

		/* Graphics Raytracing Code */

		if (optixLaunchParams.frame.size.x != 0)
		{
			//printf("Beginning graphical raytrace\n");
			//DEVICE_DEBUG(__LINE__);
			vec3f pixelColorPRD = vec3f(0.f);
			packPointer(&pixelColorPRD, u0, u1);
			const auto& camera = optixLaunchParams.camera;
			// normalized screen plane position, in [0,1]^2
			const vec2f screen(vec2f(ix + .5f, iy + .5f) / vec2f(optixLaunchParams.frame.size));
			//DEVICE_DEBUG(__LINE__);

			// generate ray direction
			vec3f rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
			//DEVICE_DEBUG(__LINE__);

			optixTrace(optixLaunchParams.traversable,
				camera.position,
				rayDir,
				0.f,   // tmin
				1e20f, // tmax
				0.0f,  // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
				SURFACE_RAY_TYPE,			  // SBT offset
				RAY_TYPE_COUNT,				  // SBT stride
				SURFACE_RAY_TYPE,			  // missSBTIndex
				u0, u1);

			const int r = int(255.99f * pixelColorPRD.x);
			const int g = int(255.99f * pixelColorPRD.y);
			const int b = int(255.99f * pixelColorPRD.z);

			// convert to 32-bit rgba value (we explicitly set alpha to 0xff
			// to make stb_image_write happy ...
			const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

			// and write to frame buffer ...
			const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
			optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
			//DEVICE_DEBUG(__LINE__);
			//printf("ray %i %i successful\n", ix, iy);
		}
		else
		{
			const int n_rays = optixGetLaunchDimensions().x;

			//Using the ray number to compute the azimuth and elevation
			const float energy_0 = 2.f / n_rays;
			const float offset = 2.f / n_rays;

			float increment = pi * (3.f - sqrt(5.f)); // phi increment

			const float z = (ix * offset - 1) + offset / 2.f;
			const float rho = sqrt(1.f - z * z);

			const float phi = ix * increment;

			const float x = cos(phi) * rho;
			const float y = sin(phi) * rho;

			/*TODO: Do I need to compute the azimuth and elevation if {x, y, z} is the vector?*/
			const float azimuth = atan2(y, x);
			const float elevation = atan2(sqrt(x * x + y * y), z);

			// PRD: accumulated transmitted energy and distance

			//vec3f rayDir = { sin(elevation) * cos(azimuth), sin(elevation) * sin(azimuth), cos(elevation) };
			vec3f rayDir = { cos(ix * 2.0f * pi / n_rays), sin(ix * 2.0f * pi / n_rays), 0 };
			PRD ray_data;
			ray_data.transmitted = optixLaunchParams.d_transmitted + ix * optixLaunchParams.freq_bands;
			for (int i = 0; i < optixLaunchParams.freq_bands; i++)
			{
				ray_data.transmitted[i] = energy_0;
			}
			ray_data.distance = 0;
			packPointer(&ray_data, u0, u1);
			ray_data.recursion_depth = 0;
			ray_data.position = { optixLaunchParams.pos.x, optixLaunchParams.pos.y, optixLaunchParams.pos.z };
			ray_data.direction = rayDir;
			while (ray_data.distance < optixLaunchParams.dist_thres && ray_data.transmitted[0] > optixLaunchParams.energy_thres) {
				optixTrace(optixLaunchParams.traversable,
					ray_data.position,
					ray_data.direction,
					0.1f,   // tmin
					1e20f, // tmax
					0.0f,  // rayTime
					OptixVisibilityMask(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
					SURFACE_RAY_TYPE,			  // SBT offset
					RAY_TYPE_COUNT,				  // SBT stride
					SURFACE_RAY_TYPE,			  // missSBTIndex
					u0, u1);
			}

			//printf("recursionLevel: %i\n", ray_data.recursion_depth);
		}
	}

} // namespace osc
