#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <optix.h>


class SoundItem {
public:

    static int num_mics, num_src, fs;
    SoundItem(float3 pos, float3 orientation);
    SoundItem();
protected:
    float3 m_position;
    float3 m_orientation;
    
};