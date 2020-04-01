#pragma once

#include "SoundItem.h"

/*CUDA Includes*/

#include <cuda_runtime.h>
class Microphone:SoundItem {
public:
    Microphone();
    Microphone(float3 pos);
    Microphone(float3 pos, float3 orientation, int frames_per_buffer);
    ~Microphone();
private:

    float *m_output;
    int m_frames_per_buffer;
    
    void zero_output();
    
};