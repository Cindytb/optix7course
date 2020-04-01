#pragma once

#include "Microphone.h"


Microphone::Microphone(){
    Microphone({0, 0, 0}, {0, 0, 0}, 256);
}
Microphone::Microphone(float3 pos){
    Microphone(pos, {0, 0, 0}, 256);
}
Microphone::Microphone(float3 pos, float3 orientation, 
                        int frames_per_buffer) :  m_frames_per_buffer(frames_per_buffer) {
    m_position = pos;
    m_orientation = orientation;
    m_output = new float[m_frames_per_buffer];
    num_mics++;

}
void Microphone::zero_output(){
    for(int i = 0; i < m_frames_per_buffer; i++){
        m_output[i] = 0.0f;
    }
}
Microphone::~Microphone(){
    num_mics--;
    delete[] m_output;
}