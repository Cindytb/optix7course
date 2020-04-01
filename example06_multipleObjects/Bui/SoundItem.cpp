#pragma once

#include "SoundItem.h"

int SoundItem::num_mics = 0;
int SoundItem::num_src = 0;
int SoundItem::fs = 48000;


SoundItem::SoundItem(float3 pos, float3 orientation)//: m_position(pos), m_orientation(orientation)
{
    m_position = pos;
    m_orientation = orientation;
}
SoundItem::SoundItem() : m_position()
{
    float3 origin = {0,0,0};
    float3 default_orientation = {0,0,0};
    SoundItem(origin, default_orientation);

}