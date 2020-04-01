
#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

struct PRD {
	gdt::vec3f position;
	float distance;
	gdt::vec3f direction;
	int recursion_depth;
	gdt::vec3f previous_intersection;
	float* transmitted;
};

