#pragma once

#include "SoundSource.h"
int next_pow_2(int v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}
/*
	hist_bin_size : time in seconds - default 0.004
		Amount of time that each histogram bin will accumulate the erngy for
	time_thres : time in seconds - default 10
		Maximum amount of time for the impulse response
	energy_thres : energy/strength of a ray - default 1e-7
		Minimum amount of energy a ray must have before it's terminated
		NOTE: time_thres and energy_thres are NOT equal and are two separate checks for two different room scenarios.
		eg: A very dry room will have rays reaching the energy_thres but not the time_thres.
	dist_thres : distance in meters - default 3430
		dist_thres = c * time_thres
		time threshold converted into a distance. 
		With c = 343 and time_thres = 10, 
		it will default to 3430
	num_rays : number of rays to shoot out from each sound source
		TBD description for number. Currently testing
	freq_bands : discrete number of frequency bands/bins in the histogram
		freq_bands = (int) log2(fs / 125.0)
		This number is related to the frequency resolution of the histogram. 
		Denotes the size of the inner-most dimension of the 2D histogram
		Currently the resolution is set to octave bins starting at 125 Hz until Nyquist. 
		With all of the default settings at fs = 48k, 
		this defaults to 8
	time_bins : discrete number of time bins in the histogram
		time_bins = fs * time_thres * hist_bin_size
		Denotes the size of the outer dimension of the 2D histogram. 
		With all of the default settings at fs = 48k,
		this defaults to 2.5k
	
	Histogram shape:
			 ________
			|		 |
			|		 |					|
			|		 |					|
			|		 |		Increasing time/distance
			|		 |					|
			|		 |					|
			|		 |					V
			*repeat x2.5k*
			|________|
		--increase fq-->
	The pyroomacoustics implementation that this code is based off inverts the axes,
	so time is the inner-most dimension and frequency is the outer dimension.
	This is swapped in this CUDA/OptiX implementation because it is more cache-friendly.

*/
float SoundSource::hist_bin_size = 0.004;
float SoundSource::time_thres = 10;
float SoundSource::energy_thres = 1e-7;
float SoundSource::dist_thres = 3430;
float SoundSource::c = 343;
int SoundSource::num_rays = 10;
int SoundSource::freq_bands;
int SoundSource::time_bins;
OptixShaderBindingTable SoundSource::sbt = {};
OptixTraversableHandle SoundSource::traversable;
OptixPipeline SoundSource::pipeline;

SoundSource::SoundSource() {
	new (this) SoundSource({ 0, 0, 0 }, { 0, 0, 0 });
}
SoundSource::SoundSource(float3 pos) {
	new (this) SoundSource(pos, { 0, 0, 0 });
}

SoundSource::SoundSource(float3 pos, float3 orientation) {
	m_position = pos;
	m_orientation = orientation;
	local_histogram = new osc::LaunchParams();
	num_src++;
	int hist_bin_size_samples = fs * hist_bin_size;
	hist_bin_size = hist_bin_size_samples / (float)fs;
	dist_thres = c * time_thres;
	freq_bands = (int)log2(fs / 125.0);
	time_bins = fs * time_thres * hist_bin_size;
	checkCudaErrors(cudaStreamCreate(&m_stream));


	checkCudaErrors(cudaMalloc(&local_histogram->d_histogram, MAX_NUM_MICS * time_bins * freq_bands * sizeof(float)));
	checkCudaErrors(cudaMalloc(&local_histogram->d_transmitted, freq_bands * num_rays * sizeof(float)));
	m_histogram = new float[MAX_NUM_MICS * time_bins * freq_bands];
	DEBUG_CHECK();
	checkCudaErrors(cudaMalloc((void**)&d_local_histogram, sizeof(osc::LaunchParams)));
	fillWithZeroesKernel(local_histogram->d_histogram, MAX_NUM_MICS * time_bins * freq_bands, m_stream);
	fillWithZeroesKernel(local_histogram->d_transmitted, freq_bands * num_rays, m_stream);
	
}

void SoundSource::add_mic(Microphone mic) {
	m_microphones.push_back(mic);
}

void SoundSource::trace() {
	local_histogram->freq_bands = freq_bands;
	local_histogram->pos = m_position;
	local_histogram->orientation = m_orientation;
	local_histogram->traversable = traversable;
	local_histogram->time_bins = time_bins;
	local_histogram->hist_bin_size = hist_bin_size;
	local_histogram->dist_thres = dist_thres;
	local_histogram->energy_thres = energy_thres;
	local_histogram->c = c;
	printf("num_rays: %i\n", num_rays);
	checkCudaErrors(cudaMemcpy(d_local_histogram, local_histogram, sizeof(osc::LaunchParams), cudaMemcpyHostToDevice));

	OPTIX_CHECK(
		optixLaunch(/*! pipeline we're launching launch: */
			pipeline, m_stream,
			/*! parameters and SBT */
			(CUdeviceptr)d_local_histogram,
			sizeof(osc::LaunchParams),
			&sbt,
			/*! dimensions of the launch: */
			num_rays,
			1,
			1
		)
	);
	DEBUG_CHECK();
	checkCudaErrors(cudaMemcpyAsync(m_histogram, 
		local_histogram->d_histogram,
		MAX_NUM_MICS * time_bins * freq_bands * sizeof(float), 
		cudaMemcpyDeviceToHost, 
		m_stream));
	DEBUG_CHECK();

	
	
	std::ofstream myfile;
	myfile.open("histogram.dump");
		
	for (int i = 0; i < time_bins; i++) {
		myfile << m_histogram[i * freq_bands] << std::endl;
	}
	myfile.close();
}

void SoundSource::compute_IRs() {

}
void SoundSource::convolve() {

}
SoundSource::~SoundSource() {
	num_src--;
	checkCudaErrors(cudaFree(local_histogram->d_histogram));
	delete[] m_histogram;
	checkCudaErrors(cudaFree(local_histogram->d_transmitted));
	checkCudaErrors(cudaFree(d_local_histogram));
}