/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <stdint.h>

#include "data_stats.h"

template<typename T>
void DataStats::GetDataMinMax(const T* data, size_t count, T& min, T& max) {
    for (size_t i = 0; i < count; i++) {
        T val = data[i];

        if (min > val) {
            min = val;
        }

        if (max < val) {
            max = val;
        }
    }
}

template<typename T>
void DataStats::GetDataAbsMax(const T* data, size_t count, T& max) {
    T min = FLT_MAX;

    GetDataMinMax(data, count, min, max);

    max = GetAbsMax(min, max);
}

template void DataStats::GetDataMinMax<float>(const float* data, size_t count, float& min, float& max);
template void DataStats::GetDataMinMax<uint8_t>(const uint8_t* data, size_t count, uint8_t& min, uint8_t& max);

template void DataStats::GetDataAbsMax<float>(const float* data, size_t count, float& max);

template<typename T>
void DataStats::GetDataAverage(const T* data, size_t count, T& ave) {
    ave = 0;

    for (size_t i = 0; i < count; i++) {
        ave += data[i];
    }

    ave /= count;
}

template void DataStats::GetDataAverage<float>(const float* data, size_t count, float& ave);

template<typename T>
T DataStats::GetAbsMax(T min, T max) {
    if (min < 0) {
        min *= -1;
    }

    if (max < 0) {
        max *= -1;
    }

    return (max > min) ? max : min;
}

template float DataStats::GetAbsMax<float>(float min, float max);

template<typename T>
void DataStats::GetHistogram(const T* data, T max, size_t count, std::vector<size_t>& hist, bool onlyPositive, bool noZero) {
    size_t bins = (hist.size() - 1);

    T val;

    for (size_t i = 0; i < count; i++) {
        val = data[i];

        if (onlyPositive && val < 0) {
            val = 0;
        }

        if (noZero && val == 0) {
            continue;
        }

        size_t idx = (val * bins) / max;

        if (idx >= hist.size()) {
            idx = 0;
        }
        hist[idx] += 1;
    }
}

template void DataStats::GetHistogram(const float* data, float max, size_t count, std::vector<size_t>& hist, bool onlyPositive, bool noZero);

void DataStats::CalculateKLDivergence(float* dataFP32, float* dataINT8, const size_t& count, double& divergence) {
    divergence = 0;

    Normalize(dataFP32, count);
    Normalize(dataINT8, count);

    float val;

    for (size_t i = 0; i < count; i++) {
        float valFP32 = dataFP32[i];
        float valINT8 = dataINT8[i];

        if (valFP32 == 0) {
            continue;
        }

        if (valINT8 == 0) {
            continue;
        }

        val = valFP32 * log(valFP32 / valINT8);

        if (val < 0) {
            val *= -1;
        }

        divergence += val;
    }
}

void DataStats::Normalize(float* data, const size_t& count) {
    float max;
    float min;

    GetDataMinMax(data, count, min, max);

    for (size_t i = 0; i < count; i++) {
        if (data[i] < 0) {
            data[i] *= -1;
        }
    }

    double sum = 0;

    for (size_t i = 0; i < count; i++) {
        sum += data[i];
    }

    for (size_t i = 0; i < count; i++) {
        if (data[i] != 0) {
            data[i] = data[i] / sum;
        }
    }
}
