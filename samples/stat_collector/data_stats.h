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

#pragma once

#include <vector>

class DataStats {
  public:
    template<typename T>
    static void GetDataMinMax(const T* data, size_t count, T& min, T& max);

    template<typename T>
    static void GetDataAverage(const T* data, size_t count, T& ave);

    template<typename T>
    static void GetDataAbsMax(const T* data, size_t count, T& max);

    template<typename T>
    static T GetAbsMax(T min, T max);

    template<typename T>
    static void GetHistogram(const T* data, T max, size_t count, std::vector<size_t>& hist, bool onlyPositive, bool noZero);

    static void CalculateKLDivergence(float* dataFP32, float* dataINT8, const size_t& count, double& divergence);

    static void Normalize(float* data, const size_t& count);
};
