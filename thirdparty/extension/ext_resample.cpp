/*
// Copyright (c) 2017-2018 Intel Corporation
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

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <immintrin.h>
#include <cmath>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ResampleImpl: public ExtLayerBase {
public:
    explicit ResampleImpl(const CNNLayer* layer): ExtLayerBase(layer) {
        try {
            if (cnnLayer.insData.size() != 1 || cnnLayer.outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            type = cnnLayer.GetParamAsString("type");
            antialias = static_cast<bool>(cnnLayer.GetParamAsInt("antialias"));

            addConfig({DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const auto *src_data = inputs[0]->cbuffer().as<const float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();
#ifdef WIN32
#undef IN
#endif
        size_t IN = inputs[0]->getTensorDesc().getDims()[0];
        size_t IC = inputs[0]->getTensorDesc().getDims()[1];
        size_t IH = inputs[0]->getTensorDesc().getDims()[2];
        size_t IW = inputs[0]->getTensorDesc().getDims()[3];

        size_t OH = outputs[0]->getTensorDesc().getDims()[2];
        size_t OW = outputs[0]->getTensorDesc().getDims()[3];

        if (IW == OW && IH == OH && type == "caffe.ResampleParameter.LINEAR") {
            memcpy(dst_data, src_data, IN * IC * IH * IW * sizeof(float));
            return OK;
        }

        float fx = static_cast<float>(IW) / static_cast<float>(OW);
        float fy = static_cast<float>(IH) / static_cast<float>(OH);

        bool isDownsample = (fx > 1) || (fy > 1);

        if (type == "caffe.ResampleParameter.NEAREST") {
#if defined(HAVE_SSE) || defined(HAVE_AVX2)
            if (!isDownsample && fx == 0.25f && fy == 0.25f)
                Upsample4x_Nearest(src_data, IW, IH, fx, fy, dst_data, OW, OH, IC, IN);
            else
#endif
            if (!isDownsample && fx == 0.5f && fy == 0.5f)
                Upsample2x_Nearest(src_data, IW, IH, dst_data, OW, OH, IC, IN);
            else
                NearestNeighborKernel(src_data, IW, IH, fx, fy, dst_data, OW, OH, IC, IN);
        } else if (type == "caffe.ResampleParameter.LINEAR") {
            size_t kernel_width = 2;

#if defined(HAVE_SSE) || defined(HAVE_AVX2)
            if (!isDownsample && fx == 0.25f && fy == 0.25f)
                Upsample4x_TriangleInterpolation(src_data, IW, IH, fx, fy, dst_data, OW, OH, IC, IN);
            else
#endif
                InterpolationKernel(src_data, IW, IH, fx, fy, dst_data, OW, OH, IC, IN, kernel_width, isDownsample && antialias);
        }
        return OK;
    }

private:
    std::string type;
    bool antialias;

    static inline float triangleCoeff(float x) {
        return std::max(0.0f, 1 - std::abs(x));
    }

    static void InterpolationKernel(const float *in_ptr_,
                                    const size_t iw, const size_t ih,
                                    const float fx, const float fy,
                                    float *out_ptr_,
                                    const size_t ow, const size_t oh, const size_t channels, const size_t batch,
                                    size_t kernel_width, bool antialias) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                const float *in_ptr = in_ptr_ + iw * ih * channels * b + iw * ih * c;
                float *out_ptr = out_ptr_ + ow * oh * channels * b + ow * oh * c;

                for (size_t oy = 0; oy < oh; oy++) {
                    for (size_t ox = 0; ox < ow; ox++) {
                        float ix = ox * fx + fy / 2.0f - 0.5f;
                        float iy = oy * fy + fx / 2.0f - 0.5f;

                        int ix_r = static_cast<int>(round(ix));
                        int iy_r = static_cast<int>(round(iy));

                        float sum = 0;
                        float wsum = 0;

                        float ax = 1.0f / (antialias ? fx : 1.0f);
                        float ay = 1.0f / (antialias ? fy : 1.0f);

                        int rx = (fx < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width) / ax);
                        int ry = (fy < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width) / ay);

                        for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                            for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                                if (y < 0 || x < 0 || y >= static_cast<int>(ih) || x >= static_cast<int>(iw))
                                    continue;

                                float dx = ix - x;
                                float dy = iy - y;

                                float w = ax * triangleCoeff(ax * dx) * ay * triangleCoeff(ay * dy);

                                sum += w * in_ptr[y * iw + x];
                                wsum += w;
                            }
                        }

                        out_ptr[oy * ow + ox] = (!wsum) ? 0 : (sum / wsum);
                    }
                }
            }
        }
    }

    static void NearestNeighborKernel(const float *in_ptr_,
                                      const size_t iw, const size_t ih,
                                      const float fx, const float fy,
                                      float *out_ptr_,
                                      const size_t ow, const size_t oh, const size_t channels, const size_t batch) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                const float *in_ptr = in_ptr_ + iw * ih * channels * b + iw * ih * c;
                float *out_ptr = out_ptr_ + ow * oh * channels * b + ow * oh * c;

                for (size_t oy = 0; oy < oh; oy++) {
                    for (size_t ox = 0; ox < ow; ox++) {
                        float ix = ox * fx + fy / 2.0f - 0.5f;
                        float iy = oy * fy + fx / 2.0f - 0.5f;

                        size_t ix_r = static_cast<size_t>(round(ix));
                        size_t iy_r = static_cast<size_t>(round(iy));

                        out_ptr[oy * ow + ox] = in_ptr[iy_r * iw + ix_r];
                    }
                }
            }
        }
    }

    void Upsample2x_Nearest(const float *in_ptr_,
                            const size_t iw, const size_t ih,
                            float *out_ptr_,
                            const size_t ow, const size_t oh, const size_t channels, const size_t batch) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                const float *in_ptr = in_ptr_ + iw * ih * channels * b + iw * ih * c;
                float *out_ptr = out_ptr_ + ow * oh * channels * b + ow * oh * c;

                for (size_t iy = 0; iy < ih; iy++) {
                    for (size_t ix = 0; ix < iw; ix++) {
                        size_t oy = 2 * iy;
                        size_t ox = 2 * ix;
                        float value = in_ptr[iy * iw + ix];
                        out_ptr[(oy + 0) * ow + ox + 0] = value;
                        out_ptr[(oy + 0) * ow + ox + 1] = value;
                        out_ptr[(oy + 1) * ow + ox + 0] = value;
                        out_ptr[(oy + 1) * ow + ox + 1] = value;
                    }
                }
            }
        }
    }

#if defined(HAVE_SSE) || defined(HAVE_AVX2)
    static void Upsample4x_Nearest(const float *in_ptr_,
                               const size_t iw, const size_t ih,
                               const float fx, const float fy,
                               float *out_ptr_,
                               const size_t ow, const size_t oh, const size_t channels, const size_t batch) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                const float* in_ptr = in_ptr_ + iw*ih*channels*b + iw*ih*c;
                float* out_ptr = out_ptr_ + ow*oh*channels*b + ow*oh*c;

                for (size_t oy = 0; oy < oh; oy++) {
                    for (size_t ox = 0; ox <= ow - 4; ox += 4) {
                        float ix = ox*fx + fy / 2.0f - 0.5f;
                        float iy = oy*fy + fx / 2.0f - 0.5f;

                        size_t ix_r = static_cast<size_t>(round(ix));
                        size_t iy_r = static_cast<size_t>(round(iy));

                        __m128 vsrc = _mm_load_ss(in_ptr+iy_r*iw+ix_r);
                        vsrc = _mm_shuffle_ps(vsrc, vsrc, 0x00);

                        _mm_store_ps(out_ptr+oy*ow+ox, vsrc);
                    }
                }
            }
        }
    }

    static void Upsample4x_TriangleInterpolation(const float *in_ptr_,
                                                 const size_t iw, const size_t ih,
                                                 const float fx, const float fy,
                                                 float *out_ptr_,
                                                 const size_t ow, const size_t oh, const size_t channels, const size_t batch) {
    #if defined(HAVE_AVX2)
        static float table_avx2[4][8*4] = {
                {
                        0.140625f, 0.046875f, 0.046875f, 0.140625f, 0.140625f, 0.046875f, 0.046875f, 0.140625f,
                        0.234375f, 0.328125f, 0.328125f, 0.234375f, 0.234375f, 0.328125f, 0.328125f, 0.234375f,
                        0.234375f, 0.078125f, 0.078125f, 0.234375f, 0.234375f, 0.078125f, 0.078125f, 0.234375f,
                        0.390625f, 0.546875f, 0.546875f, 0.390625f, 0.390625f, 0.546875f, 0.546875f, 0.390625f
                },
                {
                        0.046875f, 0.015625f, 0.015625f, 0.046875f, 0.046875f, 0.015625f, 0.015625f, 0.046875f,
                        0.078125f, 0.109375f, 0.109375f, 0.078125f, 0.078125f, 0.109375f, 0.109375f, 0.078125f,
                        0.328125f, 0.109375f, 0.109375f, 0.328125f, 0.328125f, 0.109375f, 0.109375f, 0.328125f,
                        0.546875f, 0.765625f, 0.765625f, 0.546875f, 0.546875f, 0.765625f, 0.765625f, 0.546875f
                },
                {
                        0.328125f, 0.109375f, 0.109375f, 0.328125f, 0.328125f, 0.109375f, 0.109375f, 0.328125f,
                        0.546875f, 0.765625f, 0.765625f, 0.546875f, 0.546875f, 0.765625f, 0.765625f, 0.546875f,
                        0.046875f, 0.015625f, 0.015625f, 0.046875f, 0.046875f, 0.015625f, 0.015625f, 0.046875f,
                        0.078125f, 0.109375f, 0.109375f, 0.078125f, 0.078125f, 0.109375f, 0.109375f, 0.078125f
                },
                {
                        0.234375f, 0.078125f, 0.078125f, 0.234375f, 0.234375f, 0.078125f, 0.078125f, 0.234375f,
                        0.390625f, 0.546875f, 0.546875f, 0.390625f, 0.390625f, 0.546875f, 0.546875f, 0.390625f,
                        0.140625f, 0.046875f, 0.046875f, 0.140625f, 0.140625f, 0.046875f, 0.046875f, 0.140625f,
                        0.234375f, 0.328125f, 0.328125f, 0.234375f, 0.234375f, 0.328125f, 0.328125f, 0.234375f
                }
        };
    #endif

    #if defined(HAVE_SSE) || defined(HAVE_AVX2)
        static float table_sse[4][4*4] = {
            {
                0.140625f, 0.046875f, 0.046875f, 0.140625f,
                0.234375f, 0.328125f, 0.328125f, 0.234375f,
                0.234375f, 0.078125f, 0.078125f, 0.234375f,
                0.390625f, 0.546875f, 0.546875f, 0.390625f
            },
            {
                0.046875f, 0.015625f, 0.015625f, 0.046875f,
                0.078125f, 0.109375f, 0.109375f, 0.078125f,
                0.328125f, 0.109375f, 0.109375f, 0.328125f,
                0.546875f, 0.765625f, 0.765625f, 0.546875f
            },
            {
                0.328125f, 0.109375f, 0.109375f, 0.328125f,
                0.546875f, 0.765625f, 0.765625f, 0.546875f,
                0.046875f, 0.015625f, 0.015625f, 0.046875f,
                0.078125f, 0.109375f, 0.109375f, 0.078125f
            },
            {
                0.234375f, 0.078125f, 0.078125f, 0.234375f,
                0.390625f, 0.546875f, 0.546875f, 0.390625f,
                0.140625f, 0.046875f, 0.046875f, 0.140625f,
                0.234375f, 0.328125f, 0.328125f, 0.234375f
            }
        };
    #endif
        for (size_t b = 0; b < batch; b++) {
            for (size_t c = 0; c < channels; c++) {
                const float *in_ptr = in_ptr_ + b * channels * iw * ih + c * iw * ih;
                float *out_ptr = out_ptr_ + b * channels * ow * oh + c * ow * oh;

                size_t oy = 0;
                {
                    float iy = oy * fy + fx / 2.0f - 0.5f;
                    size_t iy_r = static_cast<size_t>(round(iy));

                    size_t ox = 0;
        #if defined(HAVE_AVX2)
                    for (; ox <= ow - 8; ox += 8) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m256 vx00 = _mm256_setzero_ps();
                        __m256 vx01 = _mm256_setzero_ps();
                        __m256 vx02 = _mm256_setzero_ps();

                        __m128 vx10_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r - 1);
                        __m128 vx11_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 0);
                        __m128 vx12_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 1);
                        __m128 vx13_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 2);

                        __m128 vx20_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r - 1);
                        __m128 vx21_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 0);
                        __m128 vx22_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 1);
                        __m128 vx23_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 2);

                        __m256 vx10 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx10_), vx11_, 1);
                        __m256 vx11 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx11_), vx12_, 1);
                        __m256 vx12 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx12_), vx13_, 1);
                        __m256 vx20 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx20_), vx21_, 1);
                        __m256 vx21 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx21_), vx22_, 1);
                        __m256 vx22 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx22_), vx23_, 1);

                        for (size_t i = 0; i < 4; i++) {
                            __m256 vc0 = i < 2 ? _mm256_setzero_ps() : _mm256_loadu_ps(table_avx2[i] + 0);
                            __m256 vc1 = i < 2 ? _mm256_setzero_ps() : _mm256_loadu_ps(table_avx2[i] + 8);
                            __m256 vc2 = _mm256_loadu_ps(table_avx2[i] + 16);
                            __m256 vc3 = _mm256_loadu_ps(table_avx2[i] + 24);

                            if (ox == 0) {
                                if (i > 1)
                                    vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc0, 0), 0xD0), 0);
                                vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc2, 0), 0xD0), 0);
                            } else if (ox == ow - 8) {
                                if (i > 1)
                                    vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm256_extractf128_ps(vc0, 1), _mm_setzero_ps(), 0x07), 1);
                                vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm256_extractf128_ps(vc2, 1), _mm_setzero_ps(), 0x07), 1);
                            }

                            __m256 vsrc0 = i < 2 ? _mm256_shuffle_ps(vx00, vx02, 0x0) : _mm256_shuffle_ps(vx10, vx12, 0x0);
                            __m256 vsrc1 = i < 2 ? _mm256_shuffle_ps(vx01, vx01, 0x0) : _mm256_shuffle_ps(vx11, vx11, 0x0);
                            __m256 vsrc2 = i < 2 ? _mm256_shuffle_ps(vx10, vx12, 0x0) : _mm256_shuffle_ps(vx20, vx22, 0x0);
                            __m256 vsrc3 = i < 2 ? _mm256_shuffle_ps(vx11, vx11, 0x0) : _mm256_shuffle_ps(vx21, vx21, 0x0);

                            __m256 res = _mm256_setzero_ps();

                            res = _mm256_fmadd_ps(vsrc0, vc0, res);
                            res = _mm256_fmadd_ps(vsrc1, vc1, res);
                            res = _mm256_fmadd_ps(vsrc2, vc2, res);
                            res = _mm256_fmadd_ps(vsrc3, vc3, res);
                            __m256 wei = _mm256_add_ps(_mm256_add_ps(vc0, vc1), _mm256_add_ps(vc2, vc3));

                            res = _mm256_div_ps(res, wei);

                            _mm256_storeu_ps(out_ptr + (oy + i) * ow + ox, res);
                        }
                    }
        #endif

        #if defined(HAVE_SSE) || defined(HAVE_AVX2)
                    for (; ox <= ow - 4; ox += 4) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00 = _mm_setzero_ps();
                        __m128 vx01 = _mm_setzero_ps();
                        __m128 vx02 = _mm_setzero_ps();

                        __m128 vx10 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r-1);
                        __m128 vx11 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+0);
                        __m128 vx12 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+1);

                        __m128 vx20 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r-1);
                        __m128 vx21 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r+0);
                        __m128 vx22 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r+1);

                        for (size_t i = 0; i < 4; i++) {
                            __m128 vc0 = i < 2 ? _mm_setzero_ps() : _mm_loadu_ps(table_sse[i] + 0);
                            __m128 vc1 = i < 2 ? _mm_setzero_ps() : _mm_loadu_ps(table_sse[i] + 4);
                            __m128 vc2 = _mm_loadu_ps(table_sse[i] +  8);
                            __m128 vc3 = _mm_loadu_ps(table_sse[i] + 12);

                            if (ox == 0) {
                                if (i > 1)
                                    vc0 = _mm_shuffle_ps(_mm_setzero_ps(), vc0, 0xD0);
                                vc2 = _mm_shuffle_ps(_mm_setzero_ps(), vc2, 0xD0);
                            } else if (ox == ow - 4) {
                                if (i > 1)
                                    vc0 = _mm_shuffle_ps(vc0, _mm_setzero_ps() , 0x07);
                                vc2 = _mm_shuffle_ps(vc2, _mm_setzero_ps() , 0x07);
                            }

                            __m128 vsrc0 = i < 2 ? _mm_shuffle_ps(vx00, vx02, 0x0) : _mm_shuffle_ps(vx10, vx12, 0x0);
                            __m128 vsrc1 = i < 2 ? _mm_shuffle_ps(vx01, vx01, 0x0) : _mm_shuffle_ps(vx11, vx11, 0x0);
                            __m128 vsrc2 = i < 2 ? _mm_shuffle_ps(vx10, vx12, 0x0) : _mm_shuffle_ps(vx20, vx22, 0x0);
                            __m128 vsrc3 = i < 2 ? _mm_shuffle_ps(vx11, vx11, 0x0) : _mm_shuffle_ps(vx21, vx21, 0x0);

                            __m128 vres0 = _mm_mul_ps(vsrc0, vc0);
                            __m128 vres1 = _mm_mul_ps(vsrc1, vc1);
                            __m128 vres2 = _mm_mul_ps(vsrc2, vc2);
                            __m128 vres3 = _mm_mul_ps(vsrc3, vc3);

                            __m128 res = _mm_add_ps(_mm_add_ps(vres0, vres1), _mm_add_ps(vres2, vres3));
                            __m128 wei = _mm_add_ps(_mm_add_ps(vc0, vc1), _mm_add_ps(vc2, vc3));

                            res = _mm_div_ps(res, wei);

                            _mm_storeu_ps(out_ptr + (oy+i)*ow + ox, res);
                        }
                    }
        #endif
                }

                for (oy = 4; oy <= oh - 8; oy += 4) {
                    float iy = oy * fy + fx / 2.0f - 0.5f;
                    size_t iy_r = static_cast<size_t>(round(iy));

                    size_t ox = 0;
        #if defined(HAVE_AVX2)
                    for (; ox <= ow - 8; ox += 8) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r - 1);
                        __m128 vx01_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 0);
                        __m128 vx02_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 1);
                        __m128 vx03_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 2);

                        __m128 vx10_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r - 1);
                        __m128 vx11_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 0);
                        __m128 vx12_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 1);
                        __m128 vx13_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 2);

                        __m128 vx20_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r - 1);
                        __m128 vx21_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 0);
                        __m128 vx22_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 1);
                        __m128 vx23_ = _mm_load_ss(in_ptr + (iy_r + 1) * iw + ix_r + 2);

                        __m256 vx00 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx00_), vx01_, 1);
                        __m256 vx01 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx01_), vx02_, 1);
                        __m256 vx02 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx02_), vx03_, 1);

                        __m256 vx10 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx10_), vx11_, 1);
                        __m256 vx11 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx11_), vx12_, 1);
                        __m256 vx12 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx12_), vx13_, 1);

                        __m256 vx20 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx20_), vx21_, 1);
                        __m256 vx21 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx21_), vx22_, 1);
                        __m256 vx22 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx22_), vx23_, 1);

                        for (size_t i = 0; i < 4; i++) {
                            __m256 vc0 = _mm256_loadu_ps(table_avx2[i] + 0);
                            __m256 vc1 = _mm256_loadu_ps(table_avx2[i] + 8);
                            __m256 vc2 = _mm256_loadu_ps(table_avx2[i] + 16);
                            __m256 vc3 = _mm256_loadu_ps(table_avx2[i] + 24);

                            if (ox == 0) {
                                vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc0, 0), 0xD0), 0);
                                vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc2, 0), 0xD0), 0);
                            } else if (ox == ow - 8) {
                                vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm256_extractf128_ps(vc0, 1), _mm_setzero_ps(), 0x07), 1);
                                vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm256_extractf128_ps(vc2, 1), _mm_setzero_ps(), 0x07), 1);
                            }

                            __m256 vsrc0 = i < 2 ? _mm256_shuffle_ps(vx00, vx02, 0x0) : _mm256_shuffle_ps(vx10, vx12, 0x0);
                            __m256 vsrc1 = i < 2 ? _mm256_shuffle_ps(vx01, vx01, 0x0) : _mm256_shuffle_ps(vx11, vx11, 0x0);
                            __m256 vsrc2 = i < 2 ? _mm256_shuffle_ps(vx10, vx12, 0x0) : _mm256_shuffle_ps(vx20, vx22, 0x0);
                            __m256 vsrc3 = i < 2 ? _mm256_shuffle_ps(vx11, vx11, 0x0) : _mm256_shuffle_ps(vx21, vx21, 0x0);

                            __m256 res = _mm256_setzero_ps();

                            res = _mm256_fmadd_ps(vsrc0, vc0, res);
                            res = _mm256_fmadd_ps(vsrc1, vc1, res);
                            res = _mm256_fmadd_ps(vsrc2, vc2, res);
                            res = _mm256_fmadd_ps(vsrc3, vc3, res);

                            if (ox == 0 || ox == ow - 8) {
                                __m256 wei = _mm256_add_ps(_mm256_add_ps(vc0, vc1), _mm256_add_ps(vc2, vc3));

                                res = _mm256_div_ps(res, wei);
                            }

                            _mm256_storeu_ps(out_ptr + (oy + i) * ow + ox, res);
                        }
                    }
        #endif

        #if defined(HAVE_SSE) || defined(HAVE_AVX2)
                    for (; ox <= ow - 4; ox += 4) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r-1);
                        __m128 vx01 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r+0);
                        __m128 vx02 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r+1);

                        __m128 vx10 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r-1);
                        __m128 vx11 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+0);
                        __m128 vx12 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+1);

                        __m128 vx20 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r-1);
                        __m128 vx21 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r+0);
                        __m128 vx22 = _mm_load_ss(in_ptr+(iy_r+1)*iw+ix_r+1);

                        for (size_t i = 0; i < 4; i++) {
                            __m128 vc0 = _mm_loadu_ps(table_sse[i] +  0);
                            __m128 vc1 = _mm_loadu_ps(table_sse[i] +  4);
                            __m128 vc2 = _mm_loadu_ps(table_sse[i] +  8);
                            __m128 vc3 = _mm_loadu_ps(table_sse[i] + 12);

                            if (ox == 0) {
                                vc0 = _mm_shuffle_ps(_mm_setzero_ps(), vc0, 0xD0);
                                vc2 = _mm_shuffle_ps(_mm_setzero_ps(), vc2, 0xD0);
                            } else if (ox == ow - 4) {
                                vc0 = _mm_shuffle_ps(vc0, _mm_setzero_ps() , 0x07);
                                vc2 = _mm_shuffle_ps(vc2, _mm_setzero_ps() , 0x07);
                            }

                            __m128 vsrc0 = i < 2 ? _mm_shuffle_ps(vx00, vx02, 0x0) : _mm_shuffle_ps(vx10, vx12, 0x0);
                            __m128 vsrc1 = i < 2 ? _mm_shuffle_ps(vx01, vx01, 0x0) : _mm_shuffle_ps(vx11, vx11, 0x0);
                            __m128 vsrc2 = i < 2 ? _mm_shuffle_ps(vx10, vx12, 0x0) : _mm_shuffle_ps(vx20, vx22, 0x0);
                            __m128 vsrc3 = i < 2 ? _mm_shuffle_ps(vx11, vx11, 0x0) : _mm_shuffle_ps(vx21, vx21, 0x0);

                            __m128 vres0 = _mm_mul_ps(vsrc0, vc0);
                            __m128 vres1 = _mm_mul_ps(vsrc1, vc1);
                            __m128 vres2 = _mm_mul_ps(vsrc2, vc2);
                            __m128 vres3 = _mm_mul_ps(vsrc3, vc3);

                            __m128 res = _mm_add_ps(_mm_add_ps(vres0, vres1), _mm_add_ps(vres2, vres3));
                            if (ox == 0 || ox == ow - 4) {
                                __m128 wei = _mm_add_ps(_mm_add_ps(vc0, vc1), _mm_add_ps(vc2, vc3));

                                res = _mm_div_ps(res, wei);
                            }

                            _mm_storeu_ps(out_ptr + (oy+i)*ow + ox, res);
                        }
                    }
        #endif
                }

                oy = oh - 4;
                {
                    float iy = oy * fy + fx / 2.0f - 0.5f;
                    size_t iy_r = static_cast<size_t>(round(iy));

                    size_t ox = 0;

        #if defined(HAVE_AVX2)
                    for (; ox <= ow - 8; ox += 8) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r - 1);
                        __m128 vx01_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 0);
                        __m128 vx02_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 1);
                        __m128 vx03_ = _mm_load_ss(in_ptr + (iy_r - 1) * iw + ix_r + 2);

                        __m128 vx10_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r - 1);
                        __m128 vx11_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 0);
                        __m128 vx12_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 1);
                        __m128 vx13_ = _mm_load_ss(in_ptr + (iy_r + 0) * iw + ix_r + 2);

                        __m256 vx00 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx00_), vx01_, 1);
                        __m256 vx01 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx01_), vx02_, 1);
                        __m256 vx02 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx02_), vx03_, 1);

                        __m256 vx10 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx10_), vx11_, 1);
                        __m256 vx11 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx11_), vx12_, 1);
                        __m256 vx12 = _mm256_insertf128_ps(_mm256_castps128_ps256(vx12_), vx13_, 1);

                        __m256 vx20 = _mm256_setzero_ps();
                        __m256 vx21 = _mm256_setzero_ps();
                        __m256 vx22 = _mm256_setzero_ps();

                        for (size_t i = 0; i < 4; i++) {
                            __m256 vc0 = _mm256_loadu_ps(table_avx2[i] + 0);
                            __m256 vc1 = _mm256_loadu_ps(table_avx2[i] + 8);
                            __m256 vc2 = i < 2 ? _mm256_loadu_ps(table_avx2[i] + 16) : _mm256_setzero_ps();
                            __m256 vc3 = i < 2 ? _mm256_loadu_ps(table_avx2[i] + 24) : _mm256_setzero_ps();

                            if (ox == 0) {
                                vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc0, 0), 0xD0), 0);
                                if (i < 2)
                                    vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm_setzero_ps(), _mm256_extractf128_ps(vc2, 0), 0xD0), 0);
                            } else if (ox == ow - 8) {
                                vc0 = _mm256_insertf128_ps(vc0, _mm_shuffle_ps(_mm256_extractf128_ps(vc0, 1), _mm_setzero_ps(), 0x07), 1);
                                if (i < 2)
                                    vc2 = _mm256_insertf128_ps(vc2, _mm_shuffle_ps(_mm256_extractf128_ps(vc2, 1), _mm_setzero_ps(), 0x07), 1);
                            }

                            __m256 vsrc0 = i < 2 ? _mm256_shuffle_ps(vx00, vx02, 0x0) : _mm256_shuffle_ps(vx10, vx12, 0x0);
                            __m256 vsrc1 = i < 2 ? _mm256_shuffle_ps(vx01, vx01, 0x0) : _mm256_shuffle_ps(vx11, vx11, 0x0);
                            __m256 vsrc2 = i < 2 ? _mm256_shuffle_ps(vx10, vx12, 0x0) : _mm256_shuffle_ps(vx20, vx22, 0x0);
                            __m256 vsrc3 = i < 2 ? _mm256_shuffle_ps(vx11, vx11, 0x0) : _mm256_shuffle_ps(vx21, vx21, 0x0);

                            __m256 res = _mm256_setzero_ps();

                            res = _mm256_fmadd_ps(vsrc0, vc0, res);
                            res = _mm256_fmadd_ps(vsrc1, vc1, res);
                            res = _mm256_fmadd_ps(vsrc2, vc2, res);
                            res = _mm256_fmadd_ps(vsrc3, vc3, res);

                            __m256 wei = _mm256_add_ps(_mm256_add_ps(vc0, vc1), _mm256_add_ps(vc2, vc3));

                            res = _mm256_div_ps(res, wei);

                            _mm256_storeu_ps(out_ptr + (oy + i) * ow + ox, res);
                        }
                    }
        #endif

        #if defined(HAVE_SSE) || defined(HAVE_AVX2)
                    for (; ox <= ow - 4; ox += 4) {
                        float ix = (ox + 0) * fx + fy / 2.0f - 0.5f;
                        size_t ix_r = static_cast<size_t>(round(ix));

                        __m128 vx00 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r-1);
                        __m128 vx01 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r+0);
                        __m128 vx02 = _mm_load_ss(in_ptr+(iy_r-1)*iw+ix_r+1);

                        __m128 vx10 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r-1);
                        __m128 vx11 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+0);
                        __m128 vx12 = _mm_load_ss(in_ptr+(iy_r+0)*iw+ix_r+1);

                        __m128 vx20 = _mm_setzero_ps();
                        __m128 vx21 = _mm_setzero_ps();
                        __m128 vx22 = _mm_setzero_ps();

                        for (size_t i = 0; i < 4; i++) {
                            __m128 vc0 = _mm_loadu_ps(table_sse[i] +  0);
                            __m128 vc1 = _mm_loadu_ps(table_sse[i] +  4);
                            __m128 vc2 = i < 2 ?_mm_loadu_ps(table_sse[i] +  8) : _mm_setzero_ps();
                            __m128 vc3 = i < 2 ?_mm_loadu_ps(table_sse[i] + 12) : _mm_setzero_ps();

                            if (ox == 0) {
                                vc0 = _mm_shuffle_ps(_mm_setzero_ps(), vc0, 0xD0);
                                if (i < 2)
                                    vc2 = _mm_shuffle_ps(_mm_setzero_ps(), vc2, 0xD0);
                            } else if (ox == ow - 4) {
                                vc0 = _mm_shuffle_ps(vc0, _mm_setzero_ps() , 0x07);
                                if (i < 2)
                                    vc2 = _mm_shuffle_ps(vc2, _mm_setzero_ps() , 0x07);
                            }

                            __m128 vsrc0 = i < 2 ? _mm_shuffle_ps(vx00, vx02, 0x0) : _mm_shuffle_ps(vx10, vx12, 0x0);
                            __m128 vsrc1 = i < 2 ? _mm_shuffle_ps(vx01, vx01, 0x0) : _mm_shuffle_ps(vx11, vx11, 0x0);
                            __m128 vsrc2 = i < 2 ? _mm_shuffle_ps(vx10, vx12, 0x0) : _mm_shuffle_ps(vx20, vx22, 0x0);
                            __m128 vsrc3 = i < 2 ? _mm_shuffle_ps(vx11, vx11, 0x0) : _mm_shuffle_ps(vx21, vx21, 0x0);

                            __m128 vres0 = _mm_mul_ps(vsrc0, vc0);
                            __m128 vres1 = _mm_mul_ps(vsrc1, vc1);
                            __m128 vres2 = _mm_mul_ps(vsrc2, vc2);
                            __m128 vres3 = _mm_mul_ps(vsrc3, vc3);

                            __m128 res = _mm_add_ps(_mm_add_ps(vres0, vres1), _mm_add_ps(vres2, vres3));
                            __m128 wei = _mm_add_ps(_mm_add_ps(vc0, vc1), _mm_add_ps(vc2, vc3));

                            res = _mm_div_ps(res, wei);

                            _mm_storeu_ps(out_ptr + (oy+i)*ow + ox, res);
                        }
                    }
        #endif
                }
            }
        }
    }
#endif  // defined(HAVE_SSE) || defined(HAVE_AVX2)
};

REG_FACTORY_FOR(ImplFactory<ResampleImpl>, Resample);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
