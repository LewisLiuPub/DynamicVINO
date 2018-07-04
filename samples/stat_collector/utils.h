/*
// Copyright (c) 2017 Intel Corporation
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

#include <string>
#include <vector>

#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>

std::string GetArchPath();

InferenceEngine::InferenceEnginePluginPtr SelectPlugin(const std::vector<std::string> &pluginDirs,
                                            const std::string &plugin, InferenceEngine::TargetDevice device);

InferenceEngine::InferenceEnginePluginPtr SelectPlugin(const std::vector<std::string> &pluginDirs,
                                                       const std::string &plugin, const std::string& device);

std::string FileNameNoPath(const std::string& filePath);
std::string FileNameNoExt(const std::string& filePath);
std::string FilePath(const std::string& filePath);
std::string FileExt(const std::string& filePath);

char FilePathSeparator();

std::string VectorToStringF(std::vector<float> vec, const char sep = ',');
std::vector<float> StringToVectorF(std::string vec, const char sep = ',');

std::string VectorToStringI(std::vector<int> vec, const char sep = ',');
std::vector<int> StringToVectorI(std::string vec, const char sep = ',');

void CopyFile(const std::string& srcPath, const std::string& dstPath);

void ParseValFile(const std::string& path, std::vector<std::string>& images, std::vector<size_t>& classes);

void GetFilesInDir(const std::string& path, const std::string& ext, std::vector<std::string>& files);

bool IsDirectory(const std::string& path);

std::string ToUpper(const std::string& str);

float ScaleToDFP(float scale);
