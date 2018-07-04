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

#include <algorithm>
#include <fstream>
#ifdef _WIN32
# include "w_dirent.h"
#else
# include <sys/stat.h>
# include <sys/types.h>
# include <dirent.h>
#endif

#include <map>
#include <cmath>

#include "utils.h"

static InferenceEngine::TargetDevice GetDeviceFromStr(const std::string& deviceName) {
    static std::map<std::string, InferenceEngine::TargetDevice> deviceFromNameMap = {
        { "CPU", InferenceEngine::TargetDevice::eCPU },
        { "GPU", InferenceEngine::TargetDevice::eGPU },
        { "FPGA", InferenceEngine::TargetDevice::eFPGA }
    };

    auto val = deviceFromNameMap.find(deviceName);
    return val != deviceFromNameMap.end() ? val->second : InferenceEngine::TargetDevice::eDefault;
}

InferenceEngine::InferenceEnginePluginPtr SelectPlugin(const std::vector<std::string> &pluginDirs,
                                                       const std::string &plugin,
                                                       InferenceEngine::TargetDevice device) {
    InferenceEngine::PluginDispatcher dispatcher(pluginDirs);

    if (!plugin.empty()) {
        return dispatcher.getPluginByName(plugin);
    } else {
        return dispatcher.getSuitablePlugin(device);
    }
}

InferenceEngine::InferenceEnginePluginPtr SelectPlugin(const std::vector<std::string> &pluginDirs,
                                                       const std::string &plugin,
                                                       const std::string& device) {
    return SelectPlugin(pluginDirs, plugin, GetDeviceFromStr(device));
}

char FilePathSeparator() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

std::string FileNameNoExt(const std::string& filePath) {
    auto pos = filePath.rfind('.');

    if (pos == std::string::npos) {
        return filePath;
    }

    return filePath.substr(0, pos);
}

std::string FileNameNoPath(const std::string& filePath) {
    auto pos = filePath.rfind(FilePathSeparator());

    if (pos == std::string::npos) {
        return filePath;
    }

    return filePath.substr(pos + 1);
}

std::string FilePath(const std::string& filePath) {
    auto pos = filePath.rfind(FilePathSeparator());

    if (pos == std::string::npos) {
        return filePath;
    }

    return filePath.substr(0, pos + 1);
}

std::string FileExt(const std::string& filePath) {
    int dot = filePath.find_last_of('.');

    if (dot == -1) {
        return "";
    }
    return filePath.substr(dot);
}

std::string VectorToStringF(std::vector<float> vec, const char sep) {
    std::string result;

    for (auto it : vec) {
        result += std::to_string(it) + sep;
    }

    if (result.empty()) {
        return result;
    }
    result = result.substr(0, result.size() - 2);

    return result;
}

std::vector<float> StringToVectorF(std::string vec, const char sep) {
    std::vector<float> result;

    std::istringstream stream(vec);
    std::string strVal;

    while (getline(stream, strVal, sep)) {
        result.push_back(std::stof(strVal));
    }

    return result;
}

std::string VectorToStringI(std::vector<int> vec, const char sep) {
    std::string result;

    for (auto it : vec) {
        result += std::to_string(it) + sep;
    }

    if (result.empty()) {
        return result;
    }

    result = result.substr(0, result.size() - 2);

    return result;
}

std::vector<int> StringToVectorI(std::string vec, const char sep) {
    std::vector<int> result;

    std::istringstream stream(vec);
    std::string strVal;

    while (getline(stream, strVal, sep)) {
        result.push_back(std::stoi(strVal));
    }

    return result;
}

void CopyFile(const std::string& srcPath, const std::string& dstPath) {
    std::ifstream  srcStream(srcPath, std::ios::binary);
    std::ofstream  dstStream(dstPath,   std::ios::binary);

    dstStream << srcStream.rdbuf();
}

void ParseValFile(const std::string& path, std::vector<std::string>& images, std::vector<size_t>& classes) {
    std::string line;
    std::ifstream valFile(path);

    while (std::getline(valFile, line)) {
        std::istringstream iss(line);

        std::string file;
        size_t cls;

        if (!(iss >> file >> cls)) {
            break;
        }

        images.push_back(file);
        classes.push_back(cls);
    }
}

void GetFilesInDir(const std::string& path, const std::string& ext, std::vector<std::string>& files) {
    if (path.empty()) {
        return;
    }

    std::string fullPath;
    std::string fileName;
    std::string parentPath = path;
    std::string upExt = ToUpper(ext);
    std::string entName;

    DIR *dir;

    struct dirent *ent;

    if ((dir = opendir(path.c_str())) == NULL) {
        return;
    }

    char sep = path[path.size() - 1];

    if (sep != FilePathSeparator()) {
        parentPath += FilePathSeparator();
    }

    while ((ent = readdir(dir)) != NULL) {
        fileName = ToUpper(ent->d_name);
        if (fileName == "." || fileName == "..")  {
            continue;
        }

        if (fileName.find(upExt) == -1) {
            continue;
        }

        fullPath = parentPath + ent->d_name;
        files.push_back(fullPath);
    }

    closedir(dir);
}

bool IsDirectory(const std::string& path) {
    struct stat statbuf;

    if (stat(path.c_str(), &statbuf) != 0) {
        return false;
    }

    return S_ISDIR(statbuf.st_mode);
}

std::string ToUpper(const std::string& str) {
    std::string up;

    std::transform(str.begin(), str.end(), std::back_inserter(up), toupper);

    return up;
}

float ScaleToDFP(float scale) {
    float log = floor(log2(scale) + 0.5);
    return pow(2, log);
}
