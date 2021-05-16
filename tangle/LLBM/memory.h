#pragma once

#include <memory>
#include <vector>
#include <cstring>

template <typename T>
class DeviceBuffer {
protected:
  const std::size_t _size;
  T* _data;

public:
  DeviceBuffer(std::size_t size):
    _size(size) {
    cudaMalloc(&_data, _size*sizeof(T));
    cudaMemset(_data, 0, _size*sizeof(T));
  }
  DeviceBuffer(const T* data, std::size_t size):
    DeviceBuffer(size) {
    cudaMemcpy(_data, data, size*sizeof(T), cudaMemcpyHostToDevice);
  }
  DeviceBuffer(const std::vector<T>& data):
    DeviceBuffer(data.data(), data.size()) { }
 
  ~DeviceBuffer() {
    cudaFree(_data);
  }

  T* device() {
    return _data;
  }

  std::size_t size() const {
    return _size;
  }
};

template <typename T>
class SharedVector : public DeviceBuffer<T> {
private:
  std::unique_ptr<T[]> _host_data;
 
public:
  SharedVector(std::size_t size):
    DeviceBuffer<T>(size),
    _host_data(new T[size]{}) {
    syncDeviceFromHost();
  }

  T* host() {
    return _host_data.get();
  }

  T& operator[](unsigned i) {
    return host()[i];
  }

  void syncHostFromDevice() {
    cudaMemcpy(_host_data.get(), this->_data, this->_size*sizeof(T), cudaMemcpyDeviceToHost);
  }

  void syncDeviceFromHost() {
    cudaMemcpy(this->_data, _host_data.get(), this->_size*sizeof(T), cudaMemcpyHostToDevice);
  }

};

template <typename T>
class DeviceTexture {
protected:
  cudaExtent _extent;
  cudaArray_t _array;

  cudaChannelFormatDesc _channel_desc;
  cudaResourceDesc _res_desc;
  cudaTextureDesc  _tex_desc;

  cudaTextureObject_t _texture;
  cudaSurfaceObject_t _surface;

public:
  DeviceTexture(std::size_t nX, std::size_t nY, std::size_t nZ=0):
    _extent(make_cudaExtent(nX,nY,nZ)),
    _channel_desc(cudaCreateChannelDesc<float>()) {
    cudaMalloc3DArray(&_array, &_channel_desc, _extent);

    std::memset(&_res_desc, 0, sizeof(_res_desc));
    _res_desc.resType = cudaResourceTypeArray;
    _res_desc.res.array.array = _array;

    std::memset(&_tex_desc, 0, sizeof(_tex_desc));
    _res_desc.resType = cudaResourceTypeArray;
    _tex_desc.addressMode[0]   = cudaAddressModeClamp;
    _tex_desc.addressMode[1]   = cudaAddressModeClamp;
    _tex_desc.addressMode[2]   = cudaAddressModeClamp;
    _tex_desc.filterMode       = cudaFilterModeLinear;
    _tex_desc.normalizedCoords = 0;

    cudaCreateTextureObject(&_texture, &_res_desc, &_tex_desc, NULL);
    cudaCreateSurfaceObject(&_surface, &_res_desc);
  }

  DeviceTexture(descriptor::CuboidD<3> c):
    DeviceTexture(c.nX, c.nY, c.nZ) { }
  
  ~DeviceTexture() {
    cudaFreeArray(_array);
  }

  cudaTextureObject_t getTexture() const {
    return _texture;
  }

  cudaSurfaceObject_t getSurface() const {
    return _surface;
  }

};

__device__ float3 colorFromTexture(cudaSurfaceObject_t colormap, float value) {
  uchar4 color{};
  value = clamp(value, 0.f, 1.f);
  surf2Dread(&color, colormap, unsigned(value * 999)*sizeof(uchar4), 0);
  return make_float3(color.x / 255.f,
                     color.y / 255.f,
                     color.z / 255.f);
}

__device__ float noiseFromTexture(cudaSurfaceObject_t noisemap, int x, int y) {
  uchar4 color{};
  surf2Dread(&color, noisemap, x*sizeof(uchar4), y);
  return color.x / 255.f;
}
