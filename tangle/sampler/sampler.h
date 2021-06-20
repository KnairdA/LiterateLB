#pragma once
#include <LLBM/base.h>

class RenderWindow;
class VolumetricRenderConfig;

class Sampler {
protected:
const std::string _name;

DeviceTexture<float> _sample_buffer;
cudaTextureObject_t _sample_texture;
cudaSurfaceObject_t _sample_surface;

public:
Sampler(std::string name, descriptor::CuboidD<3> cuboid):
  _name(name),
  _sample_buffer(cuboid),
  _sample_texture(_sample_buffer.getTexture()),
  _sample_surface(_sample_buffer.getSurface())
 { }

const std::string& getName() const {
  return _name;
}

virtual void sample() = 0;
virtual void render(VolumetricRenderConfig& config) = 0;
virtual void interact() = 0;

};
