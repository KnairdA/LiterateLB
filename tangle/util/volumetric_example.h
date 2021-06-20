#pragma once
#include <LLBM/volumetric.h>
#include "camera.h"
#include "texture.h"
#include "colormap.h"
#include "noise.h"
#include "render_window.h"
#include "../sampler/sampler.h"

class VolumetricExample : public RenderWindow {
private:
std::vector<std::unique_ptr<Sampler>> _sampler;
Sampler* _current = nullptr;

Camera _camera;
VolumetricRenderConfig _config;
ColorPalette _palette;
NoiseSource _noise;

int _steps_per_second = 100;
int _samples_per_second = 30;

public:
template <template<typename...> class SAMPLER, typename... ARGS>
void add(ARGS&&... args) {
  _sampler.emplace_back(new SAMPLER(std::forward<ARGS>(args)...));
  _current = _sampler.back().get();
}

template <typename TIMESTEP>
void run(TIMESTEP step) {
  sf::Clock last_sample;
  sf::Clock last_frame;
  std::size_t iStep = 0;
  volatile bool simulate = true;

  sf::Thread simulation([&]() {
    while (this->isOpen()) {
      if (last_sample.getElapsedTime().asSeconds() > 1.0 / _samples_per_second) {
        _current->sample();
        cudaStreamSynchronize(cudaStreamPerThread);
        last_sample.restart();
        if (simulate) {
          for (unsigned i=0; i < (1.0 / _samples_per_second) * _steps_per_second; ++i) {
            step(iStep++);
          }
        }
      }
    }
  });
  simulation.launch();

  while (this->isOpen()) {
    this->draw(
      [&](){
        ImGui::Begin("Simulation", 0, ImGuiWindowFlags_AlwaysAutoResize);
        if (ImGui::BeginCombo("Source", _current->getName().c_str())) {
          for (auto& option : _sampler) {
            if (ImGui::Selectable(option->getName().c_str(), _current == option.get())) {
              _current = option.get();
            }
          }
          ImGui::EndCombo();
        }
        _current->interact();
        ImGui::SliderInt("Timestep/s", &_steps_per_second, 1, 1500);
        ImGui::SliderInt("Samples/s", &_samples_per_second, 1, 60);
        if (simulate) {
          simulate = !ImGui::Button("Pause");
        } else {
          simulate =  ImGui::Button("Continue");
        }
        ImGui::End();
        ImGui::Begin("Render", 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SliderFloat("Brightness", &_config.brightness, 0.1f, 2.f);
        ImGui::SliderFloat("Delta", &_config.delta, 0.05f, 2.f);
        ImGui::SliderFloat("Transparency", &_config.transparency, 0.001f, 1.f);
        _palette.interact();
        if (ImGui::CollapsingHeader("Details")) {
          ImGui::Checkbox("Align slices to view", &_config.align_slices_to_view);
          ImGui::SameLine();
          ImGui::Checkbox("Jitter", &_config.apply_noise);
          ImGui::SameLine();
          ImGui::Checkbox("Blur", &_config.apply_blur);
          this->setBlur(_config.apply_blur);
          if (_config.apply_noise) {
            _noise.interact();
          }
        }
        ImGui::End();
      },
      [&](sf::Event& event) {
        _camera.handle(event);
        _config.camera_position = _camera.getPosition();
        _config.camera_forward = _camera.getForward();
        _config.camera_right = _camera.getRight();
        _config.camera_up = _camera.getUp();
        _config.canvas_size = make_uint2(this->getRenderView().width, this->getRenderView().height);
      }
    );
    if (last_frame.getElapsedTime().asSeconds() > 1.0 / _samples_per_second) {
      _current->render(_config);
      cudaStreamSynchronize(cudaStreamPerThread);
      last_frame.restart();
    }
  }

  simulation.wait();
}

VolumetricExample(descriptor::CuboidD<3> cuboid):
  RenderWindow("LiterateLB"),
  _camera(make_float3(cuboid.nX/2,cuboid.nY/2,cuboid.nZ/2), cuboid.nX),
  _config(cuboid),
  _palette(_config.palette),
  _noise(_config.noise)
{
  _config.canvas = this->getRenderSurface();
  this->setBlur(_config.apply_blur);
}

};
