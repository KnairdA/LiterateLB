#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Image.hpp>

#include <imgui.h>
#include <imgui-SFML.h>

#include "texture.h"
#include "assets.h"

class RenderWindow {
private:
sf::RenderWindow _window;

sf::Sprite          _render_sprite;
sf::Texture         _render_texture;
cudaSurfaceObject_t _render_surface;
sf::Rect<int>       _render_texture_view;

sf::Shader _blur_shader;
bool _blur = false;

sf::Clock _ui_delta_clock;

public:
RenderWindow(std::string name):
  _window(sf::VideoMode(800, 600), name) {
  _render_texture.create(sf::VideoMode::getDesktopMode().width, sf::VideoMode::getDesktopMode().height);
  _render_surface = bindTextureToCuda(_render_texture);
  _render_sprite.setTexture(_render_texture);
  _render_texture_view = sf::Rect<int>(0,0,_window.getSize().x,_window.getSize().y);
  _render_sprite.setTextureRect(_render_texture_view);
  _window.setView(sf::View(sf::Vector2f(_render_texture_view.width/2, _render_texture_view.height/2),
                           sf::Vector2f(_window.getSize().x, _window.getSize().y)));
  _window.setVerticalSyncEnabled(true);
  _blur_shader.loadFromMemory(std::string(reinterpret_cast<const char*>(assets::shader::file_blur_frag)), sf::Shader::Fragment);
  _blur_shader.setUniform("texture", sf::Shader::CurrentTexture);
  ImGui::SFML::Init(_window);
  ImGuiIO& io = ImGui::GetIO();
  io.MouseDrawCursor = true;
};

bool isOpen() const {
  return _window.isOpen();
}

void setBlur(bool state) {
  _blur = state;
}

template <typename UI, typename MOUSE>
void draw(UI ui, MOUSE mouse);

cudaSurfaceObject_t getRenderSurface() {
  return _render_surface;
}

sf::Rect<int> getRenderView() {
  return _render_texture_view;
}

};

template <typename UI, typename MOUSE>
void RenderWindow::draw(UI ui, MOUSE mouse) {
  sf::Event event;
  while (_window.pollEvent(event)) {
    ImGui::SFML::ProcessEvent(event);
    if (event.type == sf::Event::Closed) {
      _window.close();
    }
    if (event.type == sf::Event::Resized) {
      _render_texture_view = sf::Rect<int>(0,0,event.size.width,event.size.height);
      _render_sprite.setTextureRect(_render_texture_view);
      sf::View view(sf::Vector2f(_render_texture_view.width/2, _render_texture_view.height/2),
                    sf::Vector2f(event.size.width, event.size.height));
      _window.setView(view);
    }
    if (!ImGui::GetIO().WantCaptureMouse) {
      mouse(event);
    }
  }

  ImGui::SFML::Update(_window, _ui_delta_clock.restart());
  ui();
  _window.clear();
  if (_blur) {
    _window.draw(_render_sprite, &_blur_shader);
  } else {
    _window.draw(_render_sprite);
  }
  ImGui::SFML::Render(_window);
  _window.display();
}
