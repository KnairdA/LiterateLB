#pragma once
#include "assets.h"
#include "texture.h"

#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/Graphics.hpp>

struct NoiseSource {
  const assets::File* current;
  sf::Texture texture;

  NoiseSource(cudaSurfaceObject_t& noise) {
    current = &assets::noise::files[0];
    texture.loadFromMemory(current->data, current->size);
    noise = bindTextureToCuda(texture);
  }

  void interact();
};

void NoiseSource::interact() {
  ImGui::Image(texture, sf::Vector2f(32,20));
  ImGui::SameLine();
  if (ImGui::BeginCombo("Noise", current->name.c_str())) {
    for (unsigned i=0; i < assets::noise::file_count; ++i) {
      bool is_selected = (current == &assets::noise::files[i]);
      if (ImGui::Selectable(assets::noise::files[i].name.c_str(), is_selected)) {
        current = &assets::noise::files[i];
        texture.loadFromMemory(current->data, current->size);
        break;
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
}
