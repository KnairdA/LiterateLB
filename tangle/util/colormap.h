#pragma once
#include "assets.h"
#include "texture.h"

#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/Graphics.hpp>

struct ColorPalette {
  const assets::File* current;
  sf::Texture texture;

  ColorPalette(cudaSurfaceObject_t& palette) {
    current = &assets::palette::files[5];
    texture.loadFromMemory(current->data, current->size);
    palette = bindTextureToCuda(texture);
  }

  void interact();
};

void ColorPalette::interact() {
  if (ImGui::BeginCombo("Color palette", current->name.c_str())) {
    for (unsigned i=0; i < assets::palette::file_count; ++i) {
      bool is_selected = (current == &assets::palette::files[i]);
      if (ImGui::Selectable(assets::palette::files[i].name.c_str(), is_selected)) {
        current = &assets::palette::files[i];
        texture.loadFromMemory(current->data, current->size);
        break;
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::Image(texture, sf::Vector2f(400.,40.));
}
