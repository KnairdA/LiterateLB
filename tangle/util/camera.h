#pragma once
#include <cuda-samples/Common/helper_math.h>
#include <glm/gtx/quaternion.hpp>
#include "SFML/Window/Event.hpp"

glm::vec3 apply(glm::quat q, glm::vec3 v) {
  return glm::axis(q * glm::quat(0, v) * glm::conjugate(q));
}

class Camera {
private:
glm::quat _rotation;

glm::vec3 _target;
glm::vec3 _position;
glm::vec3 _forward;
glm::vec3 _right;
glm::vec3 _up;
float _distance;

float2 _mouse;

bool _rotating;
bool _moving;

bool _restricted_x;
bool _restricted_y;

void update() {
  _position = _target + apply(_rotation, glm::vec3(0, _distance, 0));
  _forward = glm::normalize(_target - _position);
  _right = apply(_rotation, glm::vec3(-1, 0, 0));
  _up = apply(_rotation, glm::cross(glm::vec3(0, 1, 0), glm::vec3(-1, 0, 0)));
}

public:
Camera(float3 target, float distance):
  _distance(distance),
  _target(target.x, target.y, target.z),
  _rotating(false),
  _rotation(1,0,0,0),
  _moving(false),
  _restricted_x(false),
  _restricted_y(false) {
  update();
}

void handle(sf::Event& event) {
  switch (event.type) {
  case sf::Event::KeyPressed:
    if (event.key.code == sf::Keyboard::LShift && !_restricted_x && !_restricted_y) {
      _restricted_x = true;
      _restricted_y = true;
    }
    break;
  case sf::Event::KeyReleased:
    if (event.key.code == sf::Keyboard::LShift) {
      _restricted_x = false;
      _restricted_y = false;
    }
    break;
  case sf::Event::MouseButtonPressed:
    if (event.mouseButton.button == sf::Mouse::Left) {
      _rotating = true;
    } else if (event.mouseButton.button == sf::Mouse::Right) {
      _moving = true;
    }
    _mouse = make_float2(event.mouseButton.x, event.mouseButton.y);
    break;
  case sf::Event::MouseButtonReleased: {
    bool restricted = _restricted_x + _restricted_y;
    _restricted_x = restricted;
    _restricted_y = restricted;
    _rotating = false;
    _moving = false;
    break;
  }

  case sf::Event::MouseWheelMoved:
    _distance -= event.mouseWheel.delta * 10;
    break;

  case sf::Event::MouseMoved:
    float2 mouse = make_float2(event.mouseMove.x, event.mouseMove.y);
    if (_rotating) {
      float2 delta = 0.005 * (mouse - _mouse);
      if (_restricted_x && _restricted_y) {
        if (std::abs(delta.x) > std::abs(delta.y)) {
          _restricted_y = false;
        } else {
          _restricted_x = false;
        }
      }
      if (_restricted_x) { delta.y = 0; }
      if (_restricted_y) { delta.x = 0; }
      glm::quat rotation_z = glm::vec3(0,0,delta.x);
      glm::quat rotation_x = glm::vec3(delta.y,0,0);
      _rotation *= glm::cross(rotation_x, rotation_z);
    }

    if (_moving) {
      float2 delta = 0.04 * (mouse - _mouse);
      _target += _right*delta.x + _up*delta.y;
    }
    _mouse = mouse;
    break;
  }
  update();
}

float3 getPosition() const {
  return make_float3(_position.x, _position.y, _position.z);
}
float3 getForward() const {
  return make_float3(_forward.x, _forward.y, _forward.z);
}
float3 getRight() const {
  return make_float3(_right.x, _right.y, _right.z);
}
float3 getUp() const {
  return make_float3(_up.x, _up.y, _up.z);
}
};
