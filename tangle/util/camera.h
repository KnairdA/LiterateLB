#include <cuda-samples/Common/helper_math.h>
#include <glm/gtx/quaternion.hpp>
#include "SFML/Window/Event.hpp"

class Camera {
private:
  glm::quat _rotation;
  glm::vec3 _target;
  glm::vec3 _position;
  glm::vec3 _forward;
  glm::vec3 _right;
  glm::vec3 _up;
  float _distance;
  bool _dragging;
  bool _moving;
  float2 _lastMouse;

public:
  Camera(float3 target, float distance):
    _distance(distance),
    _target(target.x, target.y, target.z),
    _dragging(false),
    _moving(false) {
    update();
  }

  void update() {
    _position = _target + glm::axis(_rotation * glm::quat(0, 0, _distance, 0) * glm::conjugate(_rotation));
    _forward = glm::normalize(_target - _position);
    _right = glm::axis(_rotation * glm::quat(0, -1, 0, 0) * glm::conjugate(_rotation));
    _up = glm::axis(_rotation * glm::quat(0, glm::cross(glm::vec3(0, 1, 0), glm::vec3(-1, 0, 0))) * glm::conjugate(_rotation));
  }

  void handle(sf::Event& event) {
    switch (event.type) {
    case sf::Event::MouseWheelMoved:
      _distance -= event.mouseWheel.delta * 10;
      break;
    case sf::Event::MouseButtonPressed:
      if (event.mouseButton.button == sf::Mouse::Left) {
        _dragging = true;
        _lastMouse = make_float2(event.mouseButton.x, event.mouseButton.y);
      } else if (event.mouseButton.button == sf::Mouse::Right) {
        _moving = true;
        _lastMouse = make_float2(event.mouseButton.x, event.mouseButton.y);
      }
      break;
    case sf::Event::MouseButtonReleased:
      if (event.mouseButton.button == sf::Mouse::Left) {
        _dragging = false;
      } else if (event.mouseButton.button == sf::Mouse::Right) {
        _moving = false;
      }
      break;
    case sf::Event::MouseMoved:
      float2 mouse = make_float2(event.mouseMove.x, event.mouseMove.y);
      if (_dragging) {
        float2 delta = 0.005 * (mouse - _lastMouse);
        glm::quat rotation_z = glm::vec3(0,0,delta.x);
        glm::quat rotation_x = glm::vec3(delta.y,0,0);
        _rotation *= glm::cross(rotation_x, rotation_z);
      }
      if (_moving) {
        float2 delta = 0.04 * (mouse - _lastMouse);
        _target += _right*delta.x + _up*delta.y;
      }
      _lastMouse = mouse;
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
