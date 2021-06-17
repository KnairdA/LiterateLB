#include <cuda-samples/Common/helper_math.h>
#include "SFML/Window/Event.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/common.hpp>

class Camera {
private:
  glm::quat _rotation;
  glm::mat3 _matrix;
  float3 _target;
  float3 _position;
  float3 _forward;
  float3 _right;
  float3 _up;
  float _distance;
  bool _dragging;
  bool _moving;
  float2 _lastMouse;

public:
  Camera(float3 target, float distance):
    _distance(distance),
    _target(target),
    _dragging(false),
    _moving(false) {
    update();
  }

  void update() {
    glm::vec3 position = _matrix * glm::vec3(0, _distance, 0);
    _position = _target + make_float3(position[0], position[1], position[2]);
    _forward = normalize(_target - _position);
    
    glm::vec3 right = _matrix * glm::vec4(-1, 0, 0, 0);
    _right = make_float3(right[0], right[1], right[2]);
    glm::vec3 up = _matrix * glm::vec4(glm::cross(glm::vec3(0, 1, 0), glm::vec3(-1, 0, 0)), 0);
    _up = make_float3(up[0], up[1], up[2]);
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
      if (_dragging) {
        float2 mouse = make_float2(event.mouseMove.x, event.mouseMove.y);
        float2 delta = mouse - _lastMouse;
        _lastMouse = mouse;
    
        glm::quat rotation_z = glm::conjugate(_rotation) * glm::vec3(0,0,0.01*delta.x);
        glm::quat rotation_x = glm::conjugate(_rotation) * glm::vec3(0.01*delta.y,0,0);
    
        _matrix = _matrix * glm::mat3_cast(_rotation * glm::cross(rotation_x, rotation_z));
      }
      if (_moving) {
        float2 mouse = make_float2(event.mouseMove.x, event.mouseMove.y);
        float2 delta = mouse - _lastMouse;
        _lastMouse = mouse;
        _target += 0.4*_right*delta.x + 0.4*_up*delta.y;
      }
      break;
    }
    update();
  }

  float3 getPosition() const {
    return _position;
  }
  float3 getForward() const {
    return _forward;
  }
  float3 getRight() const {
    return _right;
  }
  float3 getUp() const {
    return _up;
  }
};
