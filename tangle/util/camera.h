#include <cuda-samples/Common/helper_math.h>
#include "SFML/Window/Event.hpp"

class Camera {
private:
  float _distance;
  float _phi;
  float _psi;
  float3 _target;
  float3 _eye;
  float3 _direction;
  bool _dragging;
  bool _moving;
  float2 _lastMouse;

public:
  Camera(float3 target, float distance, float phi, float psi):
    _distance(distance),
    _phi(phi),
    _psi(psi),
    _target(target),
    _dragging(false),
    _moving(false) {
    update();
  }

  void update() {
    _eye = _target + make_float3(_distance*sin(_psi)*cos(_phi), _distance*sin(_psi)*sin(_phi), _distance*cos(_psi));
    _direction = normalize(_target - _eye);
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
        _phi += 0.4*delta.x * 2*M_PI/360;
        if (delta.y > 0 && _psi <= M_PI-2*M_PI/60) {
          _psi += 0.4*delta.y * M_PI/180;
        } else if (delta.y < 0 && _psi >= 2*M_PI/60) {
          _psi += 0.4*delta.y * M_PI/180;
        }
      }
      if (_moving) {
        float2 mouse = make_float2(event.mouseMove.x, event.mouseMove.y);
        float2 delta = mouse - _lastMouse;
        _lastMouse = mouse;
        float3 forward = normalize(_target - _eye);
    	   float3 right   = normalize(cross(make_float3(0.f, 0.f, -1.f), forward));
        float3 up      = cross(right, forward);
        _target += 0.4*right*delta.x - 0.4*up*delta.y;
      }
      break;
    }
    update();
  }

  float3 getDirection() const {
    return _direction;
  }

  float3 getEyePosition() const {
    return _eye;
  }
};
