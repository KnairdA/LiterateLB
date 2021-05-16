#pragma once
#include <chrono>

namespace timer {

std::chrono::time_point<std::chrono::steady_clock> now() {
  return std::chrono::steady_clock::now();
}

double secondsSince(
  std::chrono::time_point<std::chrono::steady_clock>& pit) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(now() - pit).count();
}

double mlups(std::size_t nCells, std::size_t nSteps, std::chrono::time_point<std::chrono::steady_clock>& start) {
  return nCells * nSteps / (secondsSince(start) * 1e6);
}

}
