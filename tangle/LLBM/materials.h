#pragma once

#include "memory.h"
#include "sdf.h"

template <typename DESCRIPTOR>
class CellMaterials : public SharedVector<int> {
private:
  const descriptor::Cuboid<DESCRIPTOR> _cuboid;
  int* const _materials;

public:
  CellMaterials(descriptor::Cuboid<DESCRIPTOR> cuboid):
    SharedVector<int>(cuboid.volume),
    _cuboid(cuboid),
    _materials(this->host()) { }

  template <typename F>
  CellMaterials(descriptor::Cuboid<DESCRIPTOR> cuboid, F f):
    CellMaterials(cuboid) {
    set(f);
  }

  descriptor::Cuboid<DESCRIPTOR> cuboid() const {
    return _cuboid;
  };
 
  int get(std::size_t iCell) const {
    return _materials[iCell];
  }
  
  void set(std::size_t iCell, int material) {
    _materials[iCell] = material;
  }
  template <typename F>
  void set(F f) {
    for (std::size_t iCell=0; iCell < _cuboid.volume; ++iCell) {
      set(iCell, f(gidInverse(_cuboid, iCell)));
    }
  }
  
  template <typename S>
  void sdf(S distance, int material, float eps=1e-2) {
    for (std::size_t iCell=0; iCell < _cuboid.volume; ++iCell) {
      auto p = gidInverseSmooth(_cuboid, iCell);
      if (distance(p) < eps) {
        set(iCell, material);
      }
    }
  }
  
  void clean(int material) {
    for (std::size_t iCell=0; iCell < _cuboid.volume; ++iCell) {
      if (get(iCell) == material) {
        if (_cuboid.isInside(iCell)) {
          bool surrounded = true;
          for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
            int m = get(descriptor::neighbor<DESCRIPTOR>(_cuboid, iCell, iPop));
            surrounded &= m == material || m == 0;
          }
          if (surrounded) {
            set(iCell, 0);
          }
        }
      }
    }
  }
  DeviceBuffer<std::size_t> list_of_material(int material) {
  	std::vector<std::size_t> cells;
  	for (std::size_t iCell=0; iCell < _cuboid.volume; ++iCell) {
  		if (_materials[iCell] == material) {
  			cells.emplace_back(iCell);
  		}
  	}
  	return DeviceBuffer<std::size_t>(cells);
  }
  DeviceBuffer<bool> mask_of_material(int material) {
    std::unique_ptr<bool[]> mask(new bool[_cuboid.volume]{});
  	for (std::size_t iCell=0; iCell < _cuboid.volume; ++iCell) {
      mask[iCell] = (_materials[iCell] == material);
  	}
  	return DeviceBuffer<bool>(mask.get(), _cuboid.volume);
  }
  std::size_t get_link_count(int bulk, int solid) {
    std::size_t count = 0;
    for (pop_index_t iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
      for (std::size_t iCell=0; iCell < _cuboid.volume; ++iCell) {
        std::size_t jCell = descriptor::neighbor<DESCRIPTOR>(_cuboid, iCell, iPop);
        if (get(iCell) == bulk && get(jCell) == solid) {
          count++;
        }
      }
    }
    return count;
  }
  
  template <typename F>
  void for_links(int bulk, int solid, F f) {
    for (pop_index_t iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
      for (std::size_t iCell=0; iCell < _cuboid.volume; ++iCell) {
        std::size_t jCell = descriptor::neighbor<DESCRIPTOR>(_cuboid, iCell, iPop);
        if (get(iCell) == bulk && get(jCell) == solid) {
          f(iCell, iPop);
        }
      }
    }
  }
};
