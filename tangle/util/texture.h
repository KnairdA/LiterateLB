#pragma once

#include <cstring>
#include <SFML/Graphics.hpp>
#include <cuda_gl_interop.h>
#include <LLBM/memory.h>

cudaSurfaceObject_t bindTextureToCuda(sf::Texture& texture) {
  GLuint gl_tex_handle = texture.getNativeHandle();
  cudaGraphicsResource* cuda_tex_handle;
  cudaArray* buffer;

  cudaGraphicsGLRegisterImage(&cuda_tex_handle, gl_tex_handle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
  cudaGraphicsMapResources(1, &cuda_tex_handle, 0);
  cudaGraphicsSubResourceGetMappedArray(&buffer, cuda_tex_handle, 0, 0);

  cudaResourceDesc resDesc;
  resDesc.resType = cudaResourceTypeArray;

  resDesc.res.array.array = buffer;
  cudaSurfaceObject_t cudaSurfaceObject = 0;
  cudaCreateSurfaceObject(&cudaSurfaceObject, &resDesc);

  return cudaSurfaceObject;
}
