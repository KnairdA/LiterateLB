#include <cuda-samples/Common/helper_math.h>

#include <LLBM/sdf.h>

__device__ float2 getNormalizedScreenPos(float w, float h, float x, float y) {
	return make_float2(
		2.f * (.5f - x/w) * w/h,
		2.f * (.5f - y/h)
	);
}

__device__ float3 getEyeRayDir(float2 screen_pos, float3 eye_pos, float3 eye_target) {
	const float3 forward = normalize(eye_target - eye_pos);
	const float3 right   = normalize(cross(make_float3(0.f, 0.f, -1.f), forward));
	const float3 up      = normalize(cross(forward, right));

	return normalize(screen_pos.x*right + screen_pos.y*up + 4*forward);
}

__device__ bool aabb(float3 origin, float3 dir, float3 min, float3 max, float& tmin, float& tmax) {
  float3 invD = make_float3(1./dir.x, 1./dir.y, 1./dir.z);
  float3 t0s = (min - origin) * invD;
  float3 t1s = (max - origin) * invD;
  float3 tsmaller = fminf(t0s, t1s);
  float3 tbigger  = fmaxf(t0s, t1s);
  tmin = fmaxf(tmin, fmaxf(tsmaller.x, fmaxf(tsmaller.y, tsmaller.z)));
  tmax = fminf(tmax, fminf(tbigger.x, fminf(tbigger.y, tbigger.z)));
  return (tmin < tmax);
}

__device__ bool aabb(float3 origin, float3 dir, descriptor::CuboidD<3>& cuboid, float& tmin, float& tmax) {
  return aabb(origin, dir, make_float3(0), make_float3(cuboid.nX,cuboid.nY,cuboid.nZ), tmin, tmax);
}

struct VolumetricRenderConfig {
  descriptor::CuboidD<3> cuboid;

  cudaSurfaceObject_t palette;
  cudaSurfaceObject_t noise;

  float delta = 1;
  float transparency = 1;
  float brightness = 1;
  float3 background = make_float3(22.f / 255.f);

  float3 eye_pos;
  float3 eye_dir;

  cudaSurfaceObject_t canvas;
  uint2 canvas_size;

  bool align_slices_to_view = true;
  bool apply_noise = true;
  bool apply_blur = true;

  VolumetricRenderConfig(descriptor::CuboidD<3> c):
    cuboid(c) { }
};



template <typename SDF, typename SAMPLER, typename ATTENUATOR>
__global__ void raymarch(
  VolumetricRenderConfig config,
  SDF geometry,
  SAMPLER sampler,
  ATTENUATOR attenuator
) {
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x > config.canvas_size.x - 1 || y > config.canvas_size.y - 1) {
    return;
  }

  const float2 screen_pos = getNormalizedScreenPos(config.canvas_size.x, config.canvas_size.y, x, y);
  const float3 ray_dir = getEyeRayDir(screen_pos, config.eye_pos, config.eye_pos + config.eye_dir);

  float3 r = make_float3(0);
  float  a = 0;
  
  float tmin = 0;
  float tmax = 4000;
  
  if (aabb(config.eye_pos, ray_dir, config.cuboid, tmin, tmax)) {
    float volume_dist = tmax - tmin;
    float3 geometry_pos = config.eye_pos + tmin*ray_dir;
    float geometry_dist = approximateDistance(geometry, geometry_pos, ray_dir, 0, volume_dist);
    geometry_pos += geometry_dist * ray_dir;
  
    float jitter = config.align_slices_to_view * (floor(fabs(dot(config.eye_dir, tmin*ray_dir)) / config.delta) * config.delta - tmin)
                 + config.apply_noise          * config.delta * noiseFromTexture(config.noise, threadIdx.x, threadIdx.y);
  
    tmin          += jitter;
    volume_dist   -= jitter;
    geometry_dist -= jitter;
  
    if (volume_dist > config.delta) {
      float3 sample_pos = config.eye_pos + tmin * ray_dir;
      unsigned n_samples = floor(geometry_dist / config.delta);
      for (unsigned i=0; i < n_samples; ++i) {
        sample_pos += config.delta * ray_dir;
        
        float  sample_value = sampler(sample_pos);
        float3 sample_color = config.brightness * colorFromTexture(config.palette, sample_value);
        
        float sample_attenuation = attenuator(sample_value) * config.transparency;
        float attenuation = 1 - a;
        
        r += attenuation * sample_attenuation * sample_color;
        a += attenuation * sample_attenuation;
      }
    }
  
    if (geometry_dist < volume_dist) {
      float3 n = sdf_normal(geometry, geometry_pos);
      r = lerp((0.3f + fabs(dot(n, ray_dir))) * make_float3(0.3f), r, a);
    }
  } else {
    a = 0;
  }
  
  if (a < 1) {
    r += (1 - a) * config.background;
  }

  uchar4 pixel {
    static_cast<unsigned char>(clamp(r.x, 0.0f, 1.0f) * 255),
    static_cast<unsigned char>(clamp(r.y, 0.0f, 1.0f) * 255),
    static_cast<unsigned char>(clamp(r.z, 0.0f, 1.0f) * 255),
    255
  };
  surf2Dwrite(pixel, config.canvas, x*sizeof(uchar4), y);
}
