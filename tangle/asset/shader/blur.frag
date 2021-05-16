#version 330

uniform sampler2D texture;

layout(location = 0) out vec4 color;
layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

float kernel[7] = float[]( 0.00443184841193801, 0.0539909665131881, 0.241970724519143, 0.398942280401433, 0.241970724519143, 0.0539909665131881, 0.00443184841193801 );

void main() {
  vec3 blurred = vec3(0.0);

  for (int i=-3; i <= 3; ++i) {
    for (int j=-3; j <= 3; ++j) {
      blurred += kernel[3+j] * kernel[3+i] * texelFetch(texture, ivec2(gl_FragCoord.xy) + ivec2(i,j), 0).xyz;
    }
  }

  color = vec4(blurred, 1.0);
}
