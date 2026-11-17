import bevy_pbr::render::forward_io::{Vertex, VertexOutput, FragmentOutput};
import bevy_pbr::render::mesh_functions;
import bevy_pbr::render::view_transformations::position_world_to_clip;

// Must match `water_surface::WaterSurfaceUniform` (32 bytes, 16-byte aligned).
struct WaterSurfaceUniform {
    center: vec2<f32>,
    world_size: f32,
    depth_scale: f32,
    alpha_scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0)
var<uniform> surface_uniform: WaterSurfaceUniform;

@group(constants::MATERIAL_BIND_GROUP) @binding(1)
var surface_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2)
var surface_sampler: sampler;

const SURFACE_SIZE: f32 = 256.0;

// The surface texture is `Rgba32Float`: not linearly filterable without the
// `float32-filterable` feature, so it is re-sampled manually with `textureLoad` +
// bilinear interpolation (same approach as `field_vis.wesl`). Each cell's height is
// stored at its texel; interpolating between texel centers makes a smooth water surface.
fn surface_value(uv: vec2<f32>) -> vec2<f32> {
    let t = uv * SURFACE_SIZE;
    let base = vec2<i32>(floor(t));
    let f = t - vec2<f32>(base);
    let n_min = 0;
    let n_max = 255;
    let x0 = u32(clamp(base.x, n_min, n_max));
    let y0 = u32(clamp(base.y, n_min, n_max));
    let x1 = u32(clamp(base.x + 1, n_min, n_max));
    let y1 = u32(clamp(base.y + 1, n_min, n_max));
    let v00 = textureLoad(surface_texture, vec2<u32>(x0, y0), 0).xy;
    let v10 = textureLoad(surface_texture, vec2<u32>(x1, y0), 0).xy;
    let v01 = textureLoad(surface_texture, vec2<u32>(x0, y1), 0).xy;
    let v11 = textureLoad(surface_texture, vec2<u32>(x1, y1), 0).xy;
    return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    // The mesh is a centered plane; the entity transform places it on the domain
    // (Transform::from_xyz(CENTER.x, 0, CENTER.y)).
    var model = mesh_functions::get_world_from_local(in.instance_index);
    var world_pos = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(in.position, 1.0)
    );

    // Texture UV: texel (0,0) covers world (0,0) .. (h, h); bilinear between texel centers.
    let uv = ((world_pos.xz - surface_uniform.center) + surface_uniform.world_size * 0.5)
        / surface_uniform.world_size;
    let s = surface_value(uv);
    // Lift the vertex to the water surface height.
    world_pos.y = s.x;

    var out: VertexOutput;
    out.world_position = world_pos;
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.position = position_world_to_clip(world_pos.xyz);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    let uv = ((in.world_position.xz - surface_uniform.center) + surface_uniform.world_size * 0.5)
        / surface_uniform.world_size;
    let s = surface_value(uv);
    let depth = s.y;
    if (depth < 0.005) {
        // No water in this cell: hide the plane (it lies at y=0, below most terrain).
        discard;
    }

    // Depth tint: bright turquoise at the shore, saturated BLUE in deep water. Both
    // the original (0.02,0.16,0.38) and the intermediate (0.06,0.30,0.60) deep colors
    // still read as "dark basin shading" at alpha ~0.9 over the shadowed floors
    // (settled pools are 0.7-1.9 m deep -> t = 0.35-0.95). The ramp must never leave
    // the legibly-blue range for realistic depths.
    let t = clamp(depth / surface_uniform.depth_scale, 0.0, 1.0);
    let shallow = vec3<f32>(0.42, 0.76, 0.84);
    let deep = vec3<f32>(0.10, 0.42, 0.72);
    let base = mix(shallow, deep, t);
    // Alpha floor keeps even the thinnest films readable as water (not invisible).
    let alpha = clamp(depth / surface_uniform.alpha_scale, 0.25, 1.0) * 0.9;

    var out: FragmentOutput;
    out.color = vec4<f32>(base, alpha);
    return out;
}
