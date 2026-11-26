//! Phase 4: SPH particle point sprites (billboards reading the live sim position buffer).
//!
//! The mesh is `capacity * 4` vertices (one quad per allocated particle slot);
//! `vertex_index` decodes quad and corner. Quads past the live count (the stream brush
//! appends particles up to the capacity at runtime) are discarded via the `count` flag.
//! The vertex shader reads the particle's position from the storage buffer (mirrored
//! GPU-side by `gpu::copy_particles_to_shader_buffer` — no CPU round-trips), billboards
//! the quad using the camera's right/up axes (`view.world_from_view`), and the fragment
//! shader draws a soft round sprite with a bright core.

import bevy_render::view::View;

// Must match `particles::ParticleUniform` (16 bytes, 16-byte aligned).
struct ParticleUniform {
    size: f32,
    alpha: f32,
    count: f32,
    _pad: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0)
var<uniform> particle_uniform: ParticleUniform;

@group(constants::MATERIAL_BIND_GROUP) @binding(1)
var<storage, read> particles: array<vec4<f32>>;

@group(0) @binding(0)
var<uniform> view: View;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let quad = in.vertex_index / 4u;
    let corner = in.vertex_index % 4u;
    var out: VertexOutput;
    if (f32(quad)) >= particle_uniform.count {
        // The mesh is built at the full particle capacity (the stream brush grows the
        // live count without re-uploading): push quads past the live count outside the
        // clip volume instead of drawing them at a stale position.
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = in.uv;
        return out;
    }
    var offs = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5),
        vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, 0.5),
    );
    let off = offs[corner] * particle_uniform.size;
    let center = particles[quad];
    // Billboard: offset the center along the camera's right/up axes (world space).
    let world_pos = center.xyz
        + view.world_from_view[0].xyz * off.x
        + view.world_from_view[1].xyz * off.y;

    out.position = view.clip_from_world * vec4<f32>(world_pos, 1.0);
    out.uv = in.uv;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    // Round sprite: hard-core, soft rim (the smoothstep band is the antialiased edge).
    let r = length(in.uv - vec2<f32>(0.5)) * 2.0;
    let a = smoothstep(0.5, 0.25, r) * particle_uniform.alpha;
    if a < 0.01 {
        discard;
    }
    let rim = smoothstep(0.0, 0.6, r);
    let base = vec3<f32>(0.35, 0.74, 0.94);
    let core = vec3<f32>(0.70, 0.93, 1.0);
    let color = mix(core, base, rim);

    var out: FragmentOutput;
    out.color = vec4<f32>(color, a);
    return out;
}
