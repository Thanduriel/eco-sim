import bevy_pbr::render::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
};

@if(PREPASS_PIPELINE)
import bevy_pbr::{
    prepass::io::{VertexOutput, FragmentOutput},
    deferred::functions::deferred_output,
};
@else
import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
};

// Must match `terrain::FieldVisUniform` (32 bytes, 16-byte aligned).
struct FieldVisUniform {
    weight: f32,
    min: f32,
    max: f32,
    field_size: f32,
    domain_size: f32,
    half_texel: f32,
    kind: u32,
    _pad: u32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var<uniform> field_vis: FieldVisUniform;

@group(constants::MATERIAL_BIND_GROUP) @binding(101)
var field_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(102)
var field_sampler: sampler;

@group(constants::MATERIAL_BIND_GROUP) @binding(103)
var color_lut: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104)
var color_lut_sampler: sampler;

const LUT_SIZE: f32 = 256.0;

// Maps a world-space (domain) position, in [0, domain_size]^2, to the normalized texture
// coordinate holding the field value that should be shown there.
//
// * Vertex (point) fields store `buffer[i]` at grid point `i` (world position i*h). GPU
//   texture sampling hits texel centers, so we shift by half a texel to land grid points on
//   texel centers. With a linear sampler this reproduces bilinear interpolation between
//   neighbouring grid points.
// * Cell fields store one value per cell; sampling with no shift and a nearest sampler
//   displays each cell as a block.
fn field_uv(world_pos: vec2<f32>) -> vec2<f32> {
    let domain = world_pos / field_vis.domain_size;
    return domain + vec2<f32>(field_vis.half_texel);
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );

    var out: FragmentOutput;
    @if(PREPASS_PIPELINE) {
        // Deferred mode: only the pre-lighting PbrInput can be modified.
        out = deferred_output(in, pbr_input);
    } @else {
        if (field_vis.weight > 0.5) {
            let value = textureSampleLevel(
                field_texture,
                field_sampler,
                field_uv(in.world_position.xz),
                0.0
            ).r;

            var t = (value - field_vis.min) / (field_vis.max - field_vis.min);
            t = clamp(t, 0.0, 1.0);

            // Sample the centre of the corresponding LUT texel, letting the linear filter
            // interpolate between the color stops.
            let lut_uv = vec2<f32>(t * ((LUT_SIZE - 1.0) / LUT_SIZE) + 0.5 / LUT_SIZE, 0.5);
            let field_color = textureSampleLevel(color_lut, color_lut_sampler, lut_uv, 0.0);

            // Override the surface albedo with the field color, then light it normally. This
            // keeps the terrain lit (including its normal map) as if it were painted, instead
            // of baking the lit color on the CPU. (Build the whole vec4 — naga rejects
            // assigning to a multi-component swizzle such as `.rgb`.)
            pbr_input.material.base_color = vec4<f32>(
                mix(pbr_input.material.base_color.rgb, field_color.rgb, field_vis.weight),
                pbr_input.material.base_color.a
            );
        }

        out.color = apply_pbr_lighting(pbr_input);
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    }

    return out;
}
