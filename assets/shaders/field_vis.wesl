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

// The field texture is `Rgba32Float`: not linearly filterable without the
// `float32-filterable` feature, so values are re-sampled manually with `textureLoad` +
// bilinear interpolation. This also makes the texel mapping exact:
//
// * Vertex (point) fields store `buffer[i]` at grid point `i` (world position i*h). Texel
//   `i` holds that value, so the texel coordinate of a world position is
//   `world_pos / domain_size * field_size` and interpolation happens between texels
//   `floor(t)` and `floor(t)+1`.
// * Cell fields store one value per cell; the nearest texel displays each cell as a block.
// Clamped `textureLoad` of one field texel.
fn field_texel(x: i32, y: i32) -> f32 {
    let n_i32 = i32(field_vis.field_size);
    let coord = vec2<u32>(
        u32(clamp(x, 0, n_i32 - 1)),
        u32(clamp(y, 0, n_i32 - 1)),
    );
    return textureLoad(field_texture, coord, 0).r;
}

fn field_value(world_pos: vec2<f32>) -> f32 {
    let n = field_vis.field_size;
    let domain = field_vis.domain_size;
    let t = world_pos / domain * n;

    if (field_vis.kind == 1u) {
        // Cell field: nearest texel.
        let coord = vec2<u32>(clamp(round(t), vec2<f32>(0.0), vec2<f32>(n - 1.0)));
        return textureLoad(field_texture, coord, 0).r;
    }

    // Vertex field: bilinear between the four surrounding texels.
    let base = vec2<i32>(floor(t));
    let f = t - vec2<f32>(base);
    let v00 = field_texel(base.x, base.y);
    let v10 = field_texel(base.x + 1, base.y);
    let v01 = field_texel(base.x, base.y + 1);
    let v11 = field_texel(base.x + 1, base.y + 1);
    return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
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
            let value = field_value(in.world_position.xz);

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
