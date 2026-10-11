@if(PREPASS_PIPELINE)
import bevy_pbr::prepass::io::{Vertex, VertexOutput};
@else
import bevy_pbr::render::forward_io::{Vertex, VertexOutput};

import bevy_pbr::render::mesh_functions;
import bevy_pbr::render::view_transformations::position_world_to_clip;

const GRAVITY: f32 = 20.0;
const EPSILON: f32 = 0.01;
const PI = radians(180.0);

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    /* --- Model → World transform (Bevy helper) ------------------- */
    var model = mesh_functions::get_world_from_local(in.instance_index);
    var v = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(in.position, 0.0)
    );

    var world_pos = v + vec4<f32>(model[3].xyz, 1.0);

    // normal before bending, overridden when the blade is bent
    var normal_world = mesh_functions::mesh_normal_local_to_world(
        in.normal, in.instance_index
    );

    // height in the base mesh in [0,1]
    var h = in.position.y;
    if (h > EPSILON) {
        // angle between surface normal / gravity direction and blade
        // cos(b) == v * n / ||v|| with n = (0,1,0)
        var len_v = length(v);

        if (len_v > EPSILON) {
            var cos_b = v.y / len_v;
            var b = acos(cos_b);

            if (b >= EPSILON) {
                // scaling factor in model matrix
                var scale = len_v / length(in.position);
                // angle between ground and v
                var a = 0.5 * PI - b;
                var cos_a = cos(a);
                // Using the height as argument directly only works for very small a.
                // Instead, we take the x of the linear function (rotated but not bend).
                var x = h * cos_a;
                var x_2 = x * x;
                var x_3 = x_2 * x;
                // change in height
                var h_bend = -GRAVITY * (x_3 - x_3 * x / 2.0) * cos_a;

                // base function before bending is f(x) = tan(a) * x;
                world_pos.y += h_bend * scale;

                // blade axis and width directions in world space
                var axis = normalize(model[1].xyz);
                var width = normalize(model[0].xyz);
                // slope of the bent blade, df/dh of the deflection above
                var bend_slope = -GRAVITY * (3.0 * x_2 - 2.0 * x_3) * cos_a * cos_a;
                // tangent to the bent surface along the blade
                var bend_tangent = normalize(axis + vec3<f32>(0.0, bend_slope, 0.0));
                // normal perpendicular to both the blade width and the bent tangent
                normal_world = normalize(cross(width, bend_tangent));
            }
        }
    }

    // fill output struct
    var out: VertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);
    out.world_normal = normal_world;

    return out;
}
