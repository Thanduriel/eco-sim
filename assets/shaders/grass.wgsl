#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    mesh_functions,
    prepass_io::{Vertex, VertexOutput}
    view_transformations::position_world_to_clip
}
#else
#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip
}

#endif

const GRAVITY : f32 = 3.0;
const MIN_BEND_ANGLE : f32 = 0.05;
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

    var h = in.position.y;
    // angle between surface normal / gravity direction and blade
    // cos(b) == v * n / ||v|| with n = (0,1,0) 
    var cos_b = h / length(in.position);
    var b = acos(cos_b);

    if (b >= MIN_BEND_ANGLE) {
        // angle between ground and v
        var a = 0.5 * PI - b;
        var tan_a = tan(a);
        var cos_a = cos(a);
        // using the height as argument directly only works for very small a
        // instead we take the x of the linear function (rotated but not bend)
        var x = cos_a / h;
        var x_3 = x * x * x;
        // change in height
        var h_bend = -GRAVITY * (x_3 - x_3 * x / 2.0) * cos_a; 

        // base function before bending is f(x) = tan(a) * x;
        world_pos.y += h_bend;
    }

    // fill output struct
    var out : VertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);

    // Pass-through
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        in.normal, in.instance_index
    );
    //out.color = in.color;
    //out.uv = in.uv;

    return out;
}