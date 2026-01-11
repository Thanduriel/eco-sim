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

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    /* --- Model → World transform (Bevy helper) ------------------- */
    var model = mesh_functions::get_world_from_local(in.instance_index);
//    var world_pos = mesh_functions::mesh_position_local_to_world(
//        model,
 //       vec4<f32>(in.position, 1.0)
 //   );
    var v = mesh_functions::mesh_position_local_to_world(
        model,
        vec4<f32>(in.position, 0.0)
    );

    // get effective gravitational force g * (1 - cos(b))
    // where cos(b) == v * n / ||v|| with n = (0,1,0) 
   // var s = v.y / length(v);

    var world_pos = v + vec4<f32>(model[3].xyz, 1.0);

    var h = in.position.y;
    if h > 0.1 {
        world_pos.y -= 0.5 * (h-0.1) * (h-0.1);
    }

    /* --- Fill the required output struct ------------------------- */
    var out : VertexOutput;
    out.world_position = world_pos;
    out.position = position_world_to_clip(world_pos.xyz);

    /* Pass-through you may need later */
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        in.normal, in.instance_index
    );
    
    //out.color = in.color;
    //out.uv = in.uv;

    return out;
}