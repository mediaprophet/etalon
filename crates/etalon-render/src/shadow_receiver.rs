//! Zero-heap CPU reference for the provider's shadow receiver-plane math.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReceiverPlane { pub depth_per_texel:[f32;2], pub reliable:bool }
pub const RECEIVER_DEPTH_EPSILON:f32=8.0*f32::EPSILON;

pub fn receiver_plane_gradient(dx:[f32;3],dy:[f32;3],texel:f32,bias_limit:f32)->ReceiverPlane {
    if !dx.iter().chain(dy.iter()).all(|v|v.is_finite())||!texel.is_finite()||texel<=0.0||!bias_limit.is_finite(){return ReceiverPlane::default();}
    let scale=dx[0].abs().max(dx[1].abs()).max(dy[0].abs()).max(dy[1].abs());
    if scale<=1e-12{return ReceiverPlane::default();}
    let x=dx.map(|v|v/scale);let y=dy.map(|v|v/scale);let determinant=x[0]*y[1]-x[1]*y[0];
    if determinant.abs()<=1e-5{return ReceiverPlane::default();}
    let step=[(y[1]*x[2]-x[1]*y[2])*(texel/determinant),(x[0]*y[2]-y[0]*x[2])*(texel/determinant)];
    if step.iter().any(|v|v.is_nan()){return ReceiverPlane::default();}
    let limit=bias_limit.abs().max(4.0*texel).min(1.0);
    ReceiverPlane{depth_per_texel:step.map(|v|v.clamp(-limit,limit)),reliable:step.iter().all(|v|v.abs()<=limit)}
}

pub fn receiver_compare_depth(depth:f32,plane:ReceiverPlane,tap:[f32;2],fraction:[f32;2],normal_light:f32,bias_limit:f32)->f32 {
    let bias=if plane.reliable{
        let mut nearest=0.0;
        for axis in 0..2 {
            let lower=-fraction[axis];let upper=if fraction[axis]>0.0{1.0-fraction[axis]}else{0.0};
            nearest+=(plane.depth_per_texel[axis]*lower).min(plane.depth_per_texel[axis]*upper);
        }
        RECEIVER_DEPTH_EPSILON-nearest
    }
        else{RECEIVER_DEPTH_EPSILON.max(bias_limit.abs().clamp(0.0,1.0)*(1.0-normal_light.clamp(0.0,1.0)))};
    let correction=if plane.reliable{plane.depth_per_texel[0]*tap[0]+plane.depth_per_texel[1]*tap[1]}else{0.0};
    depth+correction-bias
}

#[cfg(test)]
#[path="shadow_receiver_tests.rs"]
mod tests;
