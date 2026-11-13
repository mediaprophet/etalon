use super::*;

#[test]
fn receiver_plane_pcf_lights_sloped_receivers_but_keeps_nearby_caster_shadows(){
    for resolution in [512.,2048.] {
        let texel=1.0/resolution;
        for slope in [[0.,0.],[0.25,-0.125],[-0.8,0.35]] {
            for scale in [0.001,1.,1000.] {
                let plane=receiver_plane_gradient([scale,0.,slope[0]*scale],[0.,scale,slope[1]*scale],texel,0.0015);
                assert!(plane.reliable);
                for fraction in [[0.,0.],[0.00001,0.99999],[0.25,0.75],[0.5,0.5],[0.99999,0.00001]] {
                    for y in -1..=1 {for x in -1..=1 {
                        let compare=receiver_compare_depth(0.5,plane,[x as f32,y as f32],fraction,1.,0.0015);
                        for corner_y in 0..2 {for corner_x in 0..2 {
                            if (fraction[0]==0.&&corner_x==1)||(fraction[1]==0.&&corner_y==1){continue;}
                            let dx=if corner_x==0{-fraction[0]}else{1.-fraction[0]};
                            let dy=if corner_y==0{-fraction[1]}else{1.-fraction[1]};
                            // Independent analytic depths at every nonzero hardware tap.
                            let receiver=0.5+texel*(slope[0]*(x as f32+dx)+slope[1]*(y as f32+dy));
                            assert!(compare<=receiver+f32::EPSILON,"self-shadow at {resolution} {slope:?} {fraction:?}");
                            assert!(compare>receiver-0.01,"real caster must still occlude the receiver");
                        }}
                    }}
                }
            }
        }
    }
    let face_on=receiver_plane_gradient([1.,0.,0.],[0.,1.,0.],1./2048.,0.0015);
    let compare=receiver_compare_depth(0.5,face_on,[0.,0.],[0.5,0.5],1.,0.0015);
    assert!(compare<0.5&&compare>0.5-0.00001,"precision floor keeps close contact shadows");
}

#[test]
fn receiver_plane_singular_grazing_and_nonfinite_inputs_have_bounded_fallback(){
    for (dx,dy) in [([0.;3],[0.;3]),([1.,2.,3.],[2.,4.,6.]),
        ([1.,0.,1.],[1.,0.000001,1.]),([0.001,0.,1.],[0.,0.001,0.]),
        ([f32::NAN,0.,0.],[0.,1.,0.]),([f32::INFINITY,0.,1.],[0.,1.,0.])] {
        let plane=receiver_plane_gradient(dx,dy,1./2048.,0.0015);
        assert!(!plane.reliable);
        assert!(plane.depth_per_texel.iter().all(|v|v.is_finite()&&v.abs()<=4./2048.));
        let compare=receiver_compare_depth(0.5,plane,[1.,-1.],[0.25,0.75],0.2,0.0015);
        assert!(compare.is_finite()&&compare<0.5&&compare>=0.5-0.0015);
        assert!(compare>0.49,"fallback keeps an actual caster shadow");
    }
}

#[test]
fn mesh_receiver_derivatives_and_cascades_validate_as_one_production_shader(){
    let module=naga::front::wgsl::parse_str(crate::shaders::MESH_WGSL).expect("provider mesh WGSL parses");
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(),naga::valid::Capabilities::all())
        .validate(&module).expect("derivatives remain in uniform control flow before discards/cascade branches");
}
