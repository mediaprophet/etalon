//! Pure POD camera and manifold projection contracts.

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameCameraPod {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub eye_pos: [f32; 4],
    pub viewport_wh_near_far: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ManifoldProjectionPod {
    pub coords_10d: [f32; 10],
    pub opacity: f32,
    pub flags: u32,
}
