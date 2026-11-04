//! Resident native/WASM mesh buffers uploaded on an embedding-owned device.
use etalon as wgpu;
use wgpu::util::DeviceExt;
pub struct MeshBuffers<R> {
    pub vertex_buf: wgpu::Buffer,
    pub color_buf: wgpu::Buffer,
    pub normal_buf: wgpu::Buffer,
    pub tangent_buf: wgpu::Buffer,
    pub uv_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub vertex_count: u32,
    _reservation: R,
}
impl<R> MeshBuffers<R> {
    pub fn upload(
        device: &wgpu::Device,
        positions: &[[f32; 3]],
        colors: &[[f32; 4]],
        normals: &[[f32; 3]],
        tangents: &[[f32; 4]],
        uvs: &[[f32; 2]],
        indices: &[u32],
        reservation: R,
    ) -> Result<Self, String> {
        let vertex_count =
            u32::try_from(positions.len()).map_err(|_| "Mesh vertex count exceeds GPU ABI")?;
        if colors.len() != positions.len()
            || normals.len() != positions.len()
            || tangents.len() != positions.len()
            || uvs.len() != positions.len()
            || indices.iter().any(|&i| i >= vertex_count)
        {
            return Err("Mesh upload streams do not match the vertex/index ABI".into());
        }
        let vertex = |label, contents| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            })
        };
        Ok(Self {
            vertex_buf: vertex("portal-mesh-verts", bytemuck::cast_slice(positions)),
            color_buf: vertex("portal-mesh-colors", bytemuck::cast_slice(colors)),
            normal_buf: vertex("portal-mesh-normals", bytemuck::cast_slice(normals)),
            tangent_buf: vertex("portal-mesh-tangents", bytemuck::cast_slice(tangents)),
            uv_buf: vertex("portal-mesh-uv0", bytemuck::cast_slice(uvs)),
            index_buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portal-mesh-indices"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }),
            vertex_count,
            _reservation: reservation,
        })
    }
}
