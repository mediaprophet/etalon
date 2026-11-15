use crate::camera::{OrthoCamera, ViewKind};
use crate::error::ReconstructError;
use crate::mesh::TriangleMesh;
use crate::raster::{Raster, Silhouette};
use crate::surface::surface_nets;
use crate::{Aabb, ReconstructConfig};

/// Empty / outside the visual hull.
pub const MAT_EMPTY: u8 = 0;
/// Myocardial wall (near-surface occupied shell).
pub const MAT_MYOCARDIUM: u8 = 1;
/// Chamber lumen (deep interior of the hull).
pub const MAT_LUMEN: u8 = 2;
/// Epicardial fat (outermost occupied band).
pub const MAT_EPICARDIAL_FAT: u8 = 3;

/// Volumetric master of a reconstruction: occupancy, SDF, and tissue labels.
#[derive(Clone, Debug)]
pub struct OccupancyVolume {
    pub resolution: u32,
    pub bounds: Aabb,
    pub sdf: Vec<f32>,
    pub material: Vec<u8>,
}

impl OccupancyVolume {
    pub fn voxel_count(&self) -> usize {
        let n = self.resolution as usize;
        n * n * n
    }

    pub fn occupied_count(&self) -> usize {
        self.material.iter().filter(|&&m| m != MAT_EMPTY).count()
    }
}

#[derive(Clone, Debug)]
pub struct ReconstructResult {
    pub mesh: TriangleMesh,
    pub volume: OccupancyVolume,
}

#[derive(Debug)]
pub struct SourceView {
    pub kind: ViewKind,
    pub raster: Raster,
    pub camera: OrthoCamera,
    pub silhouette: Silhouette,
}

impl SourceView {
    pub fn new(
        kind: ViewKind,
        raster: Raster,
        bounds: Aabb,
        luma_threshold: u8,
        camera: Option<OrthoCamera>,
    ) -> Result<Self, ReconstructError> {
        let silhouette = Silhouette::from_raster(&raster, luma_threshold);
        if silhouette.foreground_count() == 0 {
            return Err(ReconstructError::EmptySilhouette {
                view: kind.as_str().into(),
            });
        }
        Ok(Self {
            kind,
            raster,
            camera: camera.unwrap_or_else(|| OrthoCamera::for_view(kind, bounds)),
            silhouette,
        })
    }
}

pub fn carve_occupancy(
    views: &[SourceView],
    cfg: &ReconstructConfig,
) -> Result<OccupancyVolume, ReconstructError> {
    let n = cfg.resolution;
    if !(8..=256).contains(&n) {
        return Err(ReconstructError::InvalidResolution);
    }
    let n_us = n as usize;
    let mut occ = vec![1u8; n_us * n_us * n_us];
    let bounds = cfg.bounds;
    let size = bounds.size();
    let single = views.len() == 1;
    let world_per_px =
        |view: &SourceView| 2.0 * view.camera.half_height / view.silhouette.height.max(1) as f32;

    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let i = index(x, y, z, n);
                if x == 0 || y == 0 || z == 0 || x + 1 == n || y + 1 == n || z + 1 == n {
                    occ[i] = 0;
                    continue;
                }
                let p = [
                    bounds.min[0] + (x as f32 + 0.5) / n as f32 * size[0],
                    bounds.min[1] + (y as f32 + 0.5) / n as f32 * size[1],
                    bounds.min[2] + (z as f32 + 0.5) / n as f32 * size[2],
                ];
                let mut inside = true;
                for view in views {
                    let (u, v, depth) = view.camera.project(p);
                    match view.silhouette.sample(u, v) {
                        Some((true, dist_px)) => {
                            if single || cfg.force_relief {
                                let half = dist_px * world_per_px(view) * cfg.relief_scale;
                                if depth.abs() > half {
                                    inside = false;
                                    break;
                                }
                            }
                        }
                        _ => {
                            inside = false;
                            break;
                        }
                    }
                }
                occ[i] = if inside { 1 } else { 0 };
            }
        }
    }

    let sdf = occupancy_to_sdf(&occ, n);
    let material = classify_tissues(&occ, &sdf, n);
    Ok(OccupancyVolume {
        resolution: n,
        bounds,
        sdf,
        material,
    })
}

pub fn mesh_from_views(
    views: &[SourceView],
    cfg: &ReconstructConfig,
) -> Result<TriangleMesh, ReconstructError> {
    Ok(reconstruct_volume(views, cfg)?.mesh)
}

pub fn reconstruct_volume(
    views: &[SourceView],
    cfg: &ReconstructConfig,
) -> Result<ReconstructResult, ReconstructError> {
    if views.is_empty() {
        return Err(ReconstructError::EmptyViews);
    }
    let volume = carve_occupancy(views, cfg)?;
    let mut mesh = surface_nets(&volume.sdf, volume.resolution, cfg.bounds)?;
    paint_vertices(&mut mesh, views);
    Ok(ReconstructResult { mesh, volume })
}

fn classify_tissues(occ: &[u8], sdf: &[f32], n: u32) -> Vec<u8> {
    let wall = ((0.08 * n as f32).clamp(1.5, 12.0)) as f32;
    let fat = (wall * 0.22).max(1.0);
    let mut material = vec![MAT_EMPTY; occ.len()];
    for i in 0..occ.len() {
        if occ[i] == 0 {
            continue;
        }
        let depth = (-sdf[i]).max(0.0);
        material[i] = if depth <= fat {
            MAT_EPICARDIAL_FAT
        } else if depth <= wall {
            MAT_MYOCARDIUM
        } else {
            MAT_LUMEN
        };
    }
    material
}

fn occupancy_to_sdf(occ: &[u8], n: u32) -> Vec<f32> {
    let n_us = n as usize;
    let mut sdf = vec![0.0f32; occ.len()];
    for (i, &o) in occ.iter().enumerate() {
        sdf[i] = if o != 0 { -0.5 } else { 0.5 };
    }
    let inf = 1.0e6f32;
    let mut dist = vec![inf; occ.len()];
    for i in 0..occ.len() {
        dist[i] = if occ[i] != 0 { 0.0 } else { inf };
    }
    chamfer3(&mut dist, n_us);
    for z in 0..n_us {
        for y in 0..n_us {
            for x in 0..n_us {
                let i = x + n_us * (y + n_us * z);
                if occ[i] == 0 {
                    sdf[i] = dist[i].max(0.5);
                }
            }
        }
    }
    for i in 0..occ.len() {
        dist[i] = if occ[i] == 0 { 0.0 } else { inf };
    }
    chamfer3(&mut dist, n_us);
    for z in 0..n_us {
        for y in 0..n_us {
            for x in 0..n_us {
                let i = x + n_us * (y + n_us * z);
                if occ[i] != 0 {
                    sdf[i] = -dist[i].max(0.5);
                }
            }
        }
    }
    sdf
}

fn chamfer3(d: &mut [f32], n: usize) {
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let i = x + n * (y + n * z);
                let mut best = d[i];
                if x > 0 {
                    best = best.min(d[i - 1] + 1.0);
                }
                if y > 0 {
                    best = best.min(d[i - n] + 1.0);
                }
                if z > 0 {
                    best = best.min(d[i - n * n] + 1.0);
                }
                d[i] = best;
            }
        }
    }
    for z in (0..n).rev() {
        for y in (0..n).rev() {
            for x in (0..n).rev() {
                let i = x + n * (y + n * z);
                let mut best = d[i];
                if x + 1 < n {
                    best = best.min(d[i + 1] + 1.0);
                }
                if y + 1 < n {
                    best = best.min(d[i + n] + 1.0);
                }
                if z + 1 < n {
                    best = best.min(d[i + n * n] + 1.0);
                }
                d[i] = best;
            }
        }
    }
}

fn index(x: u32, y: u32, z: u32, n: u32) -> usize {
    (x + n * (y + n * z)) as usize
}

fn paint_vertices(mesh: &mut TriangleMesh, views: &[SourceView]) {
    for i in 0..mesh.positions.len() {
        let p = mesh.positions[i];
        let nrm = mesh.normals[i];
        let mut best_dot = f32::NEG_INFINITY;
        let mut color = [180u8, 70, 70, 255];
        for view in views {
            let facing = crate::camera::dot(nrm, crate::camera::scale(view.camera.forward, -1.0));
            let (u, v, _) = view.camera.project(p);
            if let Some(px) = view.raster.sample_bilinear(u, v) {
                if facing > best_dot {
                    best_dot = facing;
                    color = px;
                }
            }
        }
        mesh.colors[i] = color;
    }
}
