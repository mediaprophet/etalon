//! Local image-to-mesh reconstruction for Etalon.
//!
//! Replaces the hosted `grok-imagine-3d` deferred `/v1/3d` job (blocked under Zero Data
//! Retention) with a fully local pipeline:
//! 1. silhouette extraction from one or more stills
//! 2. multi-view visual hull, or single-view relief inflation
//! 3. occupancy/SDF volume kept as the master (myocardium / lumen / epicardial fat)
//! 4. naive surface nets (algorithm after [fast-surface-nets-rs](https://github.com/bonsairobo/fast-surface-nets-rs), MIT)
//! 5. vertex colour from the best-facing view
//! 6. OBJ (lossy surface) plus `Q42SDF16` / `OCC1` volumetric sidecars. QualiaDB `.10d`
//!    `.m64`, and `.emf` (`Q42EMF64`) are sealed afterwards by `qualia-seal-10d`.
//!    Reconstruct does not write those containers.
//!
//! Coordinate convention matches the anatomy atlas: Y-up, +Z anterior, +X patient's left.

mod camera;
mod error;
mod export;
mod mesh;
mod occupancy;
mod raster;
mod surface;

pub use camera::{Aabb, OrthoCamera, ViewKind};
pub use error::ReconstructError;
pub use export::{mesh_to_glb, mesh_to_obj, occ1_from_bytes, volume_to_occ1, volume_to_q42sdf16};
pub use mesh::TriangleMesh;
pub use occupancy::{
    mesh_from_views, reconstruct_volume, OccupancyVolume, ReconstructResult, SourceView, MAT_EMPTY,
    MAT_EPICARDIAL_FAT, MAT_LUMEN, MAT_MYOCARDIUM,
};
pub use raster::{Raster, Silhouette};

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct ReconstructConfig {
    pub resolution: u32,
    pub luma_threshold: u8,
    pub bounds: Aabb,
    pub relief_scale: f32,
    pub force_relief: bool,
}

impl Default for ReconstructConfig {
    fn default() -> Self {
        Self {
            resolution: 96,
            luma_threshold: 18,
            bounds: Aabb::heart_cm(),
            relief_scale: 1.0,
            force_relief: false,
        }
    }
}

pub fn reconstruct(
    views: &[SourceView],
    cfg: &ReconstructConfig,
) -> Result<TriangleMesh, ReconstructError> {
    occupancy::mesh_from_views(views, cfg)
}

pub fn reconstruct_with_volume(
    views: &[SourceView],
    cfg: &ReconstructConfig,
) -> Result<ReconstructResult, ReconstructError> {
    occupancy::reconstruct_volume(views, cfg)
}

pub fn reconstruct_paths(
    files: &[(ViewKind, PathBuf)],
    cfg: &ReconstructConfig,
) -> Result<TriangleMesh, ReconstructError> {
    let mut views = Vec::with_capacity(files.len());
    for (kind, path) in files {
        let raster = Raster::from_path(path)?;
        views.push(SourceView::new(
            *kind,
            raster,
            cfg.bounds,
            cfg.luma_threshold,
            None,
        )?);
    }
    reconstruct(&views, cfg)
}

pub fn reconstruct_paths_with_volume(
    files: &[(ViewKind, PathBuf)],
    cfg: &ReconstructConfig,
) -> Result<ReconstructResult, ReconstructError> {
    let mut views = Vec::with_capacity(files.len());
    for (kind, path) in files {
        let raster = Raster::from_path(path)?;
        views.push(SourceView::new(
            *kind,
            raster,
            cfg.bounds,
            cfg.luma_threshold,
            None,
        )?);
    }
    reconstruct_with_volume(&views, cfg)
}

pub fn reconstruct_directory(
    dir: &Path,
    cfg: &ReconstructConfig,
) -> Result<TriangleMesh, ReconstructError> {
    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| ReconstructError::Io(e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| ReconstructError::Io(e.to_string()))?;
        let path = entry.path();
        if let Some(kind) = ViewKind::from_path(&path) {
            files.push((kind, path));
        }
    }
    files.sort_by(|(ka, pa), (kb, pb)| {
        kind_priority(*ka)
            .cmp(&kind_priority(*kb))
            .then_with(|| prefer_generated(pb).cmp(&prefer_generated(pa)))
    });
    let mut seen = [false; 8];
    files.retain(|(kind, _)| {
        let i = kind_priority(*kind) as usize;
        if seen[i] {
            false
        } else {
            seen[i] = true;
            true
        }
    });
    reconstruct_paths(&files, cfg)
}

pub fn reconstruct_directory_with_volume(
    dir: &Path,
    cfg: &ReconstructConfig,
) -> Result<ReconstructResult, ReconstructError> {
    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| ReconstructError::Io(e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| ReconstructError::Io(e.to_string()))?;
        let path = entry.path();
        if let Some(kind) = ViewKind::from_path(&path) {
            files.push((kind, path));
        }
    }
    files.sort_by(|(ka, pa), (kb, pb)| {
        kind_priority(*ka)
            .cmp(&kind_priority(*kb))
            .then_with(|| prefer_generated(pb).cmp(&prefer_generated(pa)))
    });
    let mut seen = [false; 8];
    files.retain(|(kind, _)| {
        let i = kind_priority(*kind) as usize;
        if seen[i] {
            false
        } else {
            seen[i] = true;
            true
        }
    });
    reconstruct_paths_with_volume(&files, cfg)
}

fn prefer_generated(path: &Path) -> u8 {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|s| u8::from(s.starts_with("gen-")))
        .unwrap_or(0)
}

fn kind_priority(kind: ViewKind) -> u8 {
    match kind {
        ViewKind::Anterior => 0,
        ViewKind::ThreeQuarter => 1,
        ViewKind::LeftLateral => 2,
        ViewKind::RightLateral => 3,
        ViewKind::Posterior => 4,
        ViewKind::Superior => 5,
        ViewKind::Inferior => 6,
        ViewKind::Custom => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled_circle(w: u32, h: u32, cx: f32, cy: f32, r: f32) -> Raster {
        filled_circle_on(w, h, cx, cy, r, [0, 0, 0, 0])
    }

    fn filled_circle_on(w: u32, h: u32, cx: f32, cy: f32, r: f32, paper: [u8; 4]) -> Raster {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            let o = i * 4;
            rgba[o] = paper[0];
            rgba[o + 1] = paper[1];
            rgba[o + 2] = paper[2];
            rgba[o + 3] = paper[3];
        }
        let r2 = r * r;
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx * dx + dy * dy <= r2 {
                    let i = ((y * w + x) * 4) as usize;
                    rgba[i] = 180;
                    rgba[i + 1] = 40;
                    rgba[i + 2] = 40;
                    rgba[i + 3] = 255;
                }
            }
        }
        Raster::from_rgba(w, h, rgba).unwrap()
    }

    fn circle_views() -> Vec<SourceView> {
        let raster = filled_circle(64, 64, 32.0, 32.0, 22.0);
        let cfg = ReconstructConfig {
            resolution: 28,
            bounds: Aabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            ..ReconstructConfig::default()
        };
        let kinds = [
            ViewKind::Anterior,
            ViewKind::LeftLateral,
            ViewKind::Superior,
        ];
        kinds
            .into_iter()
            .map(|kind| {
                SourceView::new(kind, raster.clone(), cfg.bounds, cfg.luma_threshold, None).unwrap()
            })
            .collect()
    }

    #[test]
    fn three_ortho_circles_make_a_closed_blob() {
        let cfg = ReconstructConfig {
            resolution: 28,
            bounds: Aabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            ..ReconstructConfig::default()
        };
        let result = reconstruct_with_volume(&circle_views(), &cfg).unwrap();
        let mesh = result.mesh;
        assert!(mesh.positions.len() > 40);
        assert!(mesh.triangle_count() > 40);
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        let (min, max) = mesh.bounds();
        for i in 0..3 {
            assert!(min[i] < -0.2, "axis {i} min {min:?}");
            assert!(max[i] > 0.2, "axis {i} max {max:?}");
        }
        let mut outward = 0i32;
        for (p, n) in mesh.positions.iter().zip(mesh.normals.iter()) {
            if crate::camera::dot(*p, *n) > 0.0 {
                outward += 1;
            } else {
                outward -= 1;
            }
        }
        assert!(outward > 0, "majority of normals should point outward");
        assert!(result.volume.occupied_count() > 40);
        assert!(result.volume.material.iter().any(|&m| m == MAT_MYOCARDIUM));
        assert!(result.volume.material.iter().any(|&m| m == MAT_LUMEN));
        let sdf16 = volume_to_q42sdf16(&result.volume);
        assert_eq!(&sdf16[0..8], b"Q42SDF16");
        let occ = volume_to_occ1(&result.volume);
        assert_eq!(&occ[0..4], b"OCC1");
        let parsed = occ1_from_bytes(&occ).unwrap();
        assert_eq!(parsed.resolution, result.volume.resolution);
        assert_eq!(parsed.material, result.volume.material);
    }

    #[test]
    fn empty_views_fail() {
        let err = reconstruct(&[], &ReconstructConfig::default()).unwrap_err();
        assert!(matches!(err, ReconstructError::EmptyViews));
    }

    #[test]
    fn grey_studio_backdrop_does_not_fill_the_frame() {
        let raster = filled_circle_on(64, 64, 32.0, 32.0, 18.0, [48, 48, 52, 255]);
        let silhouette = Silhouette::from_raster(&raster, 18);
        let fg = silhouette.foreground_count();
        let total = 64 * 64;
        assert!(fg > 200, "circle should remain, got {fg}");
        assert!(
            fg < total / 2,
            "grey paper must not be foreground, got {fg}/{total}"
        );
        assert_eq!(silhouette.sample(0.02, 0.02), Some((false, 0.0)));
        let (inside, dist) = silhouette.sample(0.5, 0.5).unwrap();
        assert!(inside);
        assert!(dist > 4.0);
    }

    #[test]
    fn empty_silhouette_fails() {
        let raster = Raster::from_rgba(8, 8, vec![0u8; 8 * 8 * 4]).unwrap();
        let err =
            SourceView::new(ViewKind::Anterior, raster, Aabb::heart_cm(), 18, None).unwrap_err();
        assert!(matches!(err, ReconstructError::EmptySilhouette { .. }));
    }

    #[test]
    fn obj_and_glb_round_trip_headers() {
        let cfg = ReconstructConfig {
            resolution: 28,
            bounds: Aabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            ..ReconstructConfig::default()
        };
        let mesh = reconstruct(&circle_views(), &cfg).unwrap();
        let obj = mesh_to_obj(&mesh, "blob");
        assert!(obj.contains("\nv "));
        assert!(obj.contains("\nf "));
        let glb = mesh_to_glb(&mesh, "blob");
        assert_eq!(&glb[0..4], b"glTF");
    }

    #[test]
    fn single_view_relief_still_meshs() {
        let raster = filled_circle(48, 48, 24.0, 24.0, 16.0);
        let cfg = ReconstructConfig {
            resolution: 24,
            bounds: Aabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            relief_scale: 1.0,
            ..ReconstructConfig::default()
        };
        let view = SourceView::new(
            ViewKind::Anterior,
            raster,
            cfg.bounds,
            cfg.luma_threshold,
            None,
        )
        .unwrap();
        let mesh = reconstruct(&[view], &cfg).unwrap();
        assert!(mesh.triangle_count() > 16);
        let (min, max) = mesh.bounds();
        assert!(max[2] - min[2] > 0.15);
    }

    #[test]
    fn view_kind_from_heart_filenames() {
        assert_eq!(
            ViewKind::parse_name("gen-three-quarter.jpg"),
            Some(ViewKind::ThreeQuarter)
        );
        assert_eq!(
            ViewKind::parse_name("gen-anterior.jpg"),
            Some(ViewKind::Anterior)
        );
        assert_eq!(
            ViewKind::parse_name("gray-492.png"),
            Some(ViewKind::Anterior)
        );
    }
}
