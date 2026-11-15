use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::ExitCode;

use etalon_reconstruct::{
    mesh_to_glb, mesh_to_obj, reconstruct_directory_with_volume, reconstruct_paths_with_volume,
    volume_to_occ1, volume_to_q42sdf16, ReconstructConfig, ViewKind,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let mut pargs = pico_args::Arguments::from_env();
    if pargs.contains(["-h", "--help"]) {
        println!(
            "etalon-reconstruct — local image-to-mesh (visual hull + surface nets)\n\
             \n\
             USAGE:\n\
               etalon-reconstruct --dir DIR --out mesh.obj [--glb mesh.glb] [--resolution N]\n\
               etalon-reconstruct --anterior a.jpg --left l.jpg --superior s.jpg --out mesh.obj\n\
             \n\
             Views: --anterior --posterior --left --right --superior --inferior --three-quarter\n\
             Default resolution 96. Coordinate frame: Y-up, +Z anterior, +X patient's left.\n\
             Writes volumetric sidecars MESH.occ (OCC1 tissue lattice) and MESH.sdf (Q42SDF16).\n\
             Seal QualiaDB .10d/.m64/.emf with: cargo run -p qualia-seal-10d -- mesh.obj mesh.10d --occupancy mesh.occ --m64 mesh.m64 --emf mesh.emf"
        );
        return Ok(());
    }
    let resolution: u32 = pargs
        .opt_value_from_str("--resolution")
        .map_err(|e| e.to_string())?
        .unwrap_or(96);
    let out: PathBuf = pargs
        .opt_value_from_os_str("--out", os_path)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "missing --out PATH".to_string())?;
    let obj_out: Option<PathBuf> = pargs
        .opt_value_from_os_str("--obj", os_path)
        .map_err(|e| e.to_string())?;
    let glb_out: Option<PathBuf> = pargs
        .opt_value_from_os_str("--glb", os_path)
        .map_err(|e| e.to_string())?;
    let dir: Option<PathBuf> = pargs
        .opt_value_from_os_str("--dir", os_path)
        .map_err(|e| e.to_string())?;
    let named = collect_named(&mut pargs)?;
    let rest = pargs.finish();
    if !rest.is_empty() && dir.is_none() && named.is_empty() {
        let maybe_dir = PathBuf::from(&rest[0]);
        if maybe_dir.is_dir() {
            return reconstruct_and_write(maybe_dir, named, resolution, out, obj_out, glb_out);
        }
    }
    reconstruct_and_write_from(dir, named, resolution, out, obj_out, glb_out)
}

fn reconstruct_and_write_from(
    dir: Option<PathBuf>,
    named: Vec<(ViewKind, PathBuf)>,
    resolution: u32,
    out: PathBuf,
    obj_out: Option<PathBuf>,
    glb_out: Option<PathBuf>,
) -> Result<(), String> {
    if dir.is_none() && named.is_empty() {
        return Err("supply --dir or named views (--anterior, --left, ...)".into());
    }
    reconstruct_and_write(
        dir.unwrap_or_default(),
        named,
        resolution,
        out,
        obj_out,
        glb_out,
    )
}

fn reconstruct_and_write(
    dir: PathBuf,
    named: Vec<(ViewKind, PathBuf)>,
    resolution: u32,
    out: PathBuf,
    obj_out: Option<PathBuf>,
    glb_out: Option<PathBuf>,
) -> Result<(), String> {
    let cfg = ReconstructConfig {
        resolution,
        ..ReconstructConfig::default()
    };
    let result = if !named.is_empty() {
        reconstruct_paths_with_volume(&named, &cfg).map_err(|e| e.to_string())?
    } else {
        reconstruct_directory_with_volume(&dir, &cfg).map_err(|e| e.to_string())?
    };
    let mesh = &result.mesh;
    let stem = out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("reconstructed");
    let ext = out
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("obj")
        .to_ascii_lowercase();
    match ext.as_str() {
        "glb" => {
            std::fs::write(&out, mesh_to_glb(&mesh, stem)).map_err(|e| e.to_string())?;
        }
        "10d" => {
            return Err(
                "reconstruct does not write .10d (that would fork the QualiaDB container). \
                 Write an OBJ, then seal: cargo run -p qualia-seal-10d -- mesh.obj mesh.10d --occupancy mesh.occ --m64 mesh.m64 --emf mesh.emf"
                    .into(),
            );
        }
        _ => {
            std::fs::write(&out, mesh_to_obj(mesh, stem)).map_err(|e| e.to_string())?;
        }
    }
    if let Some(obj_path) = obj_out {
        std::fs::write(&obj_path, mesh_to_obj(mesh, stem)).map_err(|e| e.to_string())?;
    }
    if let Some(glb_path) = glb_out {
        std::fs::write(&glb_path, mesh_to_glb(mesh, stem)).map_err(|e| e.to_string())?;
    }
    let occ_path = out.with_extension("occ");
    let sdf_path = out.with_extension("sdf");
    std::fs::write(&occ_path, volume_to_occ1(&result.volume)).map_err(|e| e.to_string())?;
    std::fs::write(&sdf_path, volume_to_q42sdf16(&result.volume)).map_err(|e| e.to_string())?;
    eprintln!(
        "wrote {} ({} verts, {} tris); occupancy {} ({} occupied / {} voxels); sdf {}",
        out.display(),
        mesh.positions.len(),
        mesh.triangle_count(),
        occ_path.display(),
        result.volume.occupied_count(),
        result.volume.voxel_count(),
        sdf_path.display()
    );
    Ok(())
}

fn collect_named(pargs: &mut pico_args::Arguments) -> Result<Vec<(ViewKind, PathBuf)>, String> {
    let pairs = [
        ("--anterior", ViewKind::Anterior),
        ("--posterior", ViewKind::Posterior),
        ("--left", ViewKind::LeftLateral),
        ("--right", ViewKind::RightLateral),
        ("--superior", ViewKind::Superior),
        ("--inferior", ViewKind::Inferior),
        ("--three-quarter", ViewKind::ThreeQuarter),
    ];
    let mut out = Vec::new();
    for (flag, kind) in pairs {
        if let Some(path) = pargs
            .opt_value_from_os_str(flag, os_path)
            .map_err(|e| e.to_string())?
        {
            out.push((kind, path));
        }
    }
    Ok(out)
}

fn os_path(s: &OsStr) -> Result<PathBuf, &'static str> {
    Ok(PathBuf::from(s))
}
