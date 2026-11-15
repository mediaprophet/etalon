# etalon-reconstruct

Local image-to-mesh for Etalon. Use this instead of the hosted `grok-imagine-3d` `/v1/3d` job when Zero Data Retention blocks deferred endpoints.

Pipeline: silhouette (border flood of dark / low-chroma studio paper, then largest connected component) → visual hull (multi-view) or relief inflation (one view) → naive surface nets → vertex colour from the best-facing still → Wavefront OBJ. QualiaDB `.10d` is sealed by `qualia-seal-10d`. Reconstruct does not write the container.

Surface nets follows the MIT algorithm in [fast-surface-nets-rs](https://github.com/bonsairobo/fast-surface-nets-rs). Coordinate frame: Y-up, +Z anterior, +X patient's left.

Visual hull quality scales with the number of calibrated stills. Generate a camera-rig of unlabeled exterior views (anterior, posterior, left, right, superior, inferior, three-quarter) with a locked specimen, solid studio background, and one image per view — see `docs/camera-rig.md`. Cutaways and labels are inspection stills, not hull input.

```text
cargo run -p etalon-reconstruct --release -- \
  --dir C:\github\anatomy\working\heart\refs\recon \
  --out C:\github\anatomy\working\heart\master\heart-exterior.obj \
  --resolution 96

cargo run -p qualia-seal-10d --release --manifest-path C:\github\qualiaDB\Cargo.toml -- \
  C:\github\anatomy\working\heart\master\heart-exterior.obj \
  C:\github\anatomy\working\heart\master\heart-exterior.10d
```
