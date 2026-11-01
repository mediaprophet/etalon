"""Qualify an independent Git consumer; no workspace patches or hardware claims."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib

TOOLCHAIN = "1.96.1"
PACKAGES = {
    "etalon": ["std", "wgsl"],
    "etalon-nquin": ["std"],
    "etalon-tensor": ["std"],
    "etalon-wgsl-forge": ["std", "wgsl", "naga-validate"],
    "etalon-render": [],
    "etalon-wasm": ["std"],
    "etalon-agent": ["std"],
    "etalon-sense": ["camera"],
}
SOURCE = r'''
use etalon_sense::signal::vision::{mean_u8_region, ImageRegion};

pub async fn acquire_adapter(instance: &etalon::Instance) -> Result<etalon::Adapter, etalon::RequestAdapterError> {
    instance.request_adapter(&etalon::RequestAdapterOptions::default()).await
}

pub fn extension_types() -> [usize; 7] {
    [
        core::mem::size_of::<etalon_nquin::NQuinWire>(),
        core::mem::size_of::<etalon_tensor::AttentionParams>(),
        core::mem::size_of::<etalon_wgsl_forge::WorkgroupSize>(),
        core::mem::size_of::<etalon_render::FrameCameraPod>(),
        core::mem::size_of::<etalon_wasm::MessageHeader>(),
        core::mem::size_of::<etalon_agent::QueueId>(),
        core::mem::size_of::<etalon_sense::camera::FrameMetadata>(),
    ]
}

#[no_mangle]
pub extern "C" fn consumer_portable_probe() -> u32 {
    let region = ImageRegion { x: 0, y: 0, width: 2, height: 2 };
    let mean = mean_u8_region(&[10, 20, 30, 40], 2, 2, 2, region).unwrap();
    assert_eq!(mean, 25.0);
    assert_eq!(extension_types()[0], 48);
    let stamp = etalon_sense::core::time::ClockStamp::new(1, 1, 100, 5).unwrap();
    assert_eq!(stamp.ticks_ns(), 100);
    etalon_wgsl_forge::validate_wgsl("@compute @workgroup_size(1) fn main() {}").unwrap();
    mean as u32
}

#[cfg(test)]
mod tests {
    #[test]
    fn public_portable_apis_work_from_git_dependencies() {
        assert_eq!(super::consumer_portable_probe(), 25);
        assert!(super::extension_types().iter().all(|size| *size > 0));
    }
}
'''


def manifest(repository, revision):
    if not re.fullmatch(r"https://[A-Za-z0-9.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", repository):
        raise ValueError("repository must be an HTTPS Git repository URL")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be the full lowercase 40-digit candidate SHA")
    lines = [
        '[package]', 'name = "etalon-external-consumer"', 'version = "0.0.0"',
        'edition = "2021"', 'publish = false', '', '[workspace]', 'resolver = "3"',
        '', '[lib]', 'crate-type = ["rlib", "cdylib"]', '', '[features]',
        'native = ["etalon/vulkan"]', 'browser = ["etalon/webgpu", "etalon/webgl"]',
        '', '[dependencies]',
    ]
    for package, features in PACKAGES.items():
        lines.append(f'{package} = {{ git = {json.dumps(repository)}, rev = {json.dumps(revision)}, default-features = false, features = {json.dumps(features)} }}')
    result = "\n".join(lines) + "\n"
    parsed = tomllib.loads(result)
    assert "patch" not in parsed
    assert all("path" not in dep for dep in parsed["dependencies"].values())
    return result


def verify_metadata(metadata, project, repository, revision):
    if Path(metadata["workspace_root"]).resolve() != project.resolve():
        raise ValueError("consumer unexpectedly inherited another workspace")
    found = set()
    for package in metadata["packages"]:
        name = package["name"]
        if name.startswith("etalon-") or name == "etalon" or name in ("naga", "naga-types"):
            if name == "etalon-external-consumer":
                continue
            source = package.get("source") or ""
            if not source.startswith(f"git+{repository}?") or source.rsplit("#", 1)[-1] != revision:
                raise ValueError(f"{name} resolved outside the candidate Git revision: {source}")
            if package["version"] != "0.0.2":
                raise ValueError(f"{name} has unexpected release version {package['version']}")
            found.add(name)
    if not set(PACKAGES).issubset(found):
        raise ValueError("consumer metadata is missing requested public packages")
    return sorted(found)


def run(project, *args, timeout, output=None):
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(project / "target")
    command = ["cargo", f"+{TOOLCHAIN}", *args]
    print("Consumer: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=project, env=env, check=True, timeout=timeout, stdout=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="validate the generated manifest without fetching or building")
    args = parser.parse_args()
    repository = os.environ.get("ETALON_CONSUMER_REPOSITORY", "")
    revision = os.environ.get("ETALON_CONSUMER_REVISION", "")
    generated = manifest(repository, revision)
    if args.validate_only:
        print("Independent consumer manifest validated; no Git fetch or build performed.")
        return
    with tempfile.TemporaryDirectory(prefix="etalon-git-consumer-") as scratch:
        project = Path(scratch).resolve()
        checkout = Path(__file__).resolve().parents[2]
        if project == checkout or checkout in project.parents:
            raise ValueError("consumer scratch must be outside the repository workspace")
        (project / "src").mkdir()
        (project / "Cargo.toml").write_text(generated, encoding="utf-8")
        (project / "src/lib.rs").write_text(SOURCE, encoding="utf-8")
        run(project, "generate-lockfile", timeout=240)
        with tempfile.TemporaryFile() as output:
            run(project, "metadata", "--locked", "--all-features", "--format-version", "1", timeout=120, output=output)
            if output.tell() > 8 * 1024 * 1024:
                raise ValueError("consumer metadata exceeds the 8 MiB inspection limit")
            output.seek(0)
            packages = verify_metadata(json.load(output), project, repository, revision)
        run(project, "test", "--locked", "--lib", "--features", "native", timeout=600)
        run(project, "build", "--locked", "--lib", "--target", "wasm32-unknown-unknown", "--features", "browser", timeout=600)
        artifact = project / "target/wasm32-unknown-unknown/debug/etalon_external_consumer.wasm"
        if not artifact.is_file() or artifact.stat().st_size < 8:
            raise ValueError("consumer WASM link artifact is missing")
        with artifact.open("rb") as output:
            if output.read(8) != b"\x00asm\x01\x00\x00\x00":
                raise ValueError("consumer WASM artifact has an invalid header")
        print(json.dumps({"revision": revision, "packages": packages, "native": "Vulkan feature build and portable CPU assertions passed", "wasm": "WebGPU/WebGL feature cdylib linked", "gpu_execution_tested": False}))


if __name__ == "__main__":
    main()
