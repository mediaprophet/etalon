use crate::DeviceBenchmarkResponse;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

pub const RESPONSE_BYTES: usize = 1024 * 1024;

pub struct WorkerArtifacts {
    directory: tempfile::TempDir,
}

impl WorkerArtifacts {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            directory: tempfile::Builder::new()
                .prefix("etalon-device-bench-")
                .tempdir()
                .map_err(|e| format!("create worker directory: {e}"))?,
        })
    }

    pub fn response_path(&self) -> PathBuf {
        self.directory.path().join("response.cbor")
    }

    pub fn check_budget(&self) -> Result<(), String> {
        for entry in std::fs::read_dir(self.directory.path()).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let meta = entry.path().symlink_metadata().map_err(|e| e.to_string())?;
            if entry.file_name() != "response.cbor"
                || !meta.is_file()
                || meta.len() > (RESPONSE_BYTES + 8) as u64
            {
                return Err("invalid worker artifact or byte budget exceeded".into());
            }
        }
        Ok(())
    }
}

struct BoundedPayload(Vec<u8>);

impl Write for BoundedPayload {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > RESPONSE_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other(
                "worker response byte budget exceeded",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn encode_response(path: &Path, response: &DeviceBenchmarkResponse) -> Result<(), String> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(RESPONSE_BYTES)
        .map_err(|e| format!("response byte budget: {e}"))?;
    let mut payload = BoundedPayload(bytes);
    ciborium::into_writer(response, &mut payload).map_err(|e| format!("encode response: {e}"))?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("create response: {e}"))?;
    file.write_all(&(payload.0.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&payload.0).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}

pub fn decode_response(path: &Path) -> Result<DeviceBenchmarkResponse, String> {
    let meta = path
        .symlink_metadata()
        .map_err(|e| format!("response metadata: {e}"))?;
    if !meta.is_file() || meta.len() > (RESPONSE_BYTES + 8) as u64 {
        return Err("invalid response file or byte budget exceeded".into());
    }
    let mut file = std::fs::File::open(path).map_err(|e| format!("open response: {e}"))?;
    let mut length = [0; 8];
    file.read_exact(&mut length)
        .map_err(|e| format!("read response length: {e}"))?;
    let length = u64::from_le_bytes(length);
    if length == 0 || length > RESPONSE_BYTES as u64 || meta.len() != length + 8 {
        return Err(format!("invalid worker response length {length}"));
    }
    let mut payload = vec![0; length as usize];
    file.read_exact(&mut payload)
        .map_err(|e| format!("read response: {e}"))?;
    let mut cursor = Cursor::new(payload.as_slice());
    let response = ciborium::de::from_reader_with_recursion_limit(&mut cursor, 32)
        .map_err(|e| format!("decode response: {e}"))?;
    if cursor.position() != length {
        return Err("trailing bytes in worker response".into());
    }
    Ok(response)
}

pub fn decode_matching_response(
    output: &Path,
    request: &crate::DeviceBenchmarkRequest,
) -> Result<crate::CircuitBench, String> {
    let decoded = decode_response(output);
    let response = decoded?;
    if response.request.backend != request.backend
        || response.request.vendor != request.vendor
        || response.request.device != request.device
        || response.request.gemv_n != request.gemv_n
    {
        return Err("worker response identity mismatch".into());
    }
    if let Some(error) = response.error {
        return Err(error);
    }
    let bench = response
        .bench
        .ok_or("worker returned neither result nor error")?;
    if bench.kind == crate::CircuitKind::Cpu
        || bench.backend.to_ascii_lowercase() != request.backend
        || !bench.ms_per_gemv.is_finite()
        || bench.ms_per_gemv <= 0.0
        || !bench.gflops.is_finite()
        || bench.gflops < 0.0
        || !bench.upload_gbps.is_finite()
        || bench.upload_gbps < 0.0
    {
        return Err("worker returned invalid metrics".into());
    }
    Ok(bench)
}
