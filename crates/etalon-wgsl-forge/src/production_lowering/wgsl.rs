//! Typed WGSL lowering wrapping existing Forge emit and geometry specs.

use crate::emit::wgsl::emit_wgsl;
use crate::emit::GeneratedShader;
use crate::geometry_specs::tetrahedron::point_in_tetra_wgsl;
use crate::ir::{BufferAccess, BufferElement, BufferSpec, ForgeError, KernelSpec, ScalarType};
use crate::production_ir::{GeometryKind, MathOp, ProductionContract};
use crate::schedule::{AdapterConstraints, WorkgroupSize};

use super::admission::{layout_from_contract, LayoutKind};
use super::{PRODUCTION_LOWERING_SCHEMA_ID, PRODUCTION_LOWERING_SCHEMA_VERSION};

pub const MAX_BINDINGS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmitterKind {
    ExistingForgeWgsl,
    GeometryTetrahedronSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NagaStep {
    ParsedAndValidated {
        entry_points: Vec<String>,
        binding_count: usize,
        source_hash: String,
    },
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BindingDecl {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub access: BufferAccess,
    pub wgsl_type: String,
}

impl Default for BindingDecl {
    fn default() -> Self {
        Self {
            group: 0,
            binding: 0,
            name: String::new(),
            access: BufferAccess::StorageRead,
            wgsl_type: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoweredWgsl {
    pub schema_version: u32,
    pub emitter: EmitterKind,
    pub layout: LayoutKind,
    pub workgroup: WorkgroupSize,
    pub binding_count: usize,
    pub naga: NagaStep,
    pub shader: GeneratedShader,
}

pub fn lower_contract(
    contract: &ProductionContract,
    constraints: &AdapterConstraints,
    bindings_out: &mut [BindingDecl],
) -> Result<LoweredWgsl, ForgeError> {
    contract.validate_with(constraints)?;
    if bindings_out.len() < contract.kernel.buffers.len() {
        return Err(ForgeError::Validation(
            "caller binding buffer is too small".to_string(),
        ));
    }
    let layout = layout_from_contract(contract);
    let workgroup = WorkgroupSize::new(contract.schedule.effective_workgroup_size(), 1, 1);
    let (emitter, mut shader) = emit_for_contract(contract)?;
    let header = format!(
        "// {PRODUCTION_LOWERING_SCHEMA_ID} schema={PRODUCTION_LOWERING_SCHEMA_VERSION} emitter={}\n// kernel={}@{} arithmetic={:?} layout={:?}\n// workgroup={},{},{}\n",
        match emitter {
            EmitterKind::ExistingForgeWgsl => "existing_forge_wgsl",
            EmitterKind::GeometryTetrahedronSpec => "geometry_tetrahedron_spec",
        },
        contract.kernel.id,
        contract.kernel.semantic_version,
        contract.arithmetic,
        layout,
        workgroup.x,
        workgroup.y,
        workgroup.z,
    );
    shader.source = format!("{header}\n{}", shader.source);
    shader.source_hash = blake3::hash(shader.source.as_bytes()).to_hex().to_string();
    shader.schedule = contract.schedule;

    let mut scanned = [ScannedBinding::default(); MAX_BINDINGS];
    let scanned_count = scan_bindings(&shader.source, &mut scanned)?;
    let parsed_wg = parse_workgroup_size(&shader.source)?;
    if parsed_wg.x != workgroup.x {
        return Err(ForgeError::Validation(format!(
            "type/binding mismatch: emitted workgroup {} does not match schedule {}",
            parsed_wg.x, workgroup.x
        )));
    }
    let binding_count =
        match_contract_bindings(&contract.kernel, &scanned[..scanned_count], bindings_out)?;
    if super::admission::contains_wgsl_f64_type(&shader.source) {
        return Err(ForgeError::Validation(
            "native WGSL f64 is not supported; existing df64 is paired f32".to_string(),
        ));
    }
    let naga = naga_step(&shader.source, &contract.kernel)?;
    Ok(LoweredWgsl {
        schema_version: PRODUCTION_LOWERING_SCHEMA_VERSION,
        emitter,
        layout,
        workgroup,
        binding_count,
        naga,
        shader,
    })
}

fn emit_for_contract(
    contract: &ProductionContract,
) -> Result<(EmitterKind, GeneratedShader), ForgeError> {
    match contract.math {
        MathOp::Geometry {
            kind: GeometryKind::PointInTetrahedron,
            uncertain_code,
            outside_code,
            inside_code,
        } => {
            let source = point_in_tetra_wgsl(
                contract.schedule.effective_workgroup_size(),
                [uncertain_code, outside_code, inside_code],
            );
            Ok((
                EmitterKind::GeometryTetrahedronSpec,
                GeneratedShader {
                    kernel_id: contract.kernel.id.clone(),
                    semantic_hash: contract.kernel.semantic_hash()?,
                    source_hash: blake3::hash(source.as_bytes()).to_hex().to_string(),
                    schedule: contract.schedule,
                    source,
                },
            ))
        }
        _ => Ok((
            EmitterKind::ExistingForgeWgsl,
            emit_wgsl(&contract.kernel, contract.schedule)?,
        )),
    }
}

fn naga_step(source: &str, kernel: &KernelSpec) -> Result<NagaStep, ForgeError> {
    #[cfg(feature = "naga-validate")]
    {
        let report = crate::certify::validate_wgsl(source)?;
        if !report.naga_validated {
            return Ok(NagaStep::Unsupported);
        }
        if !report
            .entry_points
            .iter()
            .any(|entry| entry == &kernel.entry_point)
        {
            return Err(ForgeError::Validation(format!(
                "naga module missing contract entry point {}",
                kernel.entry_point
            )));
        }
        Ok(NagaStep::ParsedAndValidated {
            entry_points: report.entry_points,
            binding_count: report.binding_count,
            source_hash: report.source_hash,
        })
    }
    #[cfg(not(feature = "naga-validate"))]
    {
        let _ = (source, kernel);
        Ok(NagaStep::Unsupported)
    }
}

#[derive(Debug, Clone, Copy)]
struct ScannedBinding {
    group: u32,
    binding: u32,
    access: BufferAccess,
    name: [u8; 32],
    name_len: u8,
    wgsl_type: [u8; 48],
    type_len: u8,
}

impl Default for ScannedBinding {
    fn default() -> Self {
        Self {
            group: 0,
            binding: 0,
            access: BufferAccess::StorageRead,
            name: [0; 32],
            name_len: 0,
            wgsl_type: [0; 48],
            type_len: 0,
        }
    }
}

impl ScannedBinding {
    fn name(&self) -> &str {
        core::str::from_utf8(&self.name[..self.name_len as usize]).unwrap_or("")
    }

    fn wgsl_type(&self) -> &str {
        core::str::from_utf8(&self.wgsl_type[..self.type_len as usize]).unwrap_or("")
    }
}

fn scan_bindings(source: &str, out: &mut [ScannedBinding]) -> Result<usize, ForgeError> {
    let mut cursor = 0usize;
    let mut count = 0usize;
    while let Some(rel) = source[cursor..].find("@group(") {
        let at = cursor + rel;
        if in_line_comment(source, at) {
            cursor = at + 6;
            continue;
        }
        let mut pos = at + "@group(".len();
        let group = parse_u32_at(source, &mut pos)?;
        expect_byte(source, &mut pos, b')')?;
        skip_ws_comments(source, &mut pos);
        if !source[pos..].starts_with("@binding(") {
            cursor = pos;
            continue;
        }
        pos += "@binding(".len();
        let binding = parse_u32_at(source, &mut pos)?;
        expect_byte(source, &mut pos, b')')?;
        skip_ws_comments(source, &mut pos);
        if !source[pos..].starts_with("var") {
            cursor = pos;
            continue;
        }
        pos += 3;
        skip_ws_comments(source, &mut pos);
        let access = if source.as_bytes().get(pos) == Some(&b'<') {
            pos += 1;
            let end = source[pos..].find('>').ok_or_else(|| {
                ForgeError::Emission("unterminated var<> access in WGSL".to_string())
            })?;
            let access_src = source[pos..pos + end].trim();
            pos += end + 1;
            parse_access(access_src)?
        } else {
            BufferAccess::StorageRead
        };
        skip_ws_comments(source, &mut pos);
        let name = parse_ident_at(source, &mut pos)?;
        skip_ws_comments(source, &mut pos);
        expect_byte(source, &mut pos, b':')?;
        skip_ws_comments(source, &mut pos);
        let type_end = source[pos..].find(';').ok_or_else(|| {
            ForgeError::Emission("WGSL binding declaration missing ';'".to_string())
        })?;
        let wgsl_type = source[pos..pos + type_end].trim();
        if count >= out.len() {
            return Err(ForgeError::Validation(
                "emitted binding count exceeds caller scan buffer".to_string(),
            ));
        }
        out[count] = scanned_from(group, binding, access, name, wgsl_type)?;
        count += 1;
        cursor = pos + type_end + 1;
    }
    Ok(count)
}

fn scanned_from(
    group: u32,
    binding: u32,
    access: BufferAccess,
    name: &str,
    wgsl_type: &str,
) -> Result<ScannedBinding, ForgeError> {
    if name.len() > 32 || wgsl_type.len() > 48 {
        return Err(ForgeError::Emission(
            "binding name or type exceeds scan buffer".to_string(),
        ));
    }
    let mut item = ScannedBinding {
        group,
        binding,
        access,
        name_len: name.len() as u8,
        type_len: wgsl_type.len() as u8,
        ..ScannedBinding::default()
    };
    item.name[..name.len()].copy_from_slice(name.as_bytes());
    item.wgsl_type[..wgsl_type.len()].copy_from_slice(wgsl_type.as_bytes());
    Ok(item)
}

fn match_contract_bindings(
    kernel: &KernelSpec,
    scanned: &[ScannedBinding],
    bindings_out: &mut [BindingDecl],
) -> Result<usize, ForgeError> {
    if scanned.len() != kernel.buffers.len() {
        return Err(ForgeError::Validation(format!(
            "type/binding mismatch: emitted {} resources, contract has {}",
            scanned.len(),
            kernel.buffers.len()
        )));
    }
    for (index, spec) in kernel.buffers.iter().enumerate() {
        let found = scanned.iter().find(|item| item.name() == spec.name);
        let Some(found) = found else {
            return Err(ForgeError::Validation(format!(
                "type/binding mismatch: contract buffer {:?} missing from emitted WGSL",
                spec.name
            )));
        };
        if found.group != spec.group || found.binding != spec.binding {
            return Err(ForgeError::Validation(format!(
                "type/binding mismatch: {:?} emitted @group({}) @binding({}) vs contract @group({}) @binding({})",
                spec.name, found.group, found.binding, spec.group, spec.binding
            )));
        }
        if found.access != spec.access {
            return Err(ForgeError::Validation(format!(
                "type/binding mismatch: {:?} access {:?} vs contract {:?}",
                spec.name, found.access, spec.access
            )));
        }
        if !type_compatible(spec, found.wgsl_type()) {
            return Err(ForgeError::Validation(format!(
                "type/binding mismatch: {:?} emitted type {:?} is incompatible with {:?}",
                spec.name,
                found.wgsl_type(),
                spec.element
            )));
        }
        bindings_out[index] = BindingDecl {
            group: found.group,
            binding: found.binding,
            name: spec.name.clone(),
            access: found.access,
            wgsl_type: found.wgsl_type().to_string(),
        };
    }
    Ok(kernel.buffers.len())
}

fn type_compatible(spec: &BufferSpec, wgsl_type: &str) -> bool {
    let t = wgsl_type.trim();
    match (spec.element, spec.access) {
        (BufferElement::Scalar(ScalarType::F32), _) => t == "array<f32>" || t == "f32",
        (BufferElement::Scalar(ScalarType::I32), _) => t == "array<i32>" || t == "i32",
        (BufferElement::Scalar(ScalarType::U32), BufferAccess::Uniform) => {
            t == "Params" || t == "array<u32>" || t == "u32"
        }
        (BufferElement::Scalar(ScalarType::U32), _) => t == "array<u32>" || t == "u32",
        (BufferElement::Scalar(ScalarType::U64Words), _) => {
            t == "array<vec2<u32>>" || t == "vec2<u32>"
        }
        (BufferElement::AffineParams, _) => t == "AffineParams",
        (BufferElement::IqPowerParams, _) => t == "IqPowerParams",
        (BufferElement::IqPowerI16Params, _) => t == "IqPowerI16Params",
        (BufferElement::P64Words64, _) => t == "array<P64Words64>" || t == "P64Words64",
        (BufferElement::AccelerationStructure, _) => t == "acceleration_structure",
    }
}

fn parse_workgroup_size(source: &str) -> Result<WorkgroupSize, ForgeError> {
    let Some(at) = find_not_comment(source, "@workgroup_size(") else {
        return Err(ForgeError::Emission(
            "emitted WGSL missing @workgroup_size".to_string(),
        ));
    };
    let mut pos = at + "@workgroup_size(".len();
    let x = parse_u32_at(source, &mut pos)?;
    skip_ws_comments(source, &mut pos);
    let (y, z) = if source.as_bytes().get(pos) == Some(&b',') {
        pos += 1;
        let y = parse_u32_at(source, &mut pos)?;
        skip_ws_comments(source, &mut pos);
        expect_byte(source, &mut pos, b',')?;
        let z = parse_u32_at(source, &mut pos)?;
        (y, z)
    } else {
        (1, 1)
    };
    Ok(WorkgroupSize::new(x, y, z))
}

fn parse_access(spec: &str) -> Result<BufferAccess, ForgeError> {
    match spec {
        "storage, read" | "storage,read" => Ok(BufferAccess::StorageRead),
        "storage, read_write" | "storage,read_write" => Ok(BufferAccess::StorageReadWrite),
        "uniform" => Ok(BufferAccess::Uniform),
        other => Err(ForgeError::Emission(format!(
            "unsupported WGSL buffer access {other:?}"
        ))),
    }
}

fn in_line_comment(source: &str, idx: usize) -> bool {
    let start = source[..idx].rfind('\n').map(|i| i + 1).unwrap_or(0);
    source[start..idx].contains("//")
}

fn find_not_comment(source: &str, needle: &str) -> Option<usize> {
    let mut cursor = 0usize;
    while let Some(rel) = source[cursor..].find(needle) {
        let at = cursor + rel;
        if !in_line_comment(source, at) {
            return Some(at);
        }
        cursor = at + needle.len();
    }
    None
}

fn skip_ws_comments(source: &str, pos: &mut usize) {
    let bytes = source.as_bytes();
    loop {
        while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if *pos + 1 < bytes.len() && bytes[*pos] == b'/' && bytes[*pos + 1] == b'/' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
            continue;
        }
        break;
    }
}

fn parse_u32_at(source: &str, pos: &mut usize) -> Result<u32, ForgeError> {
    skip_ws_comments(source, pos);
    let rest = &source[*pos..];
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return Err(ForgeError::Emission(
            "expected integer in WGSL declaration".to_string(),
        ));
    }
    let value = rest[..digits]
        .parse::<u32>()
        .map_err(|error| ForgeError::Emission(error.to_string()))?;
    *pos += digits;
    if matches!(source.as_bytes().get(*pos), Some(b'u' | b'i')) {
        *pos += 1;
    }
    Ok(value)
}

fn parse_ident_at<'a>(source: &'a str, pos: &mut usize) -> Result<&'a str, ForgeError> {
    skip_ws_comments(source, pos);
    let rest = &source[*pos..];
    let mut chars = rest.char_indices();
    let Some((_, first)) = chars.next() else {
        return Err(ForgeError::Emission(
            "expected identifier in WGSL declaration".to_string(),
        ));
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(ForgeError::Emission(format!(
            "expected identifier, found {:?}",
            rest.chars().next()
        )));
    }
    let mut end = first.len_utf8();
    for (index, ch) in chars {
        if ch == '_' || ch.is_ascii_alphanumeric() {
            end = index + ch.len_utf8();
        } else {
            break;
        }
    }
    *pos += end;
    Ok(&rest[..end])
}

fn expect_byte(source: &str, pos: &mut usize, byte: u8) -> Result<(), ForgeError> {
    skip_ws_comments(source, pos);
    if source.as_bytes().get(*pos) != Some(&byte) {
        return Err(ForgeError::Emission(format!(
            "expected '{}' in WGSL declaration",
            byte as char
        )));
    }
    *pos += 1;
    Ok(())
}
