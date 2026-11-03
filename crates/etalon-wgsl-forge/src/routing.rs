use crate::{
    emit_cuda_c, emit_shader, ForgeError, GeneratedShader, KernelSpec, Schedule, TargetBackend,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoutePreference {
    #[default]
    Portable,
    PreferCuda,
    RequireCuda,
}

/// Caller-supplied capabilities for the selected device, not a performance ranking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaRouteSupport {
    pub driver_available: bool,
    pub compiler_available: bool,
    pub selected_device_matches: bool,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteCapabilities {
    pub cuda: CudaRouteSupport,
    pub portable_target: TargetBackend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaUnavailableReason {
    UnsupportedPlatform,
    MissingDriver,
    MissingCompiler,
    SelectedDeviceMismatch,
    MissingOrInvalidArchitecture,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteReason {
    PortableRequested,
    CudaSelected,
    CudaUnavailable(CudaUnavailableReason),
    CudaEmissionUnsupported(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedShader {
    pub shader: GeneratedShader,
    pub target: TargetBackend,
    pub compute_capability: Option<(u32, u32)>,
    pub reason: RouteReason,
}

impl CudaRouteSupport {
    fn unavailable_reason(self) -> Option<CudaUnavailableReason> {
        if cfg!(not(any(target_os = "windows", target_os = "linux"))) {
            Some(CudaUnavailableReason::UnsupportedPlatform)
        } else if !self.driver_available {
            Some(CudaUnavailableReason::MissingDriver)
        } else if !self.selected_device_matches {
            Some(CudaUnavailableReason::SelectedDeviceMismatch)
        } else if !self.compiler_available {
            Some(CudaUnavailableReason::MissingCompiler)
        } else if !matches!(self.compute_capability, Some((1..=99, 0..=9))) {
            Some(CudaUnavailableReason::MissingOrInvalidArchitecture)
        } else {
            None
        }
    }
}

/// Emits source using the caller's policy and device capabilities.
/// CUDA-C output still requires compilation and a native execution owner.
pub fn emit_routed_shader(
    kernel: &KernelSpec,
    schedule: Schedule,
    preference: RoutePreference,
    capabilities: RouteCapabilities,
) -> Result<RoutedShader, ForgeError> {
    kernel.validate()?;
    let reason = if preference == RoutePreference::Portable {
        RouteReason::PortableRequested
    } else if let Some(reason) = capabilities.cuda.unavailable_reason() {
        if preference == RoutePreference::RequireCuda {
            return Err(ForgeError::GpuUnavailable(format!(
                "CUDA required: {reason:?}"
            )));
        }
        RouteReason::CudaUnavailable(reason)
    } else {
        match emit_cuda_c(kernel, schedule) {
            Ok(shader) => {
                return Ok(RoutedShader {
                    shader,
                    target: TargetBackend::CudaC,
                    compute_capability: capabilities.cuda.compute_capability,
                    reason: RouteReason::CudaSelected,
                });
            }
            Err(ForgeError::Emission(message)) if preference == RoutePreference::PreferCuda => {
                RouteReason::CudaEmissionUnsupported(message)
            }
            Err(error) => return Err(error),
        }
    };
    if !matches!(
        capabilities.portable_target,
        TargetBackend::Wgsl | TargetBackend::Hlsl | TargetBackend::Msl
    ) {
        return Err(ForgeError::Emission(
            "Portable routing requires a WGSL, HLSL, or MSL source target".into(),
        ));
    }
    Ok(RoutedShader {
        shader: emit_shader(kernel, schedule, capabilities.portable_target)?,
        target: capabilities.portable_target,
        compute_capability: None,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BuiltinKernel;

    fn supported() -> RouteCapabilities {
        RouteCapabilities {
            cuda: CudaRouteSupport {
                driver_available: true,
                compiler_available: true,
                selected_device_matches: true,
                compute_capability: Some((8, 6)),
            },
            portable_target: TargetBackend::Wgsl,
        }
    }

    fn route(
        preference: RoutePreference,
        capabilities: RouteCapabilities,
    ) -> Result<RoutedShader, ForgeError> {
        emit_routed_shader(
            &BuiltinKernel::IqPowerF32.spec(),
            Schedule::default(),
            preference,
            capabilities,
        )
    }

    #[test]
    fn portable_policy_uses_requested_portable_source() {
        for portable_target in [TargetBackend::Wgsl, TargetBackend::Hlsl, TargetBackend::Msl] {
            let result = route(
                RoutePreference::Portable,
                RouteCapabilities {
                    portable_target,
                    ..supported()
                },
            )
            .unwrap();
            assert_eq!(result.target, portable_target);
            assert_eq!(result.reason, RouteReason::PortableRequested);
            assert_eq!(result.compute_capability, None);
        }
    }

    #[test]
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    fn cuda_selection_requires_all_device_capabilities() {
        let ready = supported();
        let cases = [
            (
                CudaRouteSupport {
                    driver_available: false,
                    ..ready.cuda
                },
                CudaUnavailableReason::MissingDriver,
            ),
            (
                CudaRouteSupport {
                    compiler_available: false,
                    ..ready.cuda
                },
                CudaUnavailableReason::MissingCompiler,
            ),
            (
                CudaRouteSupport {
                    selected_device_matches: false,
                    ..ready.cuda
                },
                CudaUnavailableReason::SelectedDeviceMismatch,
            ),
            (
                CudaRouteSupport {
                    compute_capability: None,
                    ..ready.cuda
                },
                CudaUnavailableReason::MissingOrInvalidArchitecture,
            ),
            (
                CudaRouteSupport {
                    compute_capability: Some((8, 10)),
                    ..ready.cuda
                },
                CudaUnavailableReason::MissingOrInvalidArchitecture,
            ),
        ];
        for (cuda, reason) in cases {
            let facts = RouteCapabilities { cuda, ..ready };
            let fallback = route(RoutePreference::PreferCuda, facts).unwrap();
            assert_eq!(fallback.target, TargetBackend::Wgsl);
            assert_eq!(fallback.reason, RouteReason::CudaUnavailable(reason));
            assert_eq!(fallback.compute_capability, None);
            assert!(matches!(
                route(RoutePreference::RequireCuda, facts),
                Err(ForgeError::GpuUnavailable(_))
            ));
        }
        for preference in [RoutePreference::PreferCuda, RoutePreference::RequireCuda] {
            let result = route(preference, ready).unwrap();
            assert_eq!(result.target, TargetBackend::CudaC);
            assert_eq!(result.reason, RouteReason::CudaSelected);
            assert_eq!(result.compute_capability, Some((8, 6)));
            assert!(result
                .shader
                .source
                .contains("__global__ void iq_power_f32"));
        }
    }

    #[test]
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    fn device_mismatch_is_reported_without_compiler_probing() {
        let capabilities = RouteCapabilities {
            cuda: CudaRouteSupport {
                compiler_available: false,
                selected_device_matches: false,
                ..supported().cuda
            },
            ..supported()
        };
        let fallback = route(RoutePreference::PreferCuda, capabilities).unwrap();
        assert_eq!(
            fallback.reason,
            RouteReason::CudaUnavailable(CudaUnavailableReason::SelectedDeviceMismatch)
        );
        assert!(route(RoutePreference::RequireCuda, capabilities).is_err());
    }

    #[test]
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    fn unsupported_cuda_kernel_falls_back_only_when_policy_allows() {
        let kernel = BuiltinKernel::RayProbe.spec();
        let result = emit_routed_shader(
            &kernel,
            Schedule::default(),
            RoutePreference::PreferCuda,
            supported(),
        )
        .unwrap();
        assert_eq!(result.target, TargetBackend::Wgsl);
        assert!(matches!(
            result.reason,
            RouteReason::CudaEmissionUnsupported(_)
        ));
        assert!(emit_routed_shader(
            &kernel,
            Schedule::default(),
            RoutePreference::RequireCuda,
            supported()
        )
        .is_err());
    }

    #[test]
    fn portable_policy_rejects_cuda_as_fallback_target() {
        assert!(route(
            RoutePreference::Portable,
            RouteCapabilities {
                portable_target: TargetBackend::CudaC,
                ..supported()
            }
        )
        .is_err());
    }

    #[test]
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    fn native_cuda_route_is_unavailable_on_unsupported_platforms() {
        let result = route(RoutePreference::PreferCuda, supported()).unwrap();
        assert_eq!(result.target, TargetBackend::Wgsl);
        assert_eq!(
            result.reason,
            RouteReason::CudaUnavailable(CudaUnavailableReason::UnsupportedPlatform)
        );
        assert!(route(RoutePreference::RequireCuda, supported()).is_err());
    }
}
