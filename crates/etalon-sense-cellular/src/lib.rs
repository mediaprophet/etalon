//! Optional host-gated Windows WWAN connection telemetry.
//!
//! Signal bars are an OS-reported ordinal quality indicator, not dBm, CSI,
//! 4G/5G identification, distance, calibrated field strength or proof of live
//! radio measurements. A profile and its values may be unavailable or stale.
#![forbid(unsafe_code)]

pub mod observation;
pub mod telemetry;

pub use observation::{
    adapt_cellular, AdaptError as ObservationAdaptError, CellularMetric, CellularObservation,
    CellularObservationMapping, CELLULAR_METRIC_CAPACITY,
};
pub use telemetry::{
    validate_batch, CellularTelemetry, HostCellularTelemetry, OrdinalBars, RadioGeneration,
    TelemetryError, TelemetryTimestamp,
};

/// A host supplies policy for each OS query. It decides which ontologies and
/// human/environmental contexts apply; this crate does not define those rules.
pub trait PolicyGate {
    fn permit_query(&mut self) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellularError {
    Denied,
    ProfileUnavailable,
    NotCellular,
    QueryFailed,
}

/// Raw OS value and explicitly unknown freshness. No stable unit is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellularSignalBars {
    pub bars: u8,
    /// OS does not provide a measurement timestamp through this API.
    pub measurement_time_known: bool,
}

#[cfg(any(windows, test))]
trait CellularPlatform {
    type Profile;
    type BarsReference;

    fn internet_profile(&mut self) -> Result<Self::Profile, CellularError>;
    fn is_wwan(&mut self, profile: &Self::Profile) -> Result<bool, CellularError>;
    fn get_signal_bars(
        &mut self,
        profile: &Self::Profile,
    ) -> Result<Self::BarsReference, CellularError>;
    fn bars_value(&mut self, bars: &Self::BarsReference) -> Result<u8, CellularError>;
}

#[cfg(any(windows, test))]
fn read_bars_with_platform(
    gate: &mut impl PolicyGate,
    platform: &mut impl CellularPlatform,
) -> Result<CellularSignalBars, CellularError> {
    if !gate.permit_query() {
        return Err(CellularError::Denied);
    }
    let profile = platform.internet_profile()?;
    if !gate.permit_query() {
        return Err(CellularError::Denied);
    }
    if !platform.is_wwan(&profile)? {
        return Err(CellularError::NotCellular);
    }
    if !gate.permit_query() {
        return Err(CellularError::Denied);
    }
    let bars = platform.get_signal_bars(&profile)?;
    if !gate.permit_query() {
        return Err(CellularError::Denied);
    }
    let value = platform.bars_value(&bars)?;
    Ok(CellularSignalBars {
        bars: value,
        measurement_time_known: false,
    })
}

/// Query the default internet profile only if it is a WWAN profile. The caller
/// initializes WinRT on this thread and supplies the policy gate. This does not
/// enumerate radios, connect to a network, trigger a scan or infer signal dBm.
#[cfg(windows)]
pub fn read_default_wwan_bars(
    gate: &mut impl PolicyGate,
) -> Result<CellularSignalBars, CellularError> {
    use windows::Networking::Connectivity::NetworkInformation;
    struct WindowsPlatform;
    impl CellularPlatform for WindowsPlatform {
        type Profile = windows::Networking::Connectivity::ConnectionProfile;
        type BarsReference = windows::Foundation::IReference<u8>;

        fn internet_profile(&mut self) -> Result<Self::Profile, CellularError> {
            NetworkInformation::GetInternetConnectionProfile()
                .map_err(|_| CellularError::ProfileUnavailable)
        }

        fn is_wwan(&mut self, profile: &Self::Profile) -> Result<bool, CellularError> {
            profile
                .IsWwanConnectionProfile()
                .map_err(|_| CellularError::QueryFailed)
        }

        fn get_signal_bars(
            &mut self,
            profile: &Self::Profile,
        ) -> Result<Self::BarsReference, CellularError> {
            profile
                .GetSignalBars()
                .map_err(|_| CellularError::QueryFailed)
        }

        fn bars_value(&mut self, bars: &Self::BarsReference) -> Result<u8, CellularError> {
            bars.Value().map_err(|_| CellularError::QueryFailed)
        }
    }

    read_bars_with_platform(gate, &mut WindowsPlatform)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Call {
        Profile,
        IsWwan,
        GetBars,
        Value,
    }

    struct FakePlatform {
        calls: [Option<Call>; 4],
        count: usize,
        is_wwan: bool,
        bars: u8,
    }

    impl FakePlatform {
        fn new(is_wwan: bool, bars: u8) -> Self {
            Self {
                calls: [None; 4],
                count: 0,
                is_wwan,
                bars,
            }
        }

        fn record(&mut self, call: Call) {
            self.calls[self.count] = Some(call);
            self.count += 1;
        }
    }

    impl CellularPlatform for FakePlatform {
        type Profile = ();
        type BarsReference = u8;

        fn internet_profile(&mut self) -> Result<Self::Profile, CellularError> {
            self.record(Call::Profile);
            Ok(())
        }

        fn is_wwan(&mut self, _: &Self::Profile) -> Result<bool, CellularError> {
            self.record(Call::IsWwan);
            Ok(self.is_wwan)
        }

        fn get_signal_bars(
            &mut self,
            _: &Self::Profile,
        ) -> Result<Self::BarsReference, CellularError> {
            self.record(Call::GetBars);
            Ok(self.bars)
        }

        fn bars_value(&mut self, bars: &Self::BarsReference) -> Result<u8, CellularError> {
            self.record(Call::Value);
            Ok(*bars)
        }
    }

    struct AllowAll;
    impl PolicyGate for AllowAll {
        fn permit_query(&mut self) -> bool {
            true
        }
    }

    struct DenyAt {
        remaining_allows: usize,
        calls: usize,
    }
    impl PolicyGate for DenyAt {
        fn permit_query(&mut self) -> bool {
            self.calls += 1;
            if self.remaining_allows == 0 {
                false
            } else {
                self.remaining_allows -= 1;
                true
            }
        }
    }

    #[test]
    fn gates_every_platform_call_and_marks_time_unknown() {
        let mut gate = AllowAll;
        let mut platform = FakePlatform::new(true, 3);
        assert_eq!(
            read_bars_with_platform(&mut gate, &mut platform),
            Ok(CellularSignalBars {
                bars: 3,
                measurement_time_known: false,
            })
        );
        assert_eq!(
            platform.calls,
            [
                Some(Call::Profile),
                Some(Call::IsWwan),
                Some(Call::GetBars),
                Some(Call::Value)
            ]
        );
    }

    #[test]
    fn denial_before_each_platform_call_stops_the_query() {
        for allowed_calls in 0..4 {
            let mut gate = DenyAt {
                remaining_allows: allowed_calls,
                calls: 0,
            };
            let mut platform = FakePlatform::new(true, 2);
            assert_eq!(
                read_bars_with_platform(&mut gate, &mut platform),
                Err(CellularError::Denied)
            );
            assert_eq!(platform.count, allowed_calls);
            assert_eq!(gate.calls, allowed_calls + 1);
        }
    }

    #[test]
    fn non_wwan_profile_never_queries_signal_bars() {
        let mut gate = AllowAll;
        let mut platform = FakePlatform::new(false, 0);
        assert_eq!(
            read_bars_with_platform(&mut gate, &mut platform),
            Err(CellularError::NotCellular)
        );
        assert_eq!(
            platform.calls,
            [Some(Call::Profile), Some(Call::IsWwan), None, None]
        );
    }
}
