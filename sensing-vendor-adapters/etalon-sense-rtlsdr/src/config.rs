use crate::{abi::Api, abi::Dev, Error};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GainMode {
    Automatic,
    ManualTenthsDb(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    center_frequency_hz: u32,
    sample_rate_hz: u32,
    frequency_correction_ppm: i32,
    tuner_gain: GainMode,
    digital_agc: bool,
}

impl Config {
    pub const fn new(
        center_frequency_hz: u32,
        sample_rate_hz: u32,
        frequency_correction_ppm: i32,
        tuner_gain: GainMode,
        digital_agc: bool,
    ) -> Result<Self, Error> {
        if center_frequency_hz == 0
            || sample_rate_hz <= 225_000
            || sample_rate_hz > 3_200_000
            || (sample_rate_hz > 300_000 && sample_rate_hz <= 900_000)
            || frequency_correction_ppm < -1_000
            || frequency_correction_ppm > 1_000
            || matches!(tuner_gain, GainMode::ManualTenthsDb(gain) if gain < -1_000 || gain > 1_000)
        {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            center_frequency_hz,
            sample_rate_hz,
            frequency_correction_ppm,
            tuner_gain,
            digital_agc,
        })
    }
    pub const fn center_frequency_hz(self) -> u32 {
        self.center_frequency_hz
    }
    pub const fn sample_rate_hz(self) -> u32 {
        self.sample_rate_hz
    }
    pub const fn frequency_correction_ppm(self) -> i32 {
        self.frequency_correction_ppm
    }
    pub const fn tuner_gain(self) -> GainMode {
        self.tuner_gain
    }
    pub const fn digital_agc(self) -> bool {
        self.digital_agc
    }
}

/// Native getter declarations, not independently measured RF or clock values.
/// The upstream sample-rate getter truncates its computed resampler rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppliedConfig {
    center_frequency_hz: u32,
    sample_rate_hz: u32,
    frequency_correction_ppm: i32,
    tuner_gain_tenths_db: Option<i32>,
}

impl AppliedConfig {
    pub const fn center_frequency_hz(self) -> u32 {
        self.center_frequency_hz
    }
    pub const fn sample_rate_hz(self) -> u32 {
        self.sample_rate_hz
    }
    pub const fn frequency_correction_ppm(self) -> i32 {
        self.frequency_correction_ppm
    }
    pub const fn tuner_gain_tenths_db(self) -> Option<i32> {
        self.tuner_gain_tenths_db
    }
}

pub(crate) unsafe fn apply(
    api: Api,
    handle: *mut Dev,
    config: Config,
) -> Result<AppliedConfig, Error> {
    checked("set_testmode", unsafe { (api.set_testmode)(handle, 0) })?;
    checked("set_direct_sampling", unsafe {
        (api.set_direct)(handle, 0)
    })?;
    if unsafe { (api.get_direct)(handle) } != 0 {
        return Err(Error::ConfigurationChanged);
    }
    if unsafe { (api.get_correction)(handle) } != config.frequency_correction_ppm {
        checked("set_freq_correction", unsafe {
            (api.set_correction)(handle, config.frequency_correction_ppm)
        })?;
    }
    checked("set_sample_rate", unsafe {
        (api.set_rate)(handle, config.sample_rate_hz)
    })?;
    checked("set_center_freq", unsafe {
        (api.set_center)(handle, config.center_frequency_hz)
    })?;
    checked("set_tuner_gain_mode", unsafe {
        (api.set_gain_mode)(
            handle,
            i32::from(matches!(config.tuner_gain, GainMode::ManualTenthsDb(_))),
        )
    })?;
    if let GainMode::ManualTenthsDb(gain) = config.tuner_gain {
        let count = unsafe { (api.get_gains)(handle, core::ptr::null_mut()) };
        if count <= 0 || count > 64 {
            return Err(Error::InvalidNativeReport);
        }
        let mut gains = [0i32; 64];
        let returned = unsafe { (api.get_gains)(handle, gains.as_mut_ptr()) };
        if returned != count || !gains[..count as usize].contains(&gain) {
            return Err(Error::InvalidConfig);
        }
        checked("set_tuner_gain", unsafe { (api.set_gain)(handle, gain) })?;
    }
    checked("set_agc_mode", unsafe {
        (api.set_agc)(handle, i32::from(config.digital_agc))
    })?;
    checked("reset_buffer", unsafe { (api.reset)(handle) })?;
    let actual = unsafe { reported(api, handle, config) }?;
    if actual.center_frequency_hz != config.center_frequency_hz
        || actual.frequency_correction_ppm != config.frequency_correction_ppm
        || matches!(config.tuner_gain, GainMode::ManualTenthsDb(gain) if actual.tuner_gain_tenths_db != Some(gain))
    {
        return Err(Error::ConfigurationChanged);
    }
    Ok(actual)
}

pub(crate) unsafe fn reported(
    api: Api,
    handle: *mut Dev,
    config: Config,
) -> Result<AppliedConfig, Error> {
    let center = unsafe { (api.get_center)(handle) };
    let rate = unsafe { (api.get_rate)(handle) };
    if center == 0 || rate == 0 || unsafe { (api.get_direct)(handle) } != 0 {
        return Err(Error::InvalidNativeReport);
    }
    Ok(AppliedConfig {
        center_frequency_hz: center,
        sample_rate_hz: rate,
        frequency_correction_ppm: unsafe { (api.get_correction)(handle) },
        tuner_gain_tenths_db: match config.tuner_gain {
            GainMode::Automatic => None,
            GainMode::ManualTenthsDb(_) => Some(unsafe { (api.get_gain)(handle) }),
        },
    })
}

pub(crate) fn checked(operation: &'static str, code: i32) -> Result<(), Error> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Native { operation, code })
    }
}
