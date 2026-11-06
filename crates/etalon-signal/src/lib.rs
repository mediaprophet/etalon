//! Small, allocation-free DSP primitives over caller-owned sample buffers.
//!
//! These routines transform or summarize samples only. They do not acquire
//! hardware data, establish calibration, identify a source, or authenticate a
//! measurement.
#![no_std]
#![forbid(unsafe_code)]

pub mod audio;
pub mod correlation;
pub mod decode;
pub mod detection;
pub mod dft;
pub mod execution;
pub mod fft;
pub mod filter;
pub mod goertzel;
pub mod iq;
pub mod loudness;
pub mod mel;
pub mod phase;
pub mod pipeline;
pub mod pitch;
pub mod resample;
pub mod reverb;
pub mod spatial;
pub mod stats;
pub mod temporal;
pub mod vision;
pub mod wave;
pub mod window;

pub use audio::{
    convolve_fir_into, fill_hann_window_into, hann_window_sample, head_relative_position,
    istft_overlap_add_into, pcm16_channel_power, pinna_elevation_notch_hz, spherical_ild_db,
    stft_frame_into, stft_one_sided_magnitudes_into, synthesize_hrir_into, verify_cola_window,
    woodworth_itd_seconds, ChannelPower, DEFAULT_HEAD_RADIUS_M, DEFAULT_SPEED_OF_SOUND_M_S,
};
pub use dft::{
    dft_interleaved_into, fft_radix2_in_place, fft_radix2_interleaved_f32, DftScale,
    MAX_DFT_SAMPLES,
};
pub use filter::{
    design_k_weighting_stage1, design_k_weighting_stage2, df1_step_f32, df2t_step_f32,
    rbj_bandpass, rbj_high_shelf, rbj_highpass, rbj_low_shelf, rbj_lowpass, rbj_notch,
    rbj_peaking_eq, BiquadCoeffs, BiquadDf2t, CascadedBiquadDf2t, OnePoleLowPass,
};
pub use loudness::{
    bark_to_hz, crest_factor_db, crest_factor_f32, dct_ii_into, energy_to_lufs, erb_to_hz,
    hz_to_bark, hz_to_erb, rms_dbfs, rms_f32,
};
pub use mel::{hz_to_mel, mel_apply_into, mel_to_hz, MelShape, PreparedMel};
pub use pitch::{
    cents_deviation, hz_to_midi, hz_to_midi_quantized, midi_note_to_hz, midi_to_hz,
    yin_absolute_threshold, yin_cmndf_in_place, yin_difference_into, yin_estimate_f0_hz,
    yin_parabolic_interp, yin_parabolic_min, yin_pitch_confidence,
};
pub use resample::{
    anti_alias_cutoff_hz, antialias_cutoff_cycles, kaiser_bessel_i0, pcm_f32_to_i16_le_into,
    pcm_to_mono_f32_into, polyphase_fir_step, resample_linear_mono_into, resample_polyphase_into,
    windowed_sinc_resample_into, windowed_sinc_sample,
};
pub use reverb::{
    allpass_step, comb_step, fdn4_hadamard_mix, stereo_demux_into, stereo_mux_into,
    stereo_width_in_place, stereo_width_interleaved_into,
};
pub use spatial::{
    mix_spatial_voice_into, mix_spatial_voice_occluded_into, DistanceModel, SpatialEmitter,
    SpatialListener, SpatialOcclusionBridge, SpatialOcclusionFilterState, SpatialVoiceParams,
    MAX_SPATIAL_VOICES, SPEED_OF_SOUND_AIR,
};
pub use vision::{
    ahash_from_luma_64, cosine_distance_f32, cosine_similarity_f32, dhash_from_luma_72,
    frame_motion_mad_u8, hamming_distance_u64, histogram_u8, iou_xywh_u16,
    laplacian_blur_variance_u8, mean_u8_region, normalize_rgb8_to_f32_chw_into,
    normalized_hamming_u64, otsu_threshold_from_hist, resize_nearest_rgb8_into, roi_mean_rgb_u8,
    ImageRegion,
};
pub use window::{
    apply_window_in_place, apply_window_into, blackman_harris_window, blackman_window,
    hamming_window, hann_window,
};

/// Failures are explicit; no routine invents a nominal signal or estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalError {
    EmptyInput,
    LengthMismatch,
    NonFiniteInput,
    InvalidParameter,
    NumericRange,
}

#[cfg(test)]
extern crate std;
