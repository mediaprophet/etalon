#![cfg_attr(all(fuzzable_platform, fuzzing), no_main)]

#[cfg(all(fuzzable_platform, fuzzing))]
mod fuzz {
    use libfuzzer_sys::fuzz_target;

    fuzz_target!(|module: naga::Module| {
        use naga::valid as v;
        // Validate module with all validation flags
        let mut validator =
            v::Validator::new(v::ValidationFlags::all(), v::Capabilities::default());
        let _result = validator.validate(&module);
    });
}

#[cfg(not(all(fuzzable_platform, fuzzing)))]
fn main() {}
