fn main() {
    cfg_aliases::cfg_aliases! {
        windows_linux_android: { any(windows, target_os = "linux", target_os = "android", target_os = "freebsd", target_os = "netbsd") },
        send_sync: { all(
            feature = "std",
            any(
                not(target_family = "wasm"),
                all(feature = "fragile-send-sync-non-atomic-wasm", not(target_feature = "atomics"))
            )
        ) },
        dx12: { all(target_os = "windows", feature = "dx12") },
        webgl: { all(target_family = "wasm", not(target_os = "emscripten"), feature = "webgl") },
        gles: { any(
            all(windows_linux_android, feature = "gles"), // Regular GLES
            all(webgl), // WebGL
            all(target_os = "emscripten", feature = "gles"), // Emscripten GLES
            all(target_vendor = "apple", feature = "angle") // ANGLE on Apple
        ) },
        vulkan: { any(
            all(windows_linux_android, feature = "vulkan"), // Regular Vulkan
            all(target_vendor = "apple", feature = "vulkan-portability") // Vulkan Portability on Apple
        ) },
        drm: { all(
            feature = "drm",
            any(target_os = "linux", target_os = "freebsd", target_os = "netbsd", target_os = "openbsd")
        ) },
        metal: { all(target_vendor = "apple", feature = "metal") },

        // `etalon_validate_locks` can also be set directly, if desired.
        // See `etalon-core/src/lock/mod.rs`.
        etalon_validate_locks: {
            all(
                debug_assertions,
                etalon_validate_locks_debug,
                not(target_family = "wasm")
            )
        }
    }
}
