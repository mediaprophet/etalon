//! Macros for validation internal to the etalon.
//!
//! This module defines assertion macros that respect `etalon-type`'s
//! `"strict_asserts"` feature.
//!
//! Because `etalon-core`'s public APIs validate their arguments in all
//! types of builds, for performance, the `track` module skips some of
//! Rust's usual run-time checks on its internal operations in release
//! builds. However, some `etalon-core` applications have a strong
//! preference for robustness over performance. To accommodate them,
//! `etalon-core`'s `"strict_asserts"` feature enables that validation
//! in both debug and release builds.

/// This is equivalent to [`core::assert`] if the `strict_asserts` feature is activated, otherwise equal to [`core::debug_assert`].
#[cfg(feature = "strict_asserts")]
#[macro_export]
macro_rules! strict_assert {
    ( $( $arg:tt )* ) => {
        assert!( $( $arg )* )
    }
}

/// This is equivalent to [`core::assert_eq`] if the `strict_asserts` feature is activated, otherwise equal to [`core::debug_assert_eq`].
#[cfg(feature = "strict_asserts")]
#[macro_export]
macro_rules! strict_assert_eq {
    ( $( $arg:tt )* ) => {
        assert_eq!( $( $arg )* )
    }
}

/// This is equivalent to [`core::assert_ne`] if the `strict_asserts` feature is activated, otherwise equal to [`core::debug_assert_ne`].
#[cfg(feature = "strict_asserts")]
#[macro_export]
macro_rules! strict_assert_ne {
    ( $( $arg:tt )* ) => {
        assert_ne!( $( $arg )* )
    }
}

/// This is equivalent to [`core::assert`] if the `strict_asserts` feature is activated, otherwise equal to [`core::debug_assert`]
#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert {
    ( $( $arg:tt )* ) => {
        debug_assert!( $( $arg )* )
    };
}

/// This is equivalent to [`core::assert_eq`] if the `strict_asserts` feature is activated, otherwise equal to [`core::debug_assert_eq`]
#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert_eq {
    ( $( $arg:tt )* ) => {
        debug_assert_eq!( $( $arg )* )
    };
}

/// This is equivalent to [`core::assert_ne`] if the `strict_asserts` feature is activated, otherwise equal to [`core::debug_assert_ne`]
#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert_ne {
    ( $( $arg:tt )* ) => {
        debug_assert_ne!( $( $arg )* )
    };
}
