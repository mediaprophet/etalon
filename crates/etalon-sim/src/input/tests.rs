//! Unit Tests for Input Bitsets & Touch Controls.

use super::buttons::*;
use super::touch::*;

#[test]
fn test_zero_heap_button_transitions() {
    let mut input = ZeroHeapButtonInput::<4>::new();

    assert!(!input.is_pressed(KEY_W));
    assert!(!input.is_just_pressed(KEY_W));

    // Press KEY_W
    input.press(KEY_W);
    assert!(input.is_pressed(KEY_W));
    assert!(input.is_just_pressed(KEY_W));
    assert!(!input.is_just_released(KEY_W));

    // Next frame: remains pressed, but no longer just_pressed
    input.advance_frame();
    assert!(input.is_pressed(KEY_W));
    assert!(!input.is_just_pressed(KEY_W));
    assert!(!input.is_just_released(KEY_W));

    // Release KEY_W
    input.release(KEY_W);
    assert!(!input.is_pressed(KEY_W));
    assert!(!input.is_just_pressed(KEY_W));
    assert!(input.is_just_released(KEY_W));

    // Next frame: clean
    input.advance_frame();
    assert!(!input.is_pressed(KEY_W));
    assert!(!input.is_just_released(KEY_W));
}

#[test]
fn test_virtual_touch_stick_deadzone_and_clamp() {
    let mut stick = VirtualTouchStick::new([100.0, 100.0], 50.0);
    assert_eq!(stick.axis, [0.0, 0.0]);

    // Touch slightly off center (within deadzone of 0.1 * 50 = 5px)
    assert!(stick.on_touch_start(1, 103.0, 100.0));
    assert_eq!(stick.axis, [0.0, 0.0]);

    // Drag fully to right beyond radius (100 + 75 = 175px)
    assert!(stick.on_touch_move(1, 175.0, 100.0));
    assert!((stick.axis[0] - 1.0).abs() < 1e-4);
    assert_eq!(stick.axis[1], 0.0);

    // Release touch
    assert!(stick.on_touch_end(1));
    assert_eq!(stick.axis, [0.0, 0.0]);
    assert!(!stick.is_active);
}

#[test]
fn test_multi_touch_tracker() {
    let mut tracker = MultiTouchTracker::new();
    assert_eq!(tracker.active_count(), 0);

    tracker.touch_start(101, 50.0, 50.0);
    tracker.touch_start(102, 150.0, 150.0);
    assert_eq!(tracker.active_count(), 2);

    tracker.touch_move(101, 55.0, 52.0);
    assert_eq!(tracker.touches[0].delta, [5.0, 2.0]);

    tracker.touch_end(101);
    assert_eq!(tracker.active_count(), 1);
}
