//! Zero-Heap Bitset Input Engine.
//!
//! Provides $O(1)$ zero-allocation input tracking adapted from game engine architectures
//! (replacing dynamic HashSets with fixed CPU word bitsets).

pub const INPUT_WORDS_4: usize = 4; // 256 discrete key / button channels

// Standard key codes mapped to flat bit indices (0..255)
pub const KEY_W: u16 = 87;
pub const KEY_A: u16 = 65;
pub const KEY_S: u16 = 83;
pub const KEY_D: u16 = 68;
pub const KEY_SPACE: u16 = 32;
pub const KEY_SHIFT: u16 = 16;
pub const KEY_ESCAPE: u16 = 27;
pub const KEY_E: u16 = 69;
pub const KEY_Q: u16 = 81;
pub const KEY_TAB: u16 = 9;
pub const KEY_ARROW_UP: u16 = 38;
pub const KEY_ARROW_DOWN: u16 = 40;
pub const KEY_ARROW_LEFT: u16 = 37;
pub const KEY_ARROW_RIGHT: u16 = 39;
pub const BUTTON_PRIMARY: u16 = 1;
pub const BUTTON_SECONDARY: u16 = 2;
pub const BUTTON_MIDDLE: u16 = 3;

/// Zero-heap bitset representing 256 discrete button states with single-frame transition events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroHeapButtonInput<const WORDS: usize = INPUT_WORDS_4> {
    pub pressed: [u64; WORDS],
    pub just_pressed: [u64; WORDS],
    pub just_released: [u64; WORDS],
}

impl<const WORDS: usize> Default for ZeroHeapButtonInput<WORDS> {
    fn default() -> Self {
        Self {
            pressed: [0; WORDS],
            just_pressed: [0; WORDS],
            just_released: [0; WORDS],
        }
    }
}

impl<const WORDS: usize> ZeroHeapButtonInput<WORDS> {
    pub const fn new() -> Self {
        Self {
            pressed: [0; WORDS],
            just_pressed: [0; WORDS],
            just_released: [0; WORDS],
        }
    }

    /// Clears per-frame transient edge transitions (`just_pressed` and `just_released`).
    /// Call at the start of each frame.
    #[inline]
    pub fn advance_frame(&mut self) {
        self.just_pressed = [0; WORDS];
        self.just_released = [0; WORDS];
    }

    /// Records a button press event.
    #[inline]
    pub fn press(&mut self, key_code: u16) {
        let bit = (key_code as usize) % (WORDS * 64);
        let word = bit / 64;
        let mask = 1u64 << (bit % 64);

        if (self.pressed[word] & mask) == 0 {
            self.pressed[word] |= mask;
            self.just_pressed[word] |= mask;
        }
    }

    /// Records a button release event.
    #[inline]
    pub fn release(&mut self, key_code: u16) {
        let bit = (key_code as usize) % (WORDS * 64);
        let word = bit / 64;
        let mask = 1u64 << (bit % 64);

        if (self.pressed[word] & mask) != 0 {
            self.pressed[word] &= !mask;
            self.just_released[word] |= mask;
        }
    }

    /// Tests whether the button is currently held down.
    #[inline]
    pub fn is_pressed(&self, key_code: u16) -> bool {
        let bit = (key_code as usize) % (WORDS * 64);
        (self.pressed[bit / 64] & (1u64 << (bit % 64))) != 0
    }

    /// Tests whether the button was pressed during this frame.
    #[inline]
    pub fn is_just_pressed(&self, key_code: u16) -> bool {
        let bit = (key_code as usize) % (WORDS * 64);
        (self.just_pressed[bit / 64] & (1u64 << (bit % 64))) != 0
    }

    /// Tests whether the button was released during this frame.
    #[inline]
    pub fn is_just_released(&self, key_code: u16) -> bool {
        let bit = (key_code as usize) % (WORDS * 64);
        (self.just_released[bit / 64] & (1u64 << (bit % 64))) != 0
    }

    /// Clears all button states.
    pub fn reset(&mut self) {
        self.pressed = [0; WORDS];
        self.just_pressed = [0; WORDS];
        self.just_released = [0; WORDS];
    }
}
