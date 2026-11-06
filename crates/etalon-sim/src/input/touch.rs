//! Mobile Virtual Touch Controller & Multi-Touch Gestures.
//!
//! Provides responsive multi-touch virtual joysticks and touch contact tracking
//! for mobile devices and tablets without heap allocation.

pub const MAX_TOUCH_POINTS: usize = 10;

/// Touch contact point record for multi-touch finger gesture tracking.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TouchContact {
    pub id: u64,
    pub pos: [f32; 2],
    pub delta: [f32; 2],
    pub is_active: bool,
}

/// Virtual touch joystick for mobile phones and touchscreen devices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VirtualTouchStick {
    pub center_px: [f32; 2],
    pub current_px: [f32; 2],
    pub radius_px: f32,
    pub deadzone: f32,
    pub axis: [f32; 2], // [-1.0..1.0, -1.0..1.0]
    pub is_active: bool,
    pub active_touch_id: Option<u64>,
}

impl Default for VirtualTouchStick {
    fn default() -> Self {
        Self {
            center_px: [100.0, 100.0],
            current_px: [100.0, 100.0],
            radius_px: 60.0,
            deadzone: 0.1,
            axis: [0.0, 0.0],
            is_active: false,
            active_touch_id: None,
        }
    }
}

impl VirtualTouchStick {
    pub const fn new(center_px: [f32; 2], radius_px: f32) -> Self {
        Self {
            center_px,
            current_px: center_px,
            radius_px,
            deadzone: 0.1,
            axis: [0.0, 0.0],
            is_active: false,
            active_touch_id: None,
        }
    }

    pub fn on_touch_start(&mut self, touch_id: u64, x: f32, y: f32) -> bool {
        if self.is_active {
            return false;
        }

        let dx = x - self.center_px[0];
        let dy = y - self.center_px[1];
        let dist = (dx * dx + dy * dy).sqrt();

        // Allow touch start within 1.5x radius of center
        if dist <= self.radius_px * 1.5 {
            self.is_active = true;
            self.active_touch_id = Some(touch_id);
            self.update_drag(x, y);
            true
        } else {
            false
        }
    }

    pub fn on_touch_move(&mut self, touch_id: u64, x: f32, y: f32) -> bool {
        if self.is_active && self.active_touch_id == Some(touch_id) {
            self.update_drag(x, y);
            true
        } else {
            false
        }
    }

    pub fn on_touch_end(&mut self, touch_id: u64) -> bool {
        if self.is_active && self.active_touch_id == Some(touch_id) {
            self.is_active = false;
            self.active_touch_id = None;
            self.current_px = self.center_px;
            self.axis = [0.0, 0.0];
            true
        } else {
            false
        }
    }

    fn update_drag(&mut self, x: f32, y: f32) {
        let dx = x - self.center_px[0];
        let dy = y - self.center_px[1];
        let dist = (dx * dx + dy * dy).sqrt();

        if dist <= 1e-4 {
            self.current_px = self.center_px;
            self.axis = [0.0, 0.0];
            return;
        }

        let clamped_dist = dist.min(self.radius_px);
        let nx = dx / dist;
        let ny = dy / dist;

        self.current_px = [
            self.center_px[0] + nx * clamped_dist,
            self.center_px[1] + ny * clamped_dist,
        ];

        let norm_mag = clamped_dist / self.radius_px;
        if norm_mag <= self.deadzone {
            self.axis = [0.0, 0.0];
        } else {
            // Remap deadzone to 0..1
            let remapped = (norm_mag - self.deadzone) / (1.0 - self.deadzone);
            self.axis = [nx * remapped, ny * remapped];
        }
    }
}

/// Fixed 10-slot multi-touch tracker without heap allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiTouchTracker {
    pub touches: [TouchContact; MAX_TOUCH_POINTS],
}

impl Default for MultiTouchTracker {
    fn default() -> Self {
        Self {
            touches: [TouchContact::default(); MAX_TOUCH_POINTS],
        }
    }
}

impl MultiTouchTracker {
    pub const fn new() -> Self {
        Self {
            touches: [TouchContact {
                id: 0,
                pos: [0.0, 0.0],
                delta: [0.0, 0.0],
                is_active: false,
            }; MAX_TOUCH_POINTS],
        }
    }

    pub fn touch_start(&mut self, id: u64, x: f32, y: f32) {
        for slot in self.touches.iter_mut() {
            if !slot.is_active {
                slot.id = id;
                slot.pos = [x, y];
                slot.delta = [0.0, 0.0];
                slot.is_active = true;
                return;
            }
        }
    }

    pub fn touch_move(&mut self, id: u64, x: f32, y: f32) {
        for slot in self.touches.iter_mut() {
            if slot.is_active && slot.id == id {
                slot.delta = [x - slot.pos[0], y - slot.pos[1]];
                slot.pos = [x, y];
                return;
            }
        }
    }

    pub fn touch_end(&mut self, id: u64) {
        for slot in self.touches.iter_mut() {
            if slot.is_active && slot.id == id {
                slot.is_active = false;
                slot.delta = [0.0, 0.0];
                return;
            }
        }
    }

    pub fn active_count(&self) -> usize {
        self.touches.iter().filter(|t| t.is_active).count()
    }
}
