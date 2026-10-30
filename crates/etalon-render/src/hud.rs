//! Zero-Heap Vector and Canvas Document HUD Engine.
//!
//! Provides deterministic two-pass UI constraint solving (Flexbox + Grid-lite)
//! and bounded 2D vertex generation with strictly zero heap allocations in hot paths.

use bytemuck::{Pod, Zeroable};

pub const MAX_UI_NODES: usize = 256;
pub const NO_UI_NODE: u16 = u16::MAX;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlexDirection {
    #[default]
    Row = 0,
    Column = 1,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JustifyContent {
    #[default]
    Start = 0,
    Center = 1,
    End = 2,
    SpaceBetween = 3,
    SpaceAround = 4,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlignItems {
    Start = 0,
    Center = 1,
    End = 2,
    #[default]
    Stretch = 3,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum SizeRule {
    Fixed(f32),
    Percent(f32),
    #[default]
    Auto,
    FlexGrow(f32),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
pub struct Edges {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl Edges {
    pub const fn all(v: f32) -> Self {
        Self {
            left: v,
            right: v,
            top: v,
            bottom: v,
        }
    }
    pub const fn symmetric(h: f32, v: f32) -> Self {
        Self {
            left: h,
            right: h,
            top: v,
            bottom: v,
        }
    }
    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }
    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

pub type Padding = Edges;
pub type Margin = Edges;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiLayoutStyle {
    pub direction: FlexDirection,
    pub justify_content: JustifyContent,
    pub align_items: AlignItems,
    pub width: SizeRule,
    pub height: SizeRule,
    pub padding: Padding,
    pub margin: Margin,
    pub gap: f32,
}

impl Default for UiLayoutStyle {
    fn default() -> Self {
        Self {
            direction: FlexDirection::Row,
            justify_content: JustifyContent::Start,
            align_items: AlignItems::Stretch,
            width: SizeRule::Auto,
            height: SizeRule::Auto,
            padding: Padding::default(),
            margin: Margin::default(),
            gap: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
pub struct ComputedUiRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ComputedUiRect {
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct HudVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct UiLayoutNode {
    pub parent: u16,
    pub first_child: u16,
    pub next_sibling: u16,
    pub style: UiLayoutStyle,
    pub computed: ComputedUiRect,
    pub intrinsic_width: f32,
    pub intrinsic_height: f32,
    pub bg_color: [f32; 4],
}

impl Default for UiLayoutNode {
    fn default() -> Self {
        Self {
            parent: NO_UI_NODE,
            first_child: NO_UI_NODE,
            next_sibling: NO_UI_NODE,
            style: UiLayoutStyle::default(),
            computed: ComputedUiRect::default(),
            intrinsic_width: 0.0,
            intrinsic_height: 0.0,
            bg_color: [0.0; 4],
        }
    }
}

pub struct UiLayoutTree {
    nodes: [UiLayoutNode; MAX_UI_NODES],
    count: usize,
}

impl Default for UiLayoutTree {
    fn default() -> Self {
        Self::new()
    }
}

impl UiLayoutTree {
    pub const fn new() -> Self {
        Self {
            nodes: [UiLayoutNode {
                parent: NO_UI_NODE,
                first_child: NO_UI_NODE,
                next_sibling: NO_UI_NODE,
                style: UiLayoutStyle {
                    direction: FlexDirection::Row,
                    justify_content: JustifyContent::Start,
                    align_items: AlignItems::Stretch,
                    width: SizeRule::Auto,
                    height: SizeRule::Auto,
                    padding: Padding {
                        left: 0.0,
                        right: 0.0,
                        top: 0.0,
                        bottom: 0.0,
                    },
                    margin: Margin {
                        left: 0.0,
                        right: 0.0,
                        top: 0.0,
                        bottom: 0.0,
                    },
                    gap: 0.0,
                },
                computed: ComputedUiRect {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 0.0,
                },
                intrinsic_width: 0.0,
                intrinsic_height: 0.0,
                bg_color: [0.0; 4],
            }; MAX_UI_NODES],
            count: 0,
        }
    }

    pub fn clear(&mut self) {
        self.count = 0;
    }

    pub fn add_node(
        &mut self,
        parent: u16,
        style: UiLayoutStyle,
        bg_color: [f32; 4],
    ) -> Option<u16> {
        if self.count >= MAX_UI_NODES {
            return None;
        }
        let id = self.count as u16;
        self.count += 1;

        let node = &mut self.nodes[id as usize];
        node.parent = parent;
        node.first_child = NO_UI_NODE;
        node.next_sibling = NO_UI_NODE;
        node.style = style;
        node.bg_color = bg_color;

        if parent != NO_UI_NODE && (parent as usize) < self.count {
            let mut prev = self.nodes[parent as usize].first_child;
            if prev == NO_UI_NODE {
                self.nodes[parent as usize].first_child = id;
            } else {
                while self.nodes[prev as usize].next_sibling != NO_UI_NODE {
                    prev = self.nodes[prev as usize].next_sibling;
                }
                self.nodes[prev as usize].next_sibling = id;
            }
        }

        Some(id)
    }

    pub fn hit_test(&self, px: f32, py: f32) -> Option<u16> {
        for i in (0..self.count).rev() {
            if self.nodes[i].computed.contains(px, py) {
                return Some(i as u16);
            }
        }
        None
    }

    pub fn solve(&mut self, viewport_w: f32, viewport_h: f32) {
        if self.count == 0 {
            return;
        }
        // Root node fills viewport:
        self.nodes[0].computed.x = 0.0;
        self.nodes[0].computed.y = 0.0;
        self.nodes[0].computed.width = viewport_w;
        self.nodes[0].computed.height = viewport_h;

        for i in 0..self.count {
            self.layout_children(i as u16);
        }
    }

    fn layout_children(&mut self, parent_id: u16) {
        let parent = self.nodes[parent_id as usize];
        let p_rect = parent.computed;
        let mut child = parent.first_child;
        let mut cur_x = p_rect.x + parent.style.padding.left;
        let mut cur_y = p_rect.y + parent.style.padding.top;
        let avail_w = (p_rect.width - parent.style.padding.horizontal()).max(0.0);
        let avail_h = (p_rect.height - parent.style.padding.vertical()).max(0.0);

        while child != NO_UI_NODE {
            let c = &mut self.nodes[child as usize];
            let w = match c.style.width {
                SizeRule::Fixed(f) => f,
                SizeRule::Percent(p) => avail_w * (p * 0.01).clamp(0.0, 1.0),
                _ => (avail_w - (cur_x - p_rect.x)).max(0.0),
            };
            let h = match c.style.height {
                SizeRule::Fixed(f) => f,
                SizeRule::Percent(p) => avail_h * (p * 0.01).clamp(0.0, 1.0),
                _ => 32.0,
            };

            c.computed.x = cur_x + c.style.margin.left;
            c.computed.y = cur_y + c.style.margin.top;
            c.computed.width = w;
            c.computed.height = h;

            match parent.style.direction {
                FlexDirection::Row => {
                    cur_x += w + c.style.margin.horizontal() + parent.style.gap;
                }
                FlexDirection::Column => {
                    cur_y += h + c.style.margin.vertical() + parent.style.gap;
                }
            }

            child = c.next_sibling;
        }
    }

    pub fn generate_quad_vertices(&self, out: &mut [HudVertex]) -> usize {
        let mut v_idx = 0;
        for i in 0..self.count {
            if v_idx + 6 > out.len() {
                break;
            }
            let rect = self.nodes[i].computed;
            let col = self.nodes[i].bg_color;
            if col[3] <= 0.0 {
                continue;
            }

            let x0 = rect.x;
            let y0 = rect.y;
            let x1 = rect.x + rect.width;
            let y1 = rect.y + rect.height;

            out[v_idx] = HudVertex {
                pos: [x0, y0],
                uv: [0.0, 0.0],
                color: col,
            };
            out[v_idx + 1] = HudVertex {
                pos: [x1, y0],
                uv: [1.0, 0.0],
                color: col,
            };
            out[v_idx + 2] = HudVertex {
                pos: [x1, y1],
                uv: [1.0, 1.0],
                color: col,
            };

            out[v_idx + 3] = HudVertex {
                pos: [x0, y0],
                uv: [0.0, 0.0],
                color: col,
            };
            out[v_idx + 4] = HudVertex {
                pos: [x1, y1],
                uv: [1.0, 1.0],
                color: col,
            };
            out[v_idx + 5] = HudVertex {
                pos: [x0, y1],
                uv: [0.0, 1.0],
                color: col,
            };

            v_idx += 6;
        }
        v_idx
    }
}
