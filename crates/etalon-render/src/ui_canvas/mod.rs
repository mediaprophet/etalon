//! Declarative Vector UI Canvas, MSDF Font Shaper, and Zero-Heap Flexbox/Grid Layout Solver (`ui_canvas`).

pub mod msdf_text;
pub mod widgets;

pub use crate::hud::{
    AlignItems, Edges, FlexDirection, HudVertex, JustifyContent, Margin, Padding, SizeRule,
    MAX_UI_NODES, NO_UI_NODE,
};
pub use msdf_text::*;
pub use widgets::*;

/// Alias to the compact 256-node HUD layout tree in `crate::hud`.
pub type HudUiLayoutTree = crate::hud::UiLayoutTree;
/// Alias to the compact HUD computed rect in `crate::hud`.
pub type HudComputedUiRect = crate::hud::ComputedUiRect;

use bytemuck::{Pod, Zeroable};

/// Maximum layout nodes supported in a single zero-heap UI layout tree.
pub const MAX_UI_LAYOUT_NODES: usize = 512;

#[inline]
fn normalized_fraction(p: f32) -> f32 {
    if p > 1.0 {
        (p * 0.01).clamp(0.0, 1.0)
    } else {
        p.clamp(0.0, 1.0)
    }
}

/// Layout style properties for a [`UiLayoutNode`] (supports Flexbox + 2D CSS-Grid-lite + scissor clipping).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiLayoutStyle {
    pub direction: FlexDirection,
    pub justify_content: JustifyContent,
    pub align_items: AlignItems,
    pub width: SizeRule,
    pub height: SizeRule,
    pub min_width: f32,
    pub min_height: f32,
    pub padding: Padding,
    pub margin: Margin,
    /// Gap between items in a flex row/column.
    pub gap: f32,
    /// When `grid_columns > 0`, arranges children in a 2D grid with `grid_columns` columns.
    pub grid_columns: u8,
    pub column_gap: f32,
    pub row_gap: f32,
    /// Whether this container clips its descendants to its inner content rect.
    pub clip_children: bool,
}

impl Default for UiLayoutStyle {
    fn default() -> Self {
        Self {
            direction: FlexDirection::Column,
            justify_content: JustifyContent::Start,
            align_items: AlignItems::Stretch,
            width: SizeRule::Auto,
            height: SizeRule::Auto,
            min_width: 0.0,
            min_height: 0.0,
            padding: Padding::default(),
            margin: Margin::default(),
            gap: 0.0,
            grid_columns: 0,
            column_gap: 0.0,
            row_gap: 0.0,
            clip_children: false,
        }
    }
}

/// Resolved screen-space rectangle and scissor clip rectangle for a layout node.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct ComputedUiRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// Scissor clip rectangle `[min_x, min_y, max_x, max_y]`.
    pub clip_rect: [f32; 4],
}

impl Default for ComputedUiRect {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            clip_rect: [0.0, 0.0, 16384.0, 16384.0],
        }
    }
}

impl From<crate::hud::ComputedUiRect> for ComputedUiRect {
    fn from(r: crate::hud::ComputedUiRect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
            clip_rect: [0.0, 0.0, 16384.0, 16384.0],
        }
    }
}

/// Single node in the zero-heap intrusive UI layout tree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiLayoutNode {
    /// External widget ID in `ZeroHeapUiCanvas`.
    pub widget_id: u32,
    /// Intrusive parent index (`NO_UI_NODE` for roots).
    pub parent: u16,
    /// First child index (`NO_UI_NODE` if leaf).
    pub first_child: u16,
    /// Next sibling index (`NO_UI_NODE` if last child).
    pub next_sibling: u16,
    /// Layout style constraints.
    pub style: UiLayoutStyle,
    /// Intrinsic content measurement (e.g., from shaped MSDF text run).
    pub intrinsic_size: [f32; 2],
    /// Bottom-up measured desired size `[width, height]` (excluding outer margins).
    pub measured_size: [f32; 2],
    /// Final top-down resolved screen-space rectangle and clip rect.
    pub computed: ComputedUiRect,
}

impl Default for UiLayoutNode {
    fn default() -> Self {
        Self {
            widget_id: 0,
            parent: NO_UI_NODE,
            first_child: NO_UI_NODE,
            next_sibling: NO_UI_NODE,
            style: UiLayoutStyle::default(),
            intrinsic_size: [0.0, 0.0],
            measured_size: [0.0, 0.0],
            computed: ComputedUiRect::default(),
        }
    }
}

/// Zero-heap two-pass (`measure` + `arrange`) Flexbox & 2D Grid layout solver.
#[derive(Debug, Clone)]
pub struct ZeroHeapUiLayoutTree {
    pub nodes: [UiLayoutNode; MAX_UI_LAYOUT_NODES],
    pub count: usize,
}

/// Type alias so callers can refer to `UiLayoutTree` or `ZeroHeapUiLayoutTree` interchangeably.
pub type UiLayoutTree = ZeroHeapUiLayoutTree;

impl Default for ZeroHeapUiLayoutTree {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroHeapUiLayoutTree {
    /// Creates an empty zero-heap layout tree.
    pub fn new() -> Self {
        Self {
            nodes: [UiLayoutNode::default(); MAX_UI_LAYOUT_NODES],
            count: 0,
        }
    }

    /// Resets the tree for a new frame.
    #[inline]
    pub fn clear(&mut self) {
        self.count = 0;
    }

    /// Allocates a new layout node and optionally attaches it to `parent` (`NO_UI_NODE` for root).
    pub fn add_node(
        &mut self,
        widget_id: u32,
        parent: u16,
        style: UiLayoutStyle,
        intrinsic_size: [f32; 2],
    ) -> Option<u16> {
        if self.count >= MAX_UI_LAYOUT_NODES {
            return None;
        }
        let idx = self.count as u16;
        self.nodes[self.count] = UiLayoutNode {
            widget_id,
            parent,
            first_child: NO_UI_NODE,
            next_sibling: NO_UI_NODE,
            style,
            intrinsic_size,
            measured_size: [0.0, 0.0],
            computed: ComputedUiRect::default(),
        };
        self.count += 1;

        if parent != NO_UI_NODE && (parent as usize) < self.count - 1 {
            let p = parent as usize;
            if self.nodes[p].first_child == NO_UI_NODE {
                self.nodes[p].first_child = idx;
            } else {
                let mut cur = self.nodes[p].first_child as usize;
                let mut guard = 0usize;
                while self.nodes[cur].next_sibling != NO_UI_NODE && guard < MAX_UI_LAYOUT_NODES {
                    cur = self.nodes[cur].next_sibling as usize;
                    guard += 1;
                }
                self.nodes[cur].next_sibling = idx;
            }
        }
        Some(idx)
    }

    /// Solves the entire layout tree for a root viewport of `(viewport_width, viewport_height)`
    /// in two zero-heap passes:
    /// 1. Bottom-up `measure` pass computing intrinsic/auto sizes.
    /// 2. Top-down `arrange` pass resolving `FlexGrow`, `Percent`, `JustifyContent`, `AlignItems`,
    ///    2D `grid_columns`, and hierarchical `clip_rect`s.
    pub fn solve(&mut self, viewport_width: f32, viewport_height: f32) {
        if self.count == 0 {
            return;
        }

        let mut order = [0u16; MAX_UI_LAYOUT_NODES];
        let mut order_len = 0usize;
        let mut stack = [0u16; MAX_UI_LAYOUT_NODES];
        let mut stack_len = 0usize;

        for i in (0..self.count).rev() {
            if self.nodes[i].parent == NO_UI_NODE && stack_len < MAX_UI_LAYOUT_NODES {
                stack[stack_len] = i as u16;
                stack_len += 1;
            }
        }

        while stack_len > 0 && order_len < MAX_UI_LAYOUT_NODES {
            stack_len -= 1;
            let idx = stack[stack_len];
            order[order_len] = idx;
            order_len += 1;

            let mut children = [0u16; 64];
            let mut c_count = 0usize;
            let mut child = self.nodes[idx as usize].first_child;
            while child != NO_UI_NODE && c_count < 64 {
                children[c_count] = child;
                c_count += 1;
                child = self.nodes[child as usize].next_sibling;
            }
            for k in (0..c_count).rev() {
                if stack_len < MAX_UI_LAYOUT_NODES {
                    stack[stack_len] = children[k];
                    stack_len += 1;
                }
            }
        }

        // PASS 1: Bottom-up `measure` pass
        for k in (0..order_len).rev() {
            let idx = order[k] as usize;
            self.measure_node(idx);
        }

        // PASS 2: Top-down `arrange` pass
        let root_clip = [0.0, 0.0, viewport_width.max(0.0), viewport_height.max(0.0)];
        for &idx_u16 in &order[..order_len] {
            let idx = idx_u16 as usize;
            if self.nodes[idx].parent == NO_UI_NODE {
                let style = self.nodes[idx].style;
                let m = style.margin;
                let avail_w = (viewport_width - m.horizontal()).max(0.0);
                let avail_h = (viewport_height - m.vertical()).max(0.0);
                let w = match style.width {
                    SizeRule::Fixed(v) => v,
                    SizeRule::Percent(p) => avail_w * normalized_fraction(p),
                    SizeRule::Auto => self.nodes[idx].measured_size[0],
                    SizeRule::FlexGrow(_) => avail_w,
                }
                .max(style.min_width);
                let h = match style.height {
                    SizeRule::Fixed(v) => v,
                    SizeRule::Percent(p) => avail_h * normalized_fraction(p),
                    SizeRule::Auto => self.nodes[idx].measured_size[1],
                    SizeRule::FlexGrow(_) => avail_h,
                }
                .max(style.min_height);

                self.nodes[idx].computed = ComputedUiRect {
                    x: m.left,
                    y: m.top,
                    width: w,
                    height: h,
                    clip_rect: root_clip,
                };
            }
            self.arrange_children(idx);
        }
    }

    fn measure_node(&mut self, idx: usize) {
        let style = self.nodes[idx].style;
        let mut child_idx = self.nodes[idx].first_child;

        let mut content_w = self.nodes[idx].intrinsic_size[0];
        let mut content_h = self.nodes[idx].intrinsic_size[1];

        if child_idx != NO_UI_NODE {
            if style.grid_columns > 0 {
                let cols = style.grid_columns as usize;
                let col_gap = if style.column_gap > 0.0 {
                    style.column_gap
                } else {
                    style.gap
                };
                let row_gap = if style.row_gap > 0.0 {
                    style.row_gap
                } else {
                    style.gap
                };
                let mut col = 0usize;
                let mut row_w = 0.0_f32;
                let mut row_h = 0.0_f32;
                let mut max_w = 0.0_f32;
                let mut total_h = 0.0_f32;
                let mut rows = 0usize;

                while child_idx != NO_UI_NODE {
                    let c = &self.nodes[child_idx as usize];
                    let cw = c.measured_size[0] + c.style.margin.horizontal();
                    let ch = c.measured_size[1] + c.style.margin.vertical();
                    if col > 0 {
                        row_w += col_gap;
                    }
                    row_w += cw;
                    row_h = row_h.max(ch);
                    col += 1;
                    if col >= cols {
                        max_w = max_w.max(row_w);
                        if rows > 0 {
                            total_h += row_gap;
                        }
                        total_h += row_h;
                        rows += 1;
                        col = 0;
                        row_w = 0.0;
                        row_h = 0.0;
                    }
                    child_idx = c.next_sibling;
                }
                if col > 0 {
                    max_w = max_w.max(row_w);
                    if rows > 0 {
                        total_h += row_gap;
                    }
                    total_h += row_h;
                }
                content_w = content_w.max(max_w);
                content_h = content_h.max(total_h);
            } else {
                let mut sum_main = 0.0_f32;
                let mut max_cross = 0.0_f32;
                let mut count = 0usize;
                while child_idx != NO_UI_NODE {
                    let c = &self.nodes[child_idx as usize];
                    let cw = c.measured_size[0] + c.style.margin.horizontal();
                    let ch = c.measured_size[1] + c.style.margin.vertical();
                    match style.direction {
                        FlexDirection::Row => {
                            sum_main += cw;
                            max_cross = max_cross.max(ch);
                        }
                        FlexDirection::Column => {
                            sum_main += ch;
                            max_cross = max_cross.max(cw);
                        }
                    }
                    count += 1;
                    child_idx = c.next_sibling;
                }
                if count > 1 {
                    sum_main += style.gap * ((count - 1) as f32);
                }
                match style.direction {
                    FlexDirection::Row => {
                        content_w = content_w.max(sum_main);
                        content_h = content_h.max(max_cross);
                    }
                    FlexDirection::Column => {
                        content_w = content_w.max(max_cross);
                        content_h = content_h.max(sum_main);
                    }
                }
            }
        }

        let pad_w = content_w + style.padding.horizontal();
        let pad_h = content_h + style.padding.vertical();

        let measured_w = match style.width {
            SizeRule::Fixed(v) => v,
            _ => pad_w,
        }
        .max(style.min_width);

        let measured_h = match style.height {
            SizeRule::Fixed(v) => v,
            _ => pad_h,
        }
        .max(style.min_height);

        self.nodes[idx].measured_size = [measured_w, measured_h];
    }

    fn arrange_children(&mut self, parent_idx: usize) {
        let first = self.nodes[parent_idx].first_child;
        if first == NO_UI_NODE {
            return;
        }

        let p_style = self.nodes[parent_idx].style;
        let p_rect = self.nodes[parent_idx].computed;
        let inner_x = p_rect.x + p_style.padding.left;
        let inner_y = p_rect.y + p_style.padding.top;
        let inner_w = (p_rect.width - p_style.padding.horizontal()).max(0.0);
        let inner_h = (p_rect.height - p_style.padding.vertical()).max(0.0);

        let child_clip = if p_style.clip_children {
            [
                inner_x.max(p_rect.clip_rect[0]),
                inner_y.max(p_rect.clip_rect[1]),
                (inner_x + inner_w).min(p_rect.clip_rect[2]),
                (inner_y + inner_h).min(p_rect.clip_rect[3]),
            ]
        } else {
            p_rect.clip_rect
        };

        let mut children = [0u16; 128];
        let mut child_count = 0usize;
        let mut cur = first;
        while cur != NO_UI_NODE && child_count < 128 {
            children[child_count] = cur;
            child_count += 1;
            cur = self.nodes[cur as usize].next_sibling;
        }

        if p_style.grid_columns > 0 {
            let cols = (p_style.grid_columns as usize).max(1);
            let col_gap = if p_style.column_gap > 0.0 {
                p_style.column_gap
            } else {
                p_style.gap
            };
            let row_gap = if p_style.row_gap > 0.0 {
                p_style.row_gap
            } else {
                p_style.gap
            };
            let total_col_gaps = col_gap * ((cols.saturating_sub(1)) as f32);
            let cell_w = ((inner_w - total_col_gaps) / (cols as f32)).max(0.0);

            let mut row_y = inner_y;
            let mut i = 0usize;
            while i < child_count {
                let row_end = (i + cols).min(child_count);
                let mut row_h = 0.0_f32;
                for &c_u16 in &children[i..row_end] {
                    let c = &self.nodes[c_u16 as usize];
                    let ch = match c.style.height {
                        SizeRule::Fixed(v) => v,
                        SizeRule::Percent(p) => inner_h * normalized_fraction(p),
                        _ => c.measured_size[1],
                    }
                    .max(c.style.min_height)
                        + c.style.margin.vertical();
                    row_h = row_h.max(ch);
                }

                for (col_idx, &c_u16) in children[i..row_end].iter().enumerate() {
                    let c_idx = c_u16 as usize;
                    let c_style = self.nodes[c_idx].style;
                    let m = c_style.margin;
                    let cell_x = inner_x + (col_idx as f32) * (cell_w + col_gap);
                    let w = match c_style.width {
                        SizeRule::Fixed(v) => v,
                        SizeRule::Percent(p) => {
                            (cell_w - m.horizontal()).max(0.0) * normalized_fraction(p)
                        }
                        SizeRule::Auto => self.nodes[c_idx].measured_size[0],
                        SizeRule::FlexGrow(_) => (cell_w - m.horizontal()).max(0.0),
                    }
                    .max(c_style.min_width);
                    let h = match c_style.height {
                        SizeRule::Fixed(v) => v,
                        SizeRule::Percent(p) => {
                            (row_h - m.vertical()).max(0.0) * normalized_fraction(p)
                        }
                        SizeRule::Auto => {
                            if p_style.align_items == AlignItems::Stretch {
                                (row_h - m.vertical()).max(0.0)
                            } else {
                                self.nodes[c_idx].measured_size[1]
                            }
                        }
                        SizeRule::FlexGrow(_) => (row_h - m.vertical()).max(0.0),
                    }
                    .max(c_style.min_height);

                    self.nodes[c_idx].computed = ComputedUiRect {
                        x: cell_x + m.left,
                        y: row_y + m.top,
                        width: w,
                        height: h,
                        clip_rect: child_clip,
                    };
                }

                row_y += row_h + row_gap;
                i = row_end;
            }
            return;
        }

        let is_row = p_style.direction == FlexDirection::Row;
        let main_avail = if is_row { inner_w } else { inner_h };
        let cross_avail = if is_row { inner_h } else { inner_w };
        let total_gaps = if child_count > 1 {
            p_style.gap * ((child_count - 1) as f32)
        } else {
            0.0
        };

        let mut fixed_main_sum = total_gaps;
        let mut total_flex_grow = 0.0_f32;
        let mut child_main = [0.0_f32; 128];
        let mut child_cross = [0.0_f32; 128];

        for (k, &c_u16) in children[..child_count].iter().enumerate() {
            let c = &self.nodes[c_u16 as usize];
            let m = c.style.margin;
            let main_rule = if is_row { c.style.width } else { c.style.height };
            let cross_rule = if is_row { c.style.height } else { c.style.width };
            let min_main = if is_row {
                c.style.min_width
            } else {
                c.style.min_height
            };
            let min_cross = if is_row {
                c.style.min_height
            } else {
                c.style.min_width
            };
            let main_margin = if is_row { m.horizontal() } else { m.vertical() };
            let cross_margin = if is_row { m.vertical() } else { m.horizontal() };

            match main_rule {
                SizeRule::Fixed(v) => {
                    let s = v.max(min_main);
                    child_main[k] = s;
                    fixed_main_sum += s + main_margin;
                }
                SizeRule::Percent(p) => {
                    let s = (main_avail * normalized_fraction(p)).max(min_main);
                    child_main[k] = s;
                    fixed_main_sum += s + main_margin;
                }
                SizeRule::Auto => {
                    let s = (if is_row {
                        c.measured_size[0]
                    } else {
                        c.measured_size[1]
                    })
                    .max(min_main);
                    child_main[k] = s;
                    fixed_main_sum += s + main_margin;
                }
                SizeRule::FlexGrow(weight) => {
                    let w = weight.max(0.0);
                    total_flex_grow += w;
                    fixed_main_sum += main_margin;
                }
            }

            let cross_s = match cross_rule {
                SizeRule::Fixed(v) => v,
                SizeRule::Percent(p) => {
                    (cross_avail - cross_margin).max(0.0) * normalized_fraction(p)
                }
                SizeRule::Auto => {
                    if p_style.align_items == AlignItems::Stretch {
                        (cross_avail - cross_margin).max(0.0)
                    } else if is_row {
                        c.measured_size[1]
                    } else {
                        c.measured_size[0]
                    }
                }
                SizeRule::FlexGrow(_) => (cross_avail - cross_margin).max(0.0),
            }
            .max(min_cross);
            child_cross[k] = cross_s;
        }

        let remaining_main = (main_avail - fixed_main_sum).max(0.0);
        if total_flex_grow > 1e-6 {
            for (k, &c_u16) in children[..child_count].iter().enumerate() {
                let c = &self.nodes[c_u16 as usize];
                let main_rule = if is_row { c.style.width } else { c.style.height };
                let min_main = if is_row {
                    c.style.min_width
                } else {
                    c.style.min_height
                };
                if let SizeRule::FlexGrow(weight) = main_rule {
                    child_main[k] =
                        (remaining_main * (weight.max(0.0) / total_flex_grow)).max(min_main);
                }
            }
        }

        let mut used_main = total_gaps;
        for (k, &c_u16) in children[..child_count].iter().enumerate() {
            let m = self.nodes[c_u16 as usize].style.margin;
            used_main += child_main[k] + if is_row { m.horizontal() } else { m.vertical() };
        }
        let free_space = (main_avail - used_main).max(0.0);

        let (mut cursor_main, extra_gap) = match p_style.justify_content {
            JustifyContent::Start => (0.0, 0.0),
            JustifyContent::Center => (free_space * 0.5, 0.0),
            JustifyContent::End => (free_space, 0.0),
            JustifyContent::SpaceBetween => {
                if child_count > 1 {
                    (0.0, free_space / ((child_count - 1) as f32))
                } else {
                    (0.0, 0.0)
                }
            }
            JustifyContent::SpaceAround => {
                if child_count > 0 {
                    let step = free_space / (child_count as f32);
                    (step * 0.5, step)
                } else {
                    (0.0, 0.0)
                }
            }
        };

        for (k, &c_u16) in children[..child_count].iter().enumerate() {
            let c_idx = c_u16 as usize;
            let m = self.nodes[c_idx].style.margin;
            let main_lead = if is_row { m.left } else { m.top };
            let main_trail = if is_row { m.right } else { m.bottom };
            let cross_lead = if is_row { m.top } else { m.left };
            let cross_total = child_cross[k] + if is_row { m.vertical() } else { m.horizontal() };

            let cross_offset = match p_style.align_items {
                AlignItems::Start | AlignItems::Stretch => cross_lead,
                AlignItems::Center => ((cross_avail - cross_total) * 0.5).max(0.0) + cross_lead,
                AlignItems::End => (cross_avail - cross_total).max(0.0) + cross_lead,
            };

            let pos_main = cursor_main + main_lead;
            let (x, y, w, h) = if is_row {
                (
                    inner_x + pos_main,
                    inner_y + cross_offset,
                    child_main[k],
                    child_cross[k],
                )
            } else {
                (
                    inner_x + cross_offset,
                    inner_y + pos_main,
                    child_cross[k],
                    child_main[k],
                )
            };

            self.nodes[c_idx].computed = ComputedUiRect {
                x,
                y,
                width: w,
                height: h,
                clip_rect: child_clip,
            };

            cursor_main += child_main[k] + main_lead + main_trail + p_style.gap + extra_gap;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flex_row_grow_and_padding_distributes_exact_widths() {
        let mut tree = ZeroHeapUiLayoutTree::new();
        let root = tree
            .add_node(
                1,
                NO_UI_NODE,
                UiLayoutStyle {
                    direction: FlexDirection::Row,
                    width: SizeRule::Fixed(400.0),
                    height: SizeRule::Fixed(100.0),
                    padding: Padding::all(10.0),
                    gap: 10.0,
                    align_items: AlignItems::Stretch,
                    ..Default::default()
                },
                [0.0, 0.0],
            )
            .unwrap();

        let c0 = tree
            .add_node(
                2,
                root,
                UiLayoutStyle {
                    width: SizeRule::Fixed(80.0),
                    height: SizeRule::Auto,
                    ..Default::default()
                },
                [0.0, 0.0],
            )
            .unwrap();
        let c1 = tree
            .add_node(
                3,
                root,
                UiLayoutStyle {
                    width: SizeRule::FlexGrow(1.0),
                    height: SizeRule::Auto,
                    ..Default::default()
                },
                [0.0, 0.0],
            )
            .unwrap();
        let c2 = tree
            .add_node(
                4,
                root,
                UiLayoutStyle {
                    width: SizeRule::FlexGrow(3.0),
                    height: SizeRule::Auto,
                    ..Default::default()
                },
                [0.0, 0.0],
            )
            .unwrap();

        tree.solve(400.0, 100.0);

        let r0 = tree.nodes[c0 as usize].computed;
        let r1 = tree.nodes[c1 as usize].computed;
        let r2 = tree.nodes[c2 as usize].computed;

        assert!((r0.x - 10.0).abs() < 1e-4);
        assert!((r0.width - 80.0).abs() < 1e-4);
        assert!((r0.height - 80.0).abs() < 1e-4);

        assert!((r1.x - 100.0).abs() < 1e-4);
        assert!((r1.width - 70.0).abs() < 1e-4);

        assert!((r2.x - 180.0).abs() < 1e-4);
        assert!((r2.width - 210.0).abs() < 1e-4);
    }

    #[test]
    fn grid_layout_arranges_2d_columns_and_rows() {
        let mut tree = ZeroHeapUiLayoutTree::new();
        let root = tree
            .add_node(
                10,
                NO_UI_NODE,
                UiLayoutStyle {
                    width: SizeRule::Fixed(310.0),
                    height: SizeRule::Fixed(200.0),
                    grid_columns: 3,
                    column_gap: 5.0,
                    row_gap: 10.0,
                    ..Default::default()
                },
                [0.0, 0.0],
            )
            .unwrap();

        let mut items = [0u16; 6];
        for i in 0..6 {
            items[i] = tree
                .add_node(
                    100 + i as u32,
                    root,
                    UiLayoutStyle {
                        width: SizeRule::FlexGrow(1.0),
                        height: SizeRule::Fixed(40.0),
                        ..Default::default()
                    },
                    [0.0, 0.0],
                )
                .unwrap();
        }

        tree.solve(310.0, 200.0);
        let cell0 = tree.nodes[items[0] as usize].computed;
        let cell1 = tree.nodes[items[1] as usize].computed;
        let cell3 = tree.nodes[items[3] as usize].computed;
        assert!((cell0.width - 100.0).abs() < 1e-4);
        assert!((cell1.x - 105.0).abs() < 1e-4);
        assert!((cell3.x - 0.0).abs() < 1e-4);
        assert!((cell3.y - 50.0).abs() < 1e-4);
    }
}
