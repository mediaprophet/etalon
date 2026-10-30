use super::{InitTracker, MemoryInitKind};
use crate::resource::{Texture, TextureView};
use alloc::{string::String, sync::Arc, vec::Vec};
use arrayvec::ArrayVec;
use core::num::NonZeroU32;
use core::ops::Range;

#[derive(Debug, Clone)]
pub(crate) struct TextureInitRange {
    pub(crate) mip_range: Range<u32>,
    // Strictly array layers. We do *not* track volume slices separately.
    pub(crate) layer_range: Range<u32>,
}

impl TextureInitRange {
    pub(crate) fn active_layer_ranges(
        self,
        multiview_mask: Option<NonZeroU32>,
    ) -> ArrayVec<Self, 32> {
        let mut ranges = ArrayVec::new();
        let Some(mask) = multiview_mask else {
            ranges.push(self);
            return ranges;
        };
        let mut views = mask.get();
        while views != 0 {
            let view = views.trailing_zeros();
            views &= views - 1;
            if let Some(layer) = self
                .layer_range
                .start
                .checked_add(view)
                .filter(|&layer| layer < self.layer_range.end)
            {
                ranges.push(Self {
                    mip_range: self.mip_range.clone(),
                    layer_range: layer..layer + 1,
                });
            }
        }
        ranges
    }
}

// Returns true if a copy operation doesn't fully cover the texture init
// tracking granularity. I.e. if this function returns true for a pending copy
// operation, the target texture needs to be ensured to be initialized first!
pub(crate) fn has_copy_partial_init_tracker_coverage<T>(
    copy_size: &wgt::Extent3d,
    copy_info: &wgt::TexelCopyTextureInfo<T>,
    desc: &wgt::TextureDescriptor<String, Vec<wgt::TextureFormat>>,
) -> bool {
    let target_size = desc.mip_level_size(copy_info.mip_level).unwrap();
    copy_size.width != target_size.width
        || copy_size.height != target_size.height
        || (desc.dimension == wgt::TextureDimension::D3
            && copy_size.depth_or_array_layers != target_size.depth_or_array_layers)
        || copy_info.aspect != wgt::TextureAspect::All
}

impl From<&'_ TextureView> for TextureInitRange {
    fn from(view: &'_ TextureView) -> Self {
        // TextureInitRange is always array layers, never depth slices.
        let layer_range = if view.parent.desc.dimension == wgt::TextureDimension::D3 {
            0..1
        } else {
            view.selector.layers.clone()
        };
        TextureInitRange {
            mip_range: view.selector.mips.clone(),
            layer_range,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TextureInitTrackerAction {
    pub(crate) texture: Arc<Texture>,
    pub(crate) range: TextureInitRange,
    pub(crate) kind: MemoryInitKind,
}

pub(crate) type TextureLayerInitTracker = InitTracker<u32>;

#[derive(Debug)]
pub(crate) struct TextureInitTracker {
    pub mips: ArrayVec<TextureLayerInitTracker, { hal::MAX_MIP_LEVELS as usize }>,
}

impl TextureInitTracker {
    pub(crate) fn new(mip_level_count: u32, depth_or_array_layers: u32) -> Self {
        TextureInitTracker {
            mips: core::iter::repeat_n(
                TextureLayerInitTracker::new(depth_or_array_layers),
                mip_level_count as usize,
            )
            .collect(),
        }
    }

    pub(crate) fn check_action(
        &self,
        action: &TextureInitTrackerAction,
    ) -> Option<TextureInitTrackerAction> {
        let mut mip_range_start = usize::MAX;
        let mut mip_range_end = usize::MIN;
        let mut layer_range_start = u32::MAX;
        let mut layer_range_end = u32::MIN;

        for (i, mip_tracker) in self
            .mips
            .iter()
            .enumerate()
            .take(action.range.mip_range.end as usize)
            .skip(action.range.mip_range.start as usize)
        {
            if let Some(uninitialized_layer_range) =
                mip_tracker.check(action.range.layer_range.clone())
            {
                mip_range_start = mip_range_start.min(i);
                mip_range_end = i + 1;
                layer_range_start = layer_range_start.min(uninitialized_layer_range.start);
                layer_range_end = layer_range_end.max(uninitialized_layer_range.end);
            };
        }

        if mip_range_start < mip_range_end && layer_range_start < layer_range_end {
            Some(TextureInitTrackerAction {
                texture: action.texture.clone(),
                range: TextureInitRange {
                    mip_range: mip_range_start as u32..mip_range_end as u32,
                    layer_range: layer_range_start..layer_range_end,
                },
                kind: action.kind,
            })
        } else {
            None
        }
    }

    pub(crate) fn discard(&mut self, mip_level: u32, layer: u32) {
        self.mips[mip_level as usize].discard(layer);
    }
}

#[cfg(test)]
mod tests {
    use super::{TextureInitRange, TextureInitTracker};
    use core::num::NonZeroU32;

    #[test]
    fn attachment_writes_initialize_only_active_layers_at_the_view_base() {
        for (mask, written) in [(1, &[1][..]), (9, &[1, 4][..]), (15, &[1, 2, 3, 4][..])] {
            let mut tracker = TextureInitTracker::new(3, 6);
            let view = TextureInitRange {
                mip_range: 1..2,
                layer_range: 1..5,
            };
            for range in view.active_layer_ranges(NonZeroU32::new(mask)) {
                for mip in range.mip_range {
                    tracker.mips[mip as usize]
                        .drain(range.layer_range.clone())
                        .for_each(drop);
                }
            }
            for layer in 0..6 {
                assert_eq!(
                    tracker.mips[1].check(layer..layer + 1).is_none(),
                    written.contains(&layer)
                );
            }
            assert!(tracker.mips[0].check(0..6).is_some());
            assert!(tracker.mips[2].check(0..6).is_some());
        }
    }

    #[test]
    fn attachment_discard_invalidates_all_active_layers_and_preserves_inactive_data() {
        for mask in [1, 9, 15] {
            let mut tracker = TextureInitTracker::new(1, 6);
            tracker.mips[0].drain(0..6).for_each(drop);
            let view = TextureInitRange {
                mip_range: 0..1,
                layer_range: 1..5,
            };
            for range in view.active_layer_ranges(NonZeroU32::new(mask)) {
                for layer in range.layer_range {
                    tracker.discard(0, layer);
                }
            }
            for layer in 0..6 {
                let discarded = (1..5).contains(&layer) && (mask & (1 << (layer - 1))) != 0;
                assert_eq!(tracker.mips[0].check(layer..layer + 1).is_some(), discarded);
            }
        }
    }

    #[test]
    fn ordinary_and_invalid_attachment_ranges_keep_existing_bounds() {
        let view = TextureInitRange {
            mip_range: 0..1,
            layer_range: 1..5,
        };
        let ordinary = view.clone().active_layer_ranges(None);
        assert_eq!(ordinary.len(), 1);
        assert_eq!(ordinary[0].layer_range, 1..5);
        assert!(view
            .active_layer_ranges(NonZeroU32::new(1 << 31))
            .is_empty());
        let view = TextureInitRange {
            mip_range: 0..1,
            layer_range: u32::MAX - 1..u32::MAX,
        };
        assert!(view
            .active_layer_ranges(NonZeroU32::new(1 << 31))
            .is_empty());
    }
}
