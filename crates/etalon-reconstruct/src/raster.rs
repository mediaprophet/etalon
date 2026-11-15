use crate::error::ReconstructError;

#[derive(Clone, Debug)]
pub struct Raster {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Raster {
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, ReconstructError> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| ReconstructError::Decode("image dimensions overflow".into()))?;
        if rgba.len() != expected || width == 0 || height == 0 {
            return Err(ReconstructError::Decode(
                "RGBA buffer does not match width*height*4".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReconstructError> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| ReconstructError::Decode(e.to_string()))?
            .to_rgba8();
        let (width, height) = img.dimensions();
        Self::from_rgba(width, height, img.into_raw())
    }

    pub fn from_path(path: &std::path::Path) -> Result<Self, ReconstructError> {
        let bytes = std::fs::read(path).map_err(|e| ReconstructError::Io(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    pub fn pixel(&self, x: i32, y: i32) -> Option<[u8; 4]> {
        if x < 0 || y < 0 || x as u32 >= self.width || y as u32 >= self.height {
            return None;
        }
        let i = ((y as u32 * self.width + x as u32) * 4) as usize;
        Some([
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ])
    }

    pub fn sample_bilinear(&self, u: f32, v: f32) -> Option<[u8; 4]> {
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        let fx = u * (self.width.saturating_sub(1) as f32);
        let fy = v * (self.height.saturating_sub(1) as f32);
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let x1 = (x0 + 1).min(self.width as i32 - 1);
        let y1 = (y0 + 1).min(self.height as i32 - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let p00 = self.pixel(x0, y0)?;
        let p10 = self.pixel(x1, y0)?;
        let p01 = self.pixel(x0, y1)?;
        let p11 = self.pixel(x1, y1)?;
        let mut out = [0u8; 4];
        for c in 0..4 {
            let a = p00[c] as f32 * (1.0 - tx) + p10[c] as f32 * tx;
            let b = p01[c] as f32 * (1.0 - tx) + p11[c] as f32 * tx;
            out[c] = (a * (1.0 - ty) + b * ty).round() as u8;
        }
        Some(out)
    }
}

#[derive(Clone, Debug)]
pub struct Silhouette {
    pub width: u32,
    pub height: u32,
    pub mask: Vec<u8>,
    pub dist_px: Vec<f32>,
}

impl Silhouette {
    pub fn from_raster(raster: &Raster, luma_threshold: u8) -> Self {
        let n = (raster.width * raster.height) as usize;
        let mut background = vec![0u8; n];
        flood_studio_background(&mut background, raster, luma_threshold);
        let mut mask = vec![0u8; n];
        for i in 0..n {
            let a = raster.rgba[i * 4 + 3];
            if a > 8 && background[i] == 0 {
                mask[i] = 1;
            }
        }
        keep_largest_component(&mut mask, raster.width, raster.height);
        let dist_px = interior_distance(&mask, raster.width, raster.height);
        Self {
            width: raster.width,
            height: raster.height,
            mask,
            dist_px,
        }
    }

    pub fn sample(&self, u: f32, v: f32) -> Option<(bool, f32)> {
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        let x = (u * (self.width.saturating_sub(1) as f32)).round() as u32;
        let y = (v * (self.height.saturating_sub(1) as f32)).round() as u32;
        let i = (y * self.width + x) as usize;
        Some((self.mask[i] != 0, self.dist_px[i]))
    }

    pub fn foreground_count(&self) -> usize {
        self.mask.iter().filter(|&&m| m != 0).count()
    }
}

/// Dark pixels and low-chroma studio grey count as paper. Saturated organ
/// colours do not, so a grey seamless backdrop can be keyed without a
/// luma-only threshold filling the whole frame.
fn is_studio_paper(r: u8, g: u8, b: u8, a: u8, luma_threshold: u8) -> bool {
    if a <= 8 {
        return true;
    }
    let maxc = r.max(g).max(b);
    if maxc <= luma_threshold {
        return true;
    }
    let chroma = maxc - r.min(g).min(b);
    chroma <= 28
}

fn flood_studio_background(bg: &mut [u8], raster: &Raster, luma_threshold: u8) {
    let w = raster.width as usize;
    let h = raster.height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let mut stack = Vec::new();
    let push_paper = |x: usize, y: usize, stack: &mut Vec<usize>, bg: &mut [u8]| {
        let i = y * w + x;
        if bg[i] != 0 {
            return;
        }
        let o = i * 4;
        if is_studio_paper(
            raster.rgba[o],
            raster.rgba[o + 1],
            raster.rgba[o + 2],
            raster.rgba[o + 3],
            luma_threshold,
        ) {
            bg[i] = 1;
            stack.push(i);
        }
    };
    for x in 0..w {
        push_paper(x, 0, &mut stack, bg);
        push_paper(x, h - 1, &mut stack, bg);
    }
    for y in 0..h {
        push_paper(0, y, &mut stack, bg);
        push_paper(w - 1, y, &mut stack, bg);
    }
    while let Some(p) = stack.pop() {
        let x = p % w;
        let y = p / w;
        if x > 0 {
            push_paper(x - 1, y, &mut stack, bg);
        }
        if x + 1 < w {
            push_paper(x + 1, y, &mut stack, bg);
        }
        if y > 0 {
            push_paper(x, y - 1, &mut stack, bg);
        }
        if y + 1 < h {
            push_paper(x, y + 1, &mut stack, bg);
        }
    }
}

fn keep_largest_component(mask: &mut [u8], w: u32, h: u32) {
    let n = mask.len();
    let mut labels = vec![0u32; n];
    let mut current = 0u32;
    let mut best_id = 0u32;
    let mut best_count = 0u32;
    let mut stack = Vec::new();
    let wu = w as usize;
    let hu = h as usize;
    for y in 0..hu {
        for x in 0..wu {
            let i = y * wu + x;
            if mask[i] == 0 || labels[i] != 0 {
                continue;
            }
            current += 1;
            let mut count = 0u32;
            stack.clear();
            stack.push(i);
            labels[i] = current;
            while let Some(p) = stack.pop() {
                count += 1;
                let px = p % wu;
                let py = p / wu;
                let neighbors = [
                    px.checked_sub(1).map(|nx| py * wu + nx),
                    (px + 1 < wu).then_some(py * wu + px + 1),
                    py.checked_sub(1).map(|ny| ny * wu + px),
                    (py + 1 < hu).then_some((py + 1) * wu + px),
                ];
                for nidx in neighbors.into_iter().flatten() {
                    if mask[nidx] != 0 && labels[nidx] == 0 {
                        labels[nidx] = current;
                        stack.push(nidx);
                    }
                }
            }
            if count > best_count {
                best_count = count;
                best_id = current;
            }
        }
    }
    if best_id == 0 {
        return;
    }
    for i in 0..n {
        mask[i] = if labels[i] == best_id { 1 } else { 0 };
    }
}

fn interior_distance(mask: &[u8], w: u32, h: u32) -> Vec<f32> {
    let n = mask.len();
    let wu = w as usize;
    let hu = h as usize;
    let inf = 1.0e9f32;
    let mut d = vec![inf; n];
    for i in 0..n {
        if mask[i] == 0 {
            d[i] = 0.0;
        }
    }
    for y in 0..hu {
        for x in 0..wu {
            let i = y * wu + x;
            let mut best = d[i];
            if x > 0 {
                best = best.min(d[i - 1] + 1.0);
            }
            if y > 0 {
                best = best.min(d[i - wu] + 1.0);
            }
            if x > 0 && y > 0 {
                best = best.min(d[i - wu - 1] + 1.414);
            }
            if x + 1 < wu && y > 0 {
                best = best.min(d[i - wu + 1] + 1.414);
            }
            d[i] = best;
        }
    }
    for y in (0..hu).rev() {
        for x in (0..wu).rev() {
            let i = y * wu + x;
            let mut best = d[i];
            if x + 1 < wu {
                best = best.min(d[i + 1] + 1.0);
            }
            if y + 1 < hu {
                best = best.min(d[i + wu] + 1.0);
            }
            if x + 1 < wu && y + 1 < hu {
                best = best.min(d[i + wu + 1] + 1.414);
            }
            if x > 0 && y + 1 < hu {
                best = best.min(d[i + wu - 1] + 1.414);
            }
            d[i] = best;
        }
    }
    d
}
