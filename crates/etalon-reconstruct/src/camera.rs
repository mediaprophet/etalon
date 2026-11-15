use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKind {
    Anterior,
    Posterior,
    LeftLateral,
    RightLateral,
    Superior,
    Inferior,
    ThreeQuarter,
    Custom,
}

impl ViewKind {
    pub fn parse_name(name: &str) -> Option<Self> {
        let n = name.to_ascii_lowercase();
        let n = n
            .trim_end_matches(".jpg")
            .trim_end_matches(".jpeg")
            .trim_end_matches(".png")
            .trim_end_matches(".webp");
        if n.contains("three-quarter") || n.contains("three_quarter") || n.contains("3q") {
            return Some(Self::ThreeQuarter);
        }
        if n.contains("anterior") || n.ends_with("front") {
            return Some(Self::Anterior);
        }
        if n.contains("posterior") || n.ends_with("back") {
            return Some(Self::Posterior);
        }
        if n.contains("left-lateral") || n.contains("left_lateral") || n.ends_with("-left") {
            return Some(Self::LeftLateral);
        }
        if n.contains("right-lateral") || n.contains("right_lateral") || n.ends_with("-right") {
            return Some(Self::RightLateral);
        }
        if n.contains("superior") || n.contains("base") {
            return Some(Self::Superior);
        }
        if n.contains("inferior") {
            return Some(Self::Inferior);
        }
        if n.contains("gray-492") {
            return Some(Self::Anterior);
        }
        if n.contains("gray-491") {
            return Some(Self::Posterior);
        }
        if n.contains("gray-494") {
            return Some(Self::Superior);
        }
        None
    }

    pub fn from_path(path: &Path) -> Option<Self> {
        path.file_name()
            .and_then(|s| s.to_str())
            .and_then(Self::parse_name)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anterior => "anterior",
            Self::Posterior => "posterior",
            Self::LeftLateral => "left_lateral",
            Self::RightLateral => "right_lateral",
            Self::Superior => "superior",
            Self::Inferior => "inferior",
            Self::ThreeQuarter => "three_quarter",
            Self::Custom => "custom",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    pub fn heart_cm() -> Self {
        Self {
            min: [-6.0, -7.0, -5.5],
            max: [6.0, 7.0, 5.5],
        }
    }

    pub fn center(self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    pub fn size(self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OrthoCamera {
    pub center: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    pub half_width: f32,
    pub half_height: f32,
}

impl OrthoCamera {
    pub fn for_view(kind: ViewKind, bounds: Aabb) -> Self {
        let center = bounds.center();
        let size = bounds.size();
        let (forward, right, up) = match kind {
            ViewKind::Anterior | ViewKind::Custom => {
                ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            }
            ViewKind::Posterior => ([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ViewKind::LeftLateral => ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ViewKind::RightLateral => ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
            ViewKind::Superior => ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            ViewKind::Inferior => ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ViewKind::ThreeQuarter => {
                let f = normalize([-0.55, -0.18, -0.82]);
                let r = normalize(cross(f, [0.0, 1.0, 0.0]));
                let u = cross(r, f);
                (f, r, u)
            }
        };
        let extent = size[0].max(size[1]).max(size[2]) * 0.58;
        Self {
            center,
            right,
            up,
            forward,
            half_width: extent,
            half_height: extent,
        }
    }

    pub fn project(&self, p: [f32; 3]) -> (f32, f32, f32) {
        let d = sub(p, self.center);
        let u = dot(d, self.right) / self.half_width;
        let v = dot(d, self.up) / self.half_height;
        let depth = -dot(d, self.forward);
        let su = u * 0.5 + 0.5;
        let sv = 0.5 - v * 0.5;
        (su, sv, depth)
    }
}

pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

pub fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = length(a);
    if l < 1.0e-8 {
        [0.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / l)
    }
}
