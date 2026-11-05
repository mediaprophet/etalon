//! Neutral, serializable scene contract — the formal hand-off between the
//! decentralized state/runtime layer and the GPU.
//!
//! This is deliberately **below** webizen-studio's `SemanticScene`: studio (and
//! any host) maps its semantic graph + a geometry layout into a [`RenderScene`],
//! which carries only what the GPU needs — world-space node positions, edges, and
//! CSS colors. Keeping it serde-serializable lets a host build it from QualiaDB
//! data, ship it across the Tauri IPC, or persist it, without webizen-render ever
//! depending on the UI crate.
//!
//! Colors are CSS strings (`"#67e8f9"`, `"rgba(...)"`) to match the rest of the
//! engine; the renderer parses them. Positions are world-space; the renderer
//! projects them with its `Camera`.

use serde::{Deserialize, Serialize};

/// A look-at camera for the render contract.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneCamera {
    pub eye: [f64; 3],
    pub target: [f64; 3],
    pub up: [f64; 3],
    pub fov_degrees: f64,
}

impl Default for SceneCamera {
    fn default() -> Self {
        Self {
            eye: [0.0, 0.0, 5.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_degrees: 60.0,
        }
    }
}

/// A node: a world-space point drawn as a billboarded disc. In a semantic scene
/// this is one [`SceneItem`](https://docs.rs/) — an organ highlight, a graph
/// entity — colored by its state/intensity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneNode {
    /// World-space position.
    pub position: [f64; 3],
    /// CSS color string.
    pub color: String,
    /// Disc radius in screen pixels.
    pub radius: f64,
    /// Opacity 0.0–1.0.
    #[serde(default = "one")]
    pub alpha: f64,
}

/// A relation between two nodes, drawn as a line. Indices refer to
/// [`RenderScene::nodes`]; out-of-range edges are skipped.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneEdge {
    pub from: usize,
    pub to: usize,
    pub color: String,
    #[serde(default = "one")]
    pub width: f64,
    #[serde(default = "one")]
    pub alpha: f64,
}

/// The complete GPU-facing scene: camera, background, nodes, and edges.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RenderScene {
    #[serde(default)]
    pub camera: SceneCamera,
    /// CSS background color cleared before drawing.
    #[serde(default = "default_background")]
    pub background: String,
    #[serde(default)]
    pub nodes: Vec<SceneNode>,
    #[serde(default)]
    pub edges: Vec<SceneEdge>,
}

fn one() -> f64 {
    1.0
}

fn default_background() -> String {
    "#0b0e14".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_camera_defaults() {
        let cam = SceneCamera::default();
        assert_eq!(cam.eye, [0.0, 0.0, 5.0]);
        assert_eq!(cam.target, [0.0, 0.0, 0.0]);
        assert_eq!(cam.up, [0.0, 1.0, 0.0]);
        assert_eq!(cam.fov_degrees, 60.0);
    }

    #[test]
    fn render_scene_construction() {
        let mut scene = RenderScene {
            camera: SceneCamera::default(),
            background: default_background(),
            nodes: vec![SceneNode {
                position: [1.0, 2.0, 3.0],
                color: "#67e8f9".to_string(),
                radius: 4.0,
                alpha: one(),
            }],
            edges: vec![SceneEdge {
                from: 0,
                to: 0,
                color: "#ffffff".to_string(),
                width: one(),
                alpha: one(),
            }],
        };
        assert_eq!(scene.background, "#0b0e14");
        assert_eq!(scene.nodes.len(), 1);
        assert_eq!(scene.edges.len(), 1);
        scene.nodes.clear();
        assert!(scene.nodes.is_empty());
    }
}
