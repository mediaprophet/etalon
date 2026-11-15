use std::fmt;

#[derive(Debug)]
pub enum ReconstructError {
    EmptyViews,
    InvalidResolution,
    Decode(String),
    EmptySilhouette { view: String },
    EmptyMesh,
    Io(String),
}

impl fmt::Display for ReconstructError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyViews => write!(f, "at least one source view is required"),
            Self::InvalidResolution => {
                write!(f, "resolution must be in 8..=256")
            }
            Self::Decode(msg) => write!(f, "image decode: {msg}"),
            Self::EmptySilhouette { view } => {
                write!(f, "silhouette is empty in view {view}")
            }
            Self::EmptyMesh => write!(f, "reconstruction produced no surface"),
            Self::Io(msg) => write!(f, "io: {msg}"),
        }
    }
}

impl std::error::Error for ReconstructError {}
