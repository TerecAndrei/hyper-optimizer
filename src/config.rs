pub struct BayesianSearchConfig {
    pub buget: u64,
    pub init_samples: Option<u64>,
    pub k: f64,
}

impl Default for BayesianSearchConfig {
    fn default() -> Self {
        Self {
            buget: 40,
            init_samples: None,
            k: 2.,
        }
    }
}

pub struct RandomSearchConfig {
    pub buget: u64,
}

pub enum GridDelta {
    Value(f64),
    Array(Vec<f64>),
}

impl From<f64> for GridDelta {
    fn from(value: f64) -> Self {
        Self::Value(value)
    }
}

impl From<Vec<f64>> for GridDelta {
    fn from(value: Vec<f64>) -> Self {
        Self::Array(value)
    }
}

pub struct GridSearchConfig {
    pub delta: GridDelta,
}
