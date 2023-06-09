use serde::{Deserialize, Serialize};

use crate::library::{Domain, InputData, InputDataExt};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputStats {
    // pub method_type: MethodType,
    pub domains: Vec<Domain>,
    pub field_names: Vec<String>,
    pub evaluations: Vec<Evaluation>,
}

impl OutputStats {
    pub fn best<Data: InputData>(&self) -> Option<(Data, f64)> {
        self.evaluations
            .iter()
            .min_by(|a, b| a.output.total_cmp(&b.output))
            .map(|e| (Data::from_f64(e.input.iter().cloned()), e.output))
    }
}

#[derive(Serialize)]
#[serde(tag = "methodType", content = "extra")]
#[serde(rename_all = "camelCase")]
pub enum MethodType {
    BayesianOptimization(BayesianOptimizationExtraFields),
    RandomSearch,
    GridSearch(GridSearchExtraFields),
    GeneticAlgorithm(GeneticAlgorithmExtraFields),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Evaluation {
    pub input: Vec<f64>,
    pub output: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BayesianOptimizationExtraFields {
    pub initial_sample: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GridSearchExtraFields {}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeneticAlgorithmExtraFields {}
