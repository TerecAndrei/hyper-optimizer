use std::{fs::File, path::Path};

use serde::{Deserialize, Serialize};

use crate::library::{Domain, InputData, InputDataExt};

#[derive(Serialize, Deserialize, Debug)]
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

    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) {
        let writer = File::options()
            .write(true)
            .truncate(true)
            .create(true)
            .open(path)
            .unwrap();
        serde_json::to_writer_pretty(writer, self).unwrap();
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Evaluation {
    pub input: Vec<f64>,
    pub output: f64,
}
