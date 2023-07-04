use std::{fs::File, path::Path};

use serde::{Deserialize, Serialize};

use crate::library::{Domain, InputData, InputDataExt};

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct OutputStats {
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

    pub fn read_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<OutputStats> {
        let file = File::open(path)?;

        Ok(serde_json::from_reader(file)?)
    }

    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let writer = File::options()
            .write(true)
            .truncate(true)
            .create(true)
            .open(path)?;
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Evaluation {
    pub input: Vec<f64>,
    pub output: f64,
}

impl Evaluation {
    pub fn input_to_data<Data: InputData>(&self) -> Data {
        let domains = Data::get_domains_ext();
        Data::from_values(
            self.input
                .iter()
                .zip(domains)
                .map(|(input, domain)| match domain {
                    Domain::Continuos(_) => crate::library::Value::Float(*input),
                    Domain::Discrete(_) => crate::library::Value::Integer(*input as i32),
                }),
        )
    }
}
