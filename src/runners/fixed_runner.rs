use std::collections::HashSet;

use rand::thread_rng;

use crate::{
    library::{InputData, InputDataExt, InputRunner, Value},
    output::Evaluation,
};

use super::RunnerMinimizer;
pub struct FixedRunnerMinimizer {
    points: Vec<Vec<Value>>,
}

impl FixedRunnerMinimizer {
    pub fn new(evaluations: impl IntoIterator<Item = Vec<Value>>) -> Self {
        Self {
            points: evaluations.into_iter().collect(),
        }
    }
}

impl<Runner, Data> RunnerMinimizer<Runner, Data> for FixedRunnerMinimizer
where
    Runner: InputRunner<Data>,
    Data: InputData + 'static,
{
    type IterResult<'a, IE> = Box<dyn Iterator<Item = (Vec<Value>, f64)> + 'a>
        where
            Runner: 'a,
            Self: 'a,
            IE: Iterator<Item = Evaluation> + Clone + 'a,;
    fn generate_iterator<'a, IE>(
        &'a self,
        runner: &'a Runner,
        _evaluations: IE,
    ) -> Self::IterResult<'a, IE>
    where
        IE: Iterator<Item = Evaluation> + Clone + 'a,
    {
        // let domains = Data::get_domains_ext();

        let iterator = self.points.iter().map(move |data| {
            let output = runner.run(Data::from_values(data.iter().cloned()));
            (data.clone(), output)
        });
        Box::new(iterator)
    }

    fn buget(&self) -> u64 {
        self.points.len() as u64
    }
}
