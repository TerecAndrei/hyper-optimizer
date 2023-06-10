use crate::{
    library::{Domain, InputData, InputDataExt, InputRunner, Value},
    output::Evaluation,
    runners::ClonableIterator,
};
use itertools::Itertools;

use super::RunnerMinimizer;

pub enum GridRunnerStep {
    One(usize),
    ForEach(Vec<usize>),
}

impl GridRunnerStep {
    pub fn get_steps_size(&self, domains: &[Domain]) -> Vec<usize> {
        match self {
            Self::One(step) => vec![*step; domains.len()],
            Self::ForEach(v) => {
                if v.len() != domains.len() {
                    panic!(
                        "Expected a domains size of {}. Received {} in grid search",
                        v.len(),
                        domains.len()
                    );
                }
                v.clone()
            }
        }
    }
}

impl From<usize> for GridRunnerStep {
    fn from(value: usize) -> Self {
        Self::One(value)
    }
}

impl From<Vec<usize>> for GridRunnerStep {
    fn from(value: Vec<usize>) -> Self {
        Self::ForEach(value)
    }
}

pub struct GridRunnerMinimizer {
    step: GridRunnerStep,
}

impl GridRunnerMinimizer {
    pub fn new(step: impl Into<GridRunnerStep>) -> Self {
        Self { step: step.into() }
    }
}

impl<Runner, Data> RunnerMinimizer<Runner, Data> for GridRunnerMinimizer
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
        let domains = Data::get_domains_ext();
        let steps = self.step.get_steps_size(&domains);

        let r = domains
            .into_iter()
            .zip(steps)
            .map(|(d, steps)| {
                let result: ClonableIterator<Value> = match d {
                    Domain::Discrete(range) => {
                        let size = range.end - range.start;
                        let steps = steps - 1;
                        if steps == 0 {
                            ClonableIterator::new(std::iter::once(range.start.into()))
                        } else {
                            let mut step_size = size as f64 / steps as f64;
                            if step_size < 1f64 {
                                step_size = 1f64;
                            }
                            ClonableIterator::new(
                                (0..=steps)
                                    .map(move |i| {
                                        (range.start as f64 + i as f64 * step_size) as i32
                                    })
                                    .map(Value::Integer),
                            )
                        }
                    }
                    Domain::Continuos(range) => {
                        let size = range.end - range.start;
                        let steps = steps - 1;
                        if steps == 0 {
                            ClonableIterator::new(std::iter::once(range.start.into()))
                        } else {
                            let step_size = size / steps as f64;
                            ClonableIterator::new(
                                (0..=steps)
                                    .map(move |i| (range.start + i as f64 * step_size))
                                    .map(Value::Float),
                            )
                        }
                    }
                };
                result
            })
            .multi_cartesian_product();

        let iterator = r.map(|v| {
            let data = Data::from_values(v.iter().cloned());
            let output = runner.run(data);
            (v, output)
        });
        Box::new(iterator)
    }

    fn buget(&self) -> u64 {
        0
    }
}
