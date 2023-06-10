pub mod bayesian_runner;
// pub mod boxed_runner;
pub mod composed_runner;
pub mod fixed_runner;
pub mod genetic_runner;
pub mod grid_runner;
pub mod random_runner;

use std::fs::File;

use crate::{
    library::{InputData, InputRunner, Value},
    output::{Evaluation, OutputStats},
};

use self::composed_runner::ComposedRunnerMinimizer;
pub trait RunnerMinimizer<Runner, Data>
where
    Runner: InputRunner<Data>,
    Data: InputData,
{
    type IterResult<'a, IE>: Iterator<Item = (Vec<Value>, f64)> + 'a
    where
        Runner: 'a,
        Self: 'a,
        IE: Iterator<Item = Evaluation> + 'a + Clone;
    fn generate_iterator<'a, IE>(
        &'a self,
        runner: &'a Runner,
        last_evaluations: IE,
    ) -> Self::IterResult<'a, IE>
    where
        IE: Iterator<Item = Evaluation> + Clone + 'a;

    fn buget(&self) -> u64;
}

pub trait RunnerMinimizerExt<Runner, Data>: RunnerMinimizer<Runner, Data> + Sized
where
    Runner: InputRunner<Data>,
    Data: InputData,
{
    fn merge<T>(self, other: T) -> ComposedRunnerMinimizer<Self, T, Runner, Data>
    where
        T: RunnerMinimizer<Runner, Data> + Sized;
}

impl<Runner, Data, RMinimizer> RunnerMinimizerExt<Runner, Data> for RMinimizer
where
    Runner: InputRunner<Data>,
    Data: InputData,
    RMinimizer: RunnerMinimizer<Runner, Data>,
{
    fn merge<T>(self, other: T) -> ComposedRunnerMinimizer<Self, T, Runner, Data>
    where
        T: RunnerMinimizer<Runner, Data> + Sized,
    {
        ComposedRunnerMinimizer::new(self, other)
    }
}

pub struct Config {
    pub(crate) save_interval: Option<u32>,
    pub(crate) last_evaluations: Option<OutputStats>,
    pub(crate) allow_different_domains: bool,
    pub(crate) allow_different_names: bool,
    pub(crate) path: Option<String>,
}

pub struct ConfigBuilder {
    save_interval: Option<u32>,
    prior_evaluations: Option<OutputStats>,
    allow_different_domains: bool,
    allow_different_names: bool,
    output_path: Option<String>,
    read_evaluations_from: Option<String>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        ConfigBuilder {
            save_interval: None,
            prior_evaluations: None,
            allow_different_domains: false,
            allow_different_names: false,
            output_path: None,
            read_evaluations_from: None,
        }
    }

    pub fn save_interval(mut self, interval: u32) -> Self {
        self.save_interval = Some(interval);
        self
    }

    pub fn set_prior_evaluations(mut self, prior_evaluations: OutputStats) -> Self {
        self.prior_evaluations = Some(prior_evaluations);
        self
    }

    pub fn allow_different_domains(mut self, allow_different_domains: bool) -> Self {
        self.allow_different_domains = allow_different_domains;
        self
    }

    pub fn allow_different_names(mut self, allow_different_names: bool) -> Self {
        self.allow_different_names = allow_different_names;
        self
    }

    pub fn read_from_file(mut self, path: String) -> Self {
        self.read_evaluations_from = Some(path);
        self
    }

    pub fn output(mut self, path: String) -> Self {
        self.output_path = Some(path);
        self
    }

    pub fn build(self) -> Config {
        let mut prior_evaluations = self.prior_evaluations;
        if let Some(path) = self.read_evaluations_from {
            let file = File::open(path).unwrap();
            prior_evaluations = Some(serde_json::from_reader(file).unwrap());
        }
        Config {
            save_interval: self.save_interval,
            last_evaluations: prior_evaluations,
            allow_different_domains: self.allow_different_domains,
            allow_different_names: self.allow_different_names,
            path: self.output_path,
        }
    }
}

pub struct OptimizerResult<Data: InputData> {
    pub best_input: Data,
    pub best_output: f64,
    pub stats: OutputStats,
}

trait ClonableIteratorTrait: Iterator {
    fn clone_box(&self) -> Box<dyn ClonableIteratorTrait<Item = Self::Item>>;
}

impl<T> ClonableIteratorTrait for T
where
    T: Iterator + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn ClonableIteratorTrait<Item = Self::Item>> {
        Box::new(self.clone())
    }
}

pub struct ClonableIterator<T> {
    iter: Box<dyn ClonableIteratorTrait<Item = T>>,
}

impl<T> ClonableIterator<T> {
    pub fn new(iter: impl Iterator<Item = T> + Clone + 'static) -> Self {
        Self {
            iter: Box::new(iter),
        }
    }
}

impl<T> Clone for ClonableIterator<T> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone_box(),
        }
    }
}

impl<T> Iterator for ClonableIterator<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
