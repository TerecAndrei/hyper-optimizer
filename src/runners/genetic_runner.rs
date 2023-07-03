use std::marker::PhantomData;

use crate::{
    library::{Domain, InputData, InputDataExt, InputRunner, Value},
    output::Evaluation,
};
use itertools::Itertools;
use rand::{prelude::Distribution, rngs::ThreadRng, Rng};

use super::RunnerMinimizer;

pub trait FitnessFunction {
    /**
     * This function should take the objective value as input and map it to a value greater than 0. The smaller the objective, the greater should be
     */
    fn transform_objective(&self, objective: f64) -> f64;
}

impl<F> FitnessFunction for F
where
    F: Fn(f64) -> f64,
{
    fn transform_objective(&self, objective: f64) -> f64 {
        self(objective)
    }
}

pub struct ExpFitnessFunction;

impl FitnessFunction for ExpFitnessFunction {
    fn transform_objective(&self, objective: f64) -> f64 {
        (-objective).exp()
    }
}

pub struct GeneticConfig<F: FitnessFunction = ExpFitnessFunction, R: rand::Rng + Clone = ThreadRng>
{
    pub mutation_rate: f64,
    pub fitness_function: F,
    pub rng: R,
    pub use_prior_evaluations_percent: f64,
    pub elitism: u64,
}

impl Default for GeneticConfig {
    fn default() -> Self {
        Self {
            mutation_rate: 0.1,
            fitness_function: ExpFitnessFunction,
            rng: rand::thread_rng(),
            use_prior_evaluations_percent: 0.5,
            elitism: 0,
        }
    }
}

#[derive(Clone)]
pub struct GeneticEvaluation {
    input: Vec<Value>,
    output: Option<f64>,
}

impl GeneticEvaluation {
    fn from_evaluation(evaluation: &Evaluation, domains: &[Domain]) -> Self {
        Self {
            input: evaluation
                .input
                .iter()
                .zip(domains.iter())
                .map(|(input, d)| match d {
                    Domain::Continuos(_) => Value::Float(*input),
                    Domain::Discrete(_) => Value::Integer(input.round() as i32),
                })
                .collect(),
            output: Some(evaluation.output),
        }
    }
}

pub struct GeneticRunnerMinimizer<
    Fitness: FitnessFunction = ExpFitnessFunction,
    R: Rng + Clone = ThreadRng,
> {
    epochs: u64,
    population_size: u64,
    config: GeneticConfig<Fitness, R>,
}

impl GeneticRunnerMinimizer {
    pub fn new(epochs: u64, population_size: u64) -> Self {
        Self {
            epochs,
            population_size,
            config: GeneticConfig::default(),
        }
    }
}

impl<Fitness: FitnessFunction, R: Rng + Clone> GeneticRunnerMinimizer<Fitness, R> {
    pub fn new_with_options(
        epochs: u64,
        population_size: u64,
        config: GeneticConfig<Fitness, R>,
    ) -> Self {
        Self {
            epochs,
            population_size,
            config,
        }
    }
}

impl<Runner, Data, R: Rng + Clone, Fitness: FitnessFunction> RunnerMinimizer<Runner, Data>
    for GeneticRunnerMinimizer<Fitness, R>
where
    Runner: InputRunner<Data>,
    Data: InputData + 'static,
{
    type IterResult<'a, IE> = GeneticRunnerMinimizerIterator<'a, Runner, Data, R, Fitness>
        where
            Runner: 'a,
            Self: 'a,
            IE: Iterator<Item = Evaluation> + Clone + 'a,;
    fn generate_iterator<'a, IE>(
        &'a self,
        runner: &'a Runner,
        evaluations: IE,
    ) -> Self::IterResult<'a, IE>
    where
        IE: Iterator<Item = Evaluation> + Clone + 'a,
    {
        let domains = Data::get_domains_ext();
        let evaluations = evaluations
            .sorted_unstable_by(|a, b| a.output.total_cmp(&b.output))
            .collect::<Vec<_>>();

        let from_prior = self.config.use_prior_evaluations_percent * self.population_size as f64;

        let from_prior = from_prior as u64;

        let elitism_count = self.config.elitism.min(from_prior);

        let crossover_count = from_prior - elitism_count;

        let random_members = self.population_size - from_prior;

        let mut rng = self.config.rng.clone();

        let elitism = evaluations
            .iter()
            .take(elitism_count as usize)
            .map(|e| GeneticEvaluation::from_evaluation(e, &domains));

        let crossover = evaluations
            .iter()
            .take(crossover_count as usize * 2)
            .tuples()
            .map(|(e1, e2)| {
                let e1 = GeneticEvaluation::from_evaluation(e1, &domains);
                let e2 = GeneticEvaluation::from_evaluation(e2, &domains);

                crossover(&e1, &e2, &mut rng)
            })
            .collect::<Vec<_>>();

        let random_population = (0..random_members)
            .map(|_| {
                let data = domains
                    .iter()
                    .map(|d| d.random_value(&mut rng))
                    .collect::<Vec<_>>();
                // let output = runner.run(Data::from_values(data.iter().cloned()));
                GeneticEvaluation {
                    input: data,
                    output: None,
                }
            })
            .collect::<Vec<_>>();

        let population = elitism
            .chain(crossover)
            .chain(random_population)
            .collect::<Vec<_>>();
        GeneticRunnerMinimizerIterator::new(
            runner,
            population,
            self.population_size,
            self.epochs,
            &self.config,
        )
    }

    fn buget(&self) -> u64 {
        self.epochs * self.population_size
    }
}

impl<Fitness: FitnessFunction, R: Rng + Clone> GeneticRunnerMinimizer<Fitness, R> {}

pub struct GeneticRunnerMinimizerIterator<
    'a,
    Runner: InputRunner<Data>,
    Data: InputData,
    R: Rng + Clone,
    Fitness: FitnessFunction,
> {
    population_size: u64,
    epochs: u64,
    current_epoch: u64,
    population: Vec<GeneticEvaluation>,
    domains: Vec<Domain>,
    config: &'a GeneticConfig<Fitness, R>,
    rng: R,
    current_evaluation_index: usize,
    runner: &'a Runner,
    _phantom_data: PhantomData<Data>,
}

impl<'a, Runner: InputRunner<Data>, Data: InputData, R: Rng + Clone, Fitness: FitnessFunction>
    GeneticRunnerMinimizerIterator<'a, Runner, Data, R, Fitness>
{
    fn new(
        runner: &'a Runner,
        population: Vec<GeneticEvaluation>,
        population_size: u64,
        epochs: u64,
        config: &'a GeneticConfig<Fitness, R>,
    ) -> Self {
        Self {
            population_size,
            epochs,
            current_epoch: 1,
            population,
            domains: Data::get_domains_ext(),
            config,
            rng: config.rng.clone(),
            current_evaluation_index: 0,
            runner,
            _phantom_data: PhantomData,
        }
    }

    fn generate_next_population(&mut self) {
        self.population
            .sort_unstable_by(|a, b| a.output.unwrap().total_cmp(&b.output.unwrap()));
        let fitness_scores = self.population.iter().map(|e| {
            self.config
                .fitness_function
                .transform_objective(e.output.unwrap()) as f32
        });

        let elitism = self
            .population
            .iter()
            .take(self.config.elitism as usize)
            .cloned()
            .collect::<Vec<_>>();

        let weight = rand::distributions::WeightedIndex::new(fitness_scores).unwrap();

        let iter_weighted = weight.sample_iter(&mut self.rng);

        let new_population = iter_weighted
            .take((self.population_size as usize - elitism.len()) * 2)
            .tuples()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(index_parent1, index_parent2)| {
                let parent_1 = &self.population[index_parent1];
                let parent_2 = &self.population[index_parent2];

                let mut child = crossover(parent_1, parent_2, &mut self.rng);
                mutate(
                    &mut child,
                    &self.domains,
                    self.config.mutation_rate,
                    &mut self.rng,
                );
                child
            })
            .chain(elitism)
            .collect::<Vec<_>>();
        self.population = new_population;
        self.current_evaluation_index = 0;
    }
}

fn crossover<R: Rng>(
    parent_1: &GeneticEvaluation,
    parent_2: &GeneticEvaluation,
    rng: &mut R,
) -> GeneticEvaluation {
    let mut result = parent_1.clone();
    let mut genes_from_parent_2 = 0;
    for (gene_child, gene_parent_2) in result.input.iter_mut().zip(parent_2.input.iter()) {
        if rng.gen_bool(0.5) {
            *gene_child = *gene_parent_2;
            genes_from_parent_2 += 1;
            result.output = None;
        }
    }
    //if all genes where copied from the second parent, we can copy the output as well
    if genes_from_parent_2 == parent_2.input.len() {
        result.output = parent_2.output;
    }
    result
}

fn mutate<R: Rng>(
    child: &mut GeneticEvaluation,
    domains: &[Domain],
    mutation_change: f64,
    rng: &mut R,
) {
    for (gene, domain) in child.input.iter_mut().zip(domains) {
        if rng.gen_bool(mutation_change) {
            child.output = None;
            *gene = domain.random_value(rng);
        }
    }
}

impl<'a, Runner: InputRunner<Data>, Data: InputData, R: Rng + Clone, Fitness: FitnessFunction>
    Iterator for GeneticRunnerMinimizerIterator<'a, Runner, Data, R, Fitness>
{
    type Item = (Vec<Value>, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_epoch > self.epochs {
            return None;
        }

        while self.current_evaluation_index < self.population.len() {
            let mut evaluation = &mut self.population[self.current_evaluation_index];
            if evaluation.output.is_none() {
                let data = Data::from_values(evaluation.input.iter().cloned());
                let output = self.runner.run(data);
                evaluation.output = Some(output);
                self.current_evaluation_index += 1;
                return Some((evaluation.input.clone(), output));
            }

            self.current_evaluation_index += 1;
        }

        self.current_epoch += 1;
        if self.current_epoch > self.epochs {
            return None;
        }

        self.generate_next_population();
        self.next()
    }
}
