use std::{marker::PhantomData, time::Duration};

use friedrich::{
    gaussian_process::GaussianProcess,
    kernel::{Kernel, SquaredExp},
    prior::{ConstantPrior, Prior},
};
use rand::thread_rng;

use crate::{
    library::{Domain, InputData, InputDataExt, InputRunner, Value},
    output::Evaluation,
};

use super::RunnerMinimizer;

pub struct BayesianRunnerMinimizer {
    buget: u64,
    initial_k: f64,
}

impl BayesianRunnerMinimizer {
    pub fn new(buget: u64) -> Self {
        BayesianRunnerMinimizer {
            buget,
            initial_k: 2.,
        }
    }

    pub fn new_with_options(budget: u64, initial_k: f64) -> Self {
        BayesianRunnerMinimizer {
            buget: budget,
            initial_k,
        }
    }
}

impl<Runner, Data> RunnerMinimizer<Runner, Data> for BayesianRunnerMinimizer
where
    Runner: InputRunner<Data>,
    Data: InputData + 'static,
{
    type IterResult<'a,IE> = BayesianIterator<'a, Runner, Data>
        where
            Runner: 'a,
            Self: 'a,
            IE: Iterator<Item = Evaluation> + 'a + Clone;
    fn generate_iterator<'a, IE>(
        &'a self,
        runner: &'a Runner,
        evaluations: IE,
    ) -> Self::IterResult<'a, IE>
    where
        IE: Iterator<Item = Evaluation> + Clone + 'a,
    {
        BayesianIterator::new(self, runner, evaluations)
    }

    fn buget(&self) -> u64 {
        self.buget
    }
}

pub struct BayesianIterator<'a, R, Data>
where
    R: InputRunner<Data>,
    Data: InputData,
{
    gp: GaussianProcess<SquaredExp, ConstantPrior>,
    runner: &'a R,
    buget_remaining: u64,
    k: f64,
    discrete_domains_index: Vec<usize>,
    nb_iterations: usize,
    input_interval: Vec<(f64, f64)>,
    last_guess: Vec<f64>,
    domains: Vec<Domain>,
    _phantom_data: PhantomData<Data>,
}

impl<'a, R, Data> BayesianIterator<'a, R, Data>
where
    R: InputRunner<Data>,
    Data: InputData,
{
    fn new<IE>(bayesian: &BayesianRunnerMinimizer, runner: &'a R, evaluations: IE) -> Self
    where
        IE: Iterator<Item = Evaluation> + Clone + 'a,
    {
        let domains = Data::get_domains_ext();

        let (training_inputs, training_outputs): (Vec<_>, Vec<_>) = evaluations
            .map(|mut e| {
                normalize_input(&mut e.input, &domains);
                (e.input, e.output)
            })
            .unzip();

        let mut gp = friedrich::gaussian_process::GaussianProcessBuilder::<
            SquaredExp,
            friedrich::prior::ConstantPrior,
        >::new(training_inputs, training_outputs)
        .set_cholesky_epsilon(Some(0.01f64.powi(2)))
        .set_noise(0.01)
        .fit_kernel()
        .fit_prior()
        .train();

        // if gp.noise < 0.1f64 {
        //     log::error!("Noise was {}. Setting it to 0.1", gp.noise);
        //     gp.noise = 0.01;
        //     gp.cholesky_epsilon = Some(0.01f64.powi(2));
        // }
        // log::error!("noise={}", gp.noise);

        let input_interval = domains
            .iter()
            .map(|d| match d {
                crate::library::Domain::Continuos(r) => (r.start, r.end),
                crate::library::Domain::Discrete(r) => (r.start as f64, r.end as f64),
            })
            .collect::<Vec<_>>();

        let discrete_domains_index = domains
            .iter()
            .enumerate()
            .filter_map(|(index, domain)| {
                if let Domain::Discrete(_) = domain {
                    Some(index)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        Self {
            _phantom_data: PhantomData,
            buget_remaining: bayesian.buget,
            discrete_domains_index,
            domains,
            gp,
            input_interval,
            k: bayesian.initial_k,
            runner,
            nb_iterations: 100,
            last_guess: vec![f64::NAN],
        }
    }

    fn get_next_guess(&mut self) -> Box<[f64]> {
        let mut retries = 0;
        let result = loop {
            let minim_function = |input: &[f64]| {
                minimize_function(
                    &self.gp,
                    self.k,
                    &self.domains,
                    self.discrete_domains_index.iter().cloned(),
                    &mut input.to_vec(),
                    true,
                )
            };

            let (value, next_guess) = simplers_optimization::Optimizer::minimize(
                &minim_function,
                &self.input_interval,
                self.nb_iterations,
            );
            if self
                .last_guess
                .iter()
                .zip(next_guess.iter())
                .all(|(x, y)| x == y)
            {
                log::warn!(
                    "Last guess identical to this guess. make k larger by 1.5. {:?}, {:?}",
                    self.last_guess,
                    next_guess
                );
                self.k *= 1.5;
                retries += 1;
                if retries < 2 {
                    continue;
                }
                if retries < 3 {
                    log::warn!("Retried for the 3 time. Fitting the model");
                    self.gp
                        .fit_parameters(true, true, 100, 0.05, Duration::from_secs(3600));
                    log::error!("{}", self.gp.noise);
                    log::error!("{:?}", self.gp.cholesky_epsilon);
                    if self.gp.noise < 0.1 || self.gp.noise.is_nan() {
                        self.gp.noise = 0.1
                    }
                    self.gp.cholesky_epsilon = Some(self.gp.noise.powi(2));
                    continue;
                }
                log::warn!("Retried for 3 times. We are letting the function be evaluated again");
            }
            self.last_guess = next_guess.to_vec();
            break next_guess;
        };
        // self.gp
        //     .fit_parameters(true, true, 100, 0.05, Duration::from_secs(3600));
        log::error!("noise={}", self.gp.noise);
        result
    }
}

impl<'a, R, Data> Iterator for BayesianIterator<'a, R, Data>
where
    R: InputRunner<Data>,
    Data: InputData,
{
    type Item = (Vec<Value>, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.buget_remaining == 0 {
            return None;
        }
        let next_guess = self.get_next_guess();
        let next_guess = next_guess.to_vec();

        let input = next_guess
            .iter()
            .zip(self.domains.iter())
            .map(|(value, domain)| match domain {
                Domain::Continuos(_) => Value::Float(*value),
                Domain::Discrete(_) => Value::Integer(value.round() as i32),
            })
            .collect::<Vec<_>>();

        let input_deserialized = Data::from_values(input.iter().cloned());
        let real_output = self.runner.run(input_deserialized);

        log::info!("Adding sample");
        self.gp.add_samples(&next_guess, &real_output);
        self.buget_remaining -= 1;
        // gp.fit_parameters(true, false, 200, 0.05, Duration::from_secs(60));
        // if gp.noise.is_nan() {
        //     log::error!("Noise was Nan. Fixing this!");
        //     gp.noise = 0.01;
        // }
        // gp.cholesky_epsilon = Some(gp.noise.powi(2));
        // log::info!(
        //     "Likelyhood: {} and noise {} and cholesky {:?}",
        //     gp.likelihood(),
        //     gp.noise,
        //     gp.cholesky_epsilon
        // );
        // log::warn!("gp at 0,0: {}", gp.predict(&vec![2f64, 3f64]));
        Some((input, real_output))
    }
}

#[inline]
fn is_integer(value: f64) -> bool {
    (value.round() - value).abs() < 0.05
}

fn normalize_input(inputs: &mut Vec<f64>, domains: &[Domain]) {
    for (input, domain) in inputs.iter_mut().zip(domains.iter()) {
        let range = domain.range();
        let size = range.end - range.start;
        *input = (*input - range.start) / size;
        assert!(0. <= *input && *input <= 1.);
    }
}

fn minimize_function<KernelType: Kernel, PriorType: Prior>(
    gp: &GaussianProcess<KernelType, PriorType>,
    k: f64,
    domains: &[Domain],
    mut discrete_domains_index: impl Iterator<Item = usize> + Clone,
    inputs: &mut Vec<f64>,
    should_normalize_input: bool,
) -> f64 {
    if let Some(index) = discrete_domains_index.next() {
        let result = if is_integer(inputs[index]) {
            minimize_function(gp, k, domains, discrete_domains_index, inputs, true)
        } else {
            let c_value = inputs[index];
            let lower = c_value as i32 as f64;
            inputs[index] = lower;
            let lower_result =
                minimize_function(gp, k, domains, discrete_domains_index.clone(), inputs, true);
            inputs[index] = lower + 1f64;
            let higher_result =
                minimize_function(gp, k, domains, discrete_domains_index, inputs, false);
            inputs[index] = c_value;
            let procent_lower = 1f64 - (c_value - lower);
            lower_result * procent_lower + higher_result * (1f64 - procent_lower)
        };
        return result;
    }
    if should_normalize_input {
        normalize_input(inputs, domains);
    }
    let (mean, mut variance) = gp.predict_mean_variance(inputs);
    if mean.is_nan() {
        println!("mean:{} variance:{}", mean, variance);
        panic!("Error at mean for {:?}!", inputs);
    }
    if variance.is_nan() {
        panic!("Error at variance for {:?}!", variance);
    }
    variance = variance.abs().sqrt();
    mean - k * variance
}
