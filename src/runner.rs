use std::io::Write;
use std::ops::Deref;
use std::{env::var, fs::File, marker::PhantomData, path::Path, time::Duration};

use crate::config::RandomSearchConfig;
use crate::library::{Domain, InputDataExt};
use crate::output::{BayesianOptimizationExtraFields, Evaluation, MethodType, OutputStats};
use crate::runners::{Config, RunnerMinimizer, RunnerResult};
use crate::{
    config::BayesianSearchConfig,
    library::{DomainBuilder, InputData, InputDeserializer, InputRunner, Value},
};
use argmin::core::Solver;
use friedrich::{
    gaussian_process::GaussianProcess,
    kernel::{Kernel, SquaredExp},
    prior::{Prior, ZeroPrior},
    Input,
};
use ndarray::{Array, ArrayView1};
use optimize::Minimizer;
use rand::{thread_rng, Rng, SeedableRng};
use serde::Serialize;

pub struct TestRunner<R: InputRunner<Data>, Data: InputData> {
    runner: R,
    _phantom_data: PhantomData<Data>,
}

impl<R: InputRunner<Data>, Data: InputData> TestRunner<R, Data> {
    pub fn new(runner: R) -> Self {
        Self {
            _phantom_data: PhantomData,
            runner,
        }
    }

    pub fn some_other_way<Minimizer: RunnerMinimizer<R, Data>>(
        &self,
        config: &Config,
        minimizer: &Minimizer,
    ) -> RunnerResult<Data> {
        let (domains, names) = Data::get_domains_and_names();
        if let Some(ref last_stats) = config.last_evaluations {
            if domains.len() != last_stats.domains.len() {
                panic!("Invalid data received!")
            }
            if !config.allow_different_domains {
                if !domains
                    .iter()
                    .zip(last_stats.domains.iter())
                    .all(|(x, y)| x == y)
                {
                    panic!("The domains are different!");
                }
            }
            if !config.allow_different_names {
                if !names
                    .iter()
                    .zip(last_stats.field_names.iter())
                    .all(|(x, y)| x == y)
                {
                    panic!("The field names are different!");
                }
            }
        }
        let last_evaluation = config
            .last_evaluations
            .iter()
            .flat_map(|v| v.evaluations.iter())
            .cloned();

        let mut stats = OutputStats {
            domains,
            field_names: names,
            evaluations: Vec::new(),
        };
        stats.evaluations.extend(
            config
                .last_evaluations
                .iter()
                .flat_map(|e| e.evaluations.iter().cloned()),
        );
        let iter = minimizer.generate_iterator(&self.runner, last_evaluation);
        let size = minimizer.buget();
        for (index, (data, result)) in iter.enumerate() {
            log::info!("Function evaluated {}/{}", index + 1, size);
            if config
                .save_interval
                .map(|interval| (index as u32 + 1) % interval == 0)
                .unwrap_or(false)
            {
                if let Some(ref path) = config.path {
                    self.save(&stats, path);
                    log::info!("Progress saved to {}", path);
                }
            }
            stats.evaluations.push(Evaluation {
                input: data.into_iter().map(|v| v.to_f64()).collect(),
                output: result,
            });
        }
        let (best_input, best_output) = stats.best::<Data>().unwrap();
        if let Some(ref path) = config.path {
            self.save(&stats, path);
        }
        RunnerResult {
            best_input,
            best_output,
            stats,
        }
    }

    fn save(&self, stats: &OutputStats, file: impl AsRef<Path>) {
        let writer = File::options()
            .write(true)
            .truncate(true)
            .create(true)
            .open(file)
            .unwrap();
        serde_json::to_writer_pretty(writer, stats).unwrap();
    }

    pub fn find_minimum(&self, config: BayesianSearchConfig) -> (Data, f64) {
        if config
            .init_samples
            .map(|s| s >= config.buget)
            .unwrap_or(false)
        {
            panic!(
                "The initial sample ({}) can't be bigger or eqaul to the the total buget({})",
                config.init_samples.unwrap(),
                config.buget
            );
        }

        let init_samples = if let Some(sample) = config.init_samples {
            sample
        } else {
            config.buget / 4
        };

        let domains = Data::get_domains_ext();
        let mut r = rand::thread_rng();

        let mut best = None;
        let mut best_value = None;
        let total = config.buget;
        for i in 0..init_samples {
            log::debug!("Running batch {i}/{total} INIT");
            let data = domains
                .iter()
                .map(|d| d.random_value(&mut r))
                .collect::<Vec<_>>();

            let data = Data::from_values(data);
            let output = self.runner.run(data.clone());
            if best_value.map(|b| output < b).unwrap_or(true) {
                best = Some(data);
                best_value = Some(output);
            }
        }

        for i in init_samples..config.buget {
            log::debug!("Running batch {i}/{total}");
            let data = domains
                .iter()
                .map(|d| d.random_value(&mut r))
                .collect::<Vec<_>>();

            let data = Data::from_values(data);
            let output = self.runner.run(data.clone());
            if best_value.map(|b| output < b).unwrap_or(true) {
                best = Some(data);
                best_value = Some(output);
            }
        }

        (best.unwrap(), best_value.unwrap())
    }

    pub fn minimum_bayesian_search(&self, config: BayesianSearchConfig) -> (Data, f64) {
        if config
            .init_samples
            .map(|s| s >= config.buget)
            .unwrap_or(false)
        {
            panic!(
                "The initial sample ({}) can't be bigger or eqaul to the the total buget({})",
                config.init_samples.unwrap(),
                config.buget
            );
        }

        let init_samples = if let Some(sample) = config.init_samples {
            sample
        } else {
            config.buget / 4
        };

        let domains = Data::get_domains_ext();
        let mut r = rand::thread_rng();

        let training_inputs = (0..init_samples)
            .map(|_| {
                domains
                    .iter()
                    .map(|d| d.random_value(&mut r))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let training_outputs = training_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let input = input.clone();

                let input_deserialized = Data::from_values(input);
                log::info!("Calling function {}/{} (INIT)", index + 1, config.buget);
                self.runner.run(input_deserialized) as f64
            })
            .collect::<Vec<_>>();

        let training_inputs_parsed = training_inputs
            .iter()
            .map(|input| input.iter().map(|v| v.to_f64()).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let (best, best_output) = training_inputs
            .iter()
            .zip(training_outputs.iter())
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();

        let mut best = best.clone();
        let mut best_output = *best_output;

        let mut gp =
            friedrich::gaussian_process::GaussianProcessBuilder::<SquaredExp, ZeroPrior>::new(
                training_inputs_parsed,
                training_outputs,
            )
            .set_noise(0f64)
            .train();

        let discrete_domains_index = Data::get_domains_ext()
            .into_iter()
            .enumerate()
            .filter_map(|(index, domain)| {
                if let Domain::Discrete(_) = domain {
                    Some(index)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for i in init_samples + 1..=config.buget {
            let minim_function = |input: &Vec<f64>| {
                minimize_function(
                    &gp,
                    2.,
                    discrete_domains_index.iter().cloned(),
                    &mut input.clone(),
                )
            };

            let steps = 100u64.pow(domains.len() as u32);

            let (input, expected_output) = (0..steps)
                .map(|_| {
                    domains
                        .iter()
                        .map(|d| match d {
                            crate::library::Domain::Continuos(d) => r.gen_range(d.clone()),
                            crate::library::Domain::Discrete(_) => todo!(),
                        })
                        .collect::<Vec<_>>()
                })
                .map(|value| {
                    let evaluation = minim_function(&value);
                    (value, evaluation)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            let input_parsed = input.iter().map(|v| Value::Float(*v)).collect::<Vec<_>>();
            let input_deserialized = Data::from_values(input_parsed.clone());

            log::info!("Calling function {}/{}", i, config.buget);
            let real_output = self.runner.run(input_deserialized);

            if best_output > real_output {
                best_output = real_output;
                best = input_parsed;
            }

            gp.add_samples(&input, &real_output);
        }

        let best = Data::from_values(best);
        (best, best_output)
    }

    pub fn minimum_bayesian_search_better(
        &mut self,
        mut config: BayesianSearchConfig,
    ) -> (Data, f64) {
        if config
            .init_samples
            .map(|s| s >= config.buget)
            .unwrap_or(false)
        {
            panic!(
                "The initial sample ({}) can't be bigger or eqaul to the the total buget({})",
                config.init_samples.unwrap(),
                config.buget
            );
        }

        let init_samples = if let Some(sample) = config.init_samples {
            sample
        } else {
            config.buget / 4
        };

        let domains = Data::get_domains_ext();
        let mut r = thread_rng();

        let mut evaluations = Vec::new();

        let training_inputs = (0..init_samples)
            .map(|_| {
                domains
                    .iter()
                    .map(|d| d.random_value(&mut r))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let training_outputs = training_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let input = input.clone();
                let input_deserialized = Data::from_values(input.iter().cloned());
                log::info!("Calling function {}/{} (INIT)", index + 1, config.buget);
                let output = self.runner.run(input_deserialized.clone()) as f64;
                evaluations.push(Evaluation {
                    input: input.into_iter().map(|i| i.to_f64()).collect(),
                    output,
                });
                output
            })
            .collect::<Vec<_>>();

        let mut training_inputs_parsed = training_inputs
            .iter()
            .map(|input| input.iter().map(|v| v.to_f64()).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let (best, best_output) = training_inputs
            .iter()
            .zip(training_outputs.iter())
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();

        let mut best = best.clone();
        let mut best_output = *best_output;

        let mut gp = friedrich::gaussian_process::GaussianProcessBuilder::<
            SquaredExp,
            friedrich::prior::ConstantPrior,
        >::new(training_inputs_parsed, training_outputs)
        .set_cholesky_epsilon(Some(0.01f64.powi(2)))
        .set_noise(0.01)
        // .fit_kernel()
        // .fit_prior()
        .train();
        let input_interval = domains
            .iter()
            .map(|d| match d {
                crate::library::Domain::Continuos(r) => (r.start, r.end),
                crate::library::Domain::Discrete(r) => (r.start as f64, r.end as f64),
            })
            .collect::<Vec<_>>();
        let mut nb_iterations = 100;
        let mut last_guess = vec![f64::NAN];
        let mut i = init_samples + 1;
        let mut retries = 0;
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
        while i <= config.buget {
            let minim_function = |input: &[f64]| {
                minimize_function(
                    &gp,
                    config.k,
                    discrete_domains_index.iter().cloned(),
                    &mut input.to_vec(),
                )
            };

            let (value, next_guess) = simplers_optimization::Optimizer::minimize(
                &minim_function,
                &input_interval,
                nb_iterations,
            );
            if last_guess
                .iter()
                .zip(next_guess.iter())
                .all(|(x, y)| x == y)
            {
                log::warn!(
                    "Last guess identical to this guess. make k larger by 1.5. {:?}, {:?}",
                    last_guess,
                    next_guess
                );
                config.k *= 1.5;
                retries += 1;
                if retries <= 2 {
                    continue;
                }
                log::warn!("Retried for 3 times. We are letting the function be evaluated again");
            }
            retries = 0;
            last_guess = next_guess.to_vec();
            let next_guess = next_guess.to_vec();

            let input_deserialized = Data::from_values(next_guess.iter().zip(domains.iter()).map(
                |(value, domain)| match domain {
                    Domain::Continuos(_) => Value::Float(*value),
                    Domain::Discrete(_) => Value::Integer(value.round() as i32),
                },
            ));
            log::info!("Calling function {}/{}", i, config.buget);
            let real_output = self.runner.run(input_deserialized.clone());

            log::info!("Adding sample");
            gp.add_samples(&next_guess, &real_output);
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
            if best_output > real_output {
                best_output = real_output;
                let input_parsed = next_guess
                    .iter()
                    .zip(domains.iter())
                    .map(|(v, domain)| match domain {
                        Domain::Continuos(_) => Value::Float(*v),
                        Domain::Discrete(_) => Value::Integer(*v as i32),
                    })
                    .collect::<Vec<_>>();
                best = input_parsed;
            }
            evaluations.push(Evaluation {
                input: next_guess,
                output: real_output,
            });
            i += 1;
        }

        let (domains, field_names) = Data::get_domains_and_names();
        let stats = OutputStats {
            evaluations,
            // method_type: MethodType::BayesianOptimization(BayesianOptimizationExtraFields {
            //     initial_sample: init_samples,
            // }),
            domains,
            field_names,
        };

        let best = Data::from_deserializer(InputDeserializer::new(best.into_iter()));

        (best, best_output)
    }
}

#[inline]
fn is_integer(value: f64) -> bool {
    (value.round() - value).abs() < 0.05
}

fn minimize_function<KernelType: Kernel, PriorType: Prior>(
    gp: &GaussianProcess<KernelType, PriorType>,
    k: f64,
    mut discrete_domains_index: impl Iterator<Item = usize> + Clone,
    inputs: &mut Vec<f64>,
) -> f64 {
    if let Some(index) = discrete_domains_index.next() {
        let result = if is_integer(inputs[index]) {
            minimize_function(gp, k, discrete_domains_index, inputs)
        } else {
            let c_value = inputs[index];
            let lower = c_value as i32 as f64;
            inputs[index] = lower;
            let lower_result = minimize_function(gp, k, discrete_domains_index.clone(), inputs);
            inputs[index] = lower + 1f64;
            let higher_result = minimize_function(gp, k, discrete_domains_index, inputs);
            inputs[index] = c_value;
            let procent_lower = 1f64 - (c_value - lower);
            lower_result * procent_lower + higher_result * (1f64 - procent_lower)
        };
        return result;
    }
    let (mean, variance) = gp.predict_mean_variance(inputs);
    if mean.is_nan() {
        println!("{:?}", gp.predict_mean_variance(&[11.].to_vec()));
        panic!("Error at mean for {:?}!", inputs);
    }
    if variance.is_nan() {
        panic!("Error at variance for {:?}!", variance);
    }
    mean - k * variance
}
