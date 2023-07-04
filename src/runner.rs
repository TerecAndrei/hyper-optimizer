use std::marker::PhantomData;

use crate::library::InputDataExt;
use crate::library::{InputData, InputRunner};
use crate::output::{Evaluation, OutputStats};
use crate::runners::{Config, OptimizerResult, RunnerMinimizer, SaveTrait};

pub struct Optimizer<R: InputRunner<Data>, Data: InputData> {
    runner: R,
    _phantom_data: PhantomData<Data>,
}

impl<R: InputRunner<Data>, Data: InputData> Optimizer<R, Data> {
    pub fn new(runner: R) -> Self {
        Self {
            _phantom_data: PhantomData,
            runner,
        }
    }

    pub fn optimize<Minimizer: RunnerMinimizer<R, Data>, SaveFn: SaveTrait>(
        &self,
        mut config: Config<SaveFn>,
        minimizer: &Minimizer,
    ) -> anyhow::Result<OptimizerResult<Data>> {
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
        let mut count = 0;
        for (index, (data, result)) in iter.enumerate() {
            count = index;
            log::debug!("Function evaluated {}/{}", index + 1, size);
            if config
                .save_interval
                .map(|interval| (index as u32 + 1) % interval == 0)
                .unwrap_or(false)
            {
                config.save_fn.save(&stats, index)?;
            }
            stats.evaluations.push(Evaluation {
                input: data.into_iter().map(|v| v.to_f64()).collect(),
                output: result,
            });
        }
        let (best_input, best_output) = stats.best::<Data>().unwrap();
        config.save_fn.save(&stats, count + 1)?;
        Ok(OptimizerResult {
            best_input,
            best_output,
            stats,
        })
    }
}
