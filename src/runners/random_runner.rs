use rand::{rngs::ThreadRng, thread_rng, Rng};

use crate::{
    library::{InputData, InputDataExt, InputRunner, Value},
    output::Evaluation,
};

use super::RunnerMinimizer;
pub struct RandomRunner<R: Rng + Clone = ThreadRng> {
    buget: u64,
    rng: R,
}

impl RandomRunner {
    pub fn new(buget: u64) -> Self {
        Self {
            buget,
            rng: thread_rng(),
        }
    }
}

impl<R: Rng + Clone> RandomRunner<R> {
    pub fn new_with_rng(buget: u64, rng: R) -> Self {
        Self { buget, rng }
    }
}

impl<Runner, Data, R: Rng + Clone> RunnerMinimizer<Runner, Data> for RandomRunner<R>
where
    Runner: InputRunner<Data>,
    Data: InputData + 'static,
{
    type IterResult<'a, IE> = Box<dyn Iterator<Item = (Vec<Value>, f64)> + 'a>
        where
            Runner: 'a,
            Self: 'a,
            IE: Iterator<Item = Evaluation> + Clone + 'a,;
    fn generate_iterator<'a, IE>(&'a self, runner: &'a Runner, _: IE) -> Self::IterResult<'a, IE>
    where
        IE: Iterator<Item = Evaluation> + Clone + 'a,
    {
        let domains = Data::get_domains_ext();
        let mut rng = self.rng.clone();

        let iterator = (0..self.buget).map(move |_| {
            let data = domains
                .iter()
                .map(|d| d.random_value(&mut rng))
                .collect::<Vec<_>>();
            let output = runner.run(Data::from_values(data.iter().cloned()));
            (data, output)
        });
        Box::new(iterator)
    }

    fn buget(&self) -> u64 {
        self.buget
    }
}
