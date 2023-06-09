use rand::thread_rng;

use crate::{
    library::{InputData, InputDataExt, InputRunner, Value},
    output::Evaluation,
};

use super::RunnerMinimizer;
pub struct RandomRunner {
    buget: u64,
}

impl RandomRunner {
    pub fn new(buget: u64) -> Self {
        Self { buget }
    }
}

impl<Runner, Data> RunnerMinimizer<Runner, Data> for RandomRunner
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
        let mut rng = thread_rng();

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
