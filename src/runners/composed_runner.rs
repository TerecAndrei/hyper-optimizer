use std::marker::PhantomData;

use crate::{
    library::{InputData, InputRunner, Value},
    output::Evaluation,
};

use super::RunnerMinimizer;

pub struct ComposedRunner<F, S, Runner, Data>
where
    F: RunnerMinimizer<Runner, Data> + Sized,
    S: RunnerMinimizer<Runner, Data> + Sized,
    Data: InputData,
    Runner: InputRunner<Data>,
{
    first: F,
    second: S,
    _runner_phantom_data: PhantomData<Runner>,
    _data_phantom_data: PhantomData<Data>,
}

impl<F, S, Runner, Data> ComposedRunner<F, S, Runner, Data>
where
    F: RunnerMinimizer<Runner, Data>,
    S: RunnerMinimizer<Runner, Data>,
    Data: InputData,
    Runner: InputRunner<Data>,
{
    pub fn new(first: F, second: S) -> Self {
        Self {
            first,
            second,
            _data_phantom_data: PhantomData,
            _runner_phantom_data: PhantomData,
        }
    }
}

impl<F, S, Runner, Data> RunnerMinimizer<Runner, Data> for ComposedRunner<F, S, Runner, Data>
where
    F: RunnerMinimizer<Runner, Data>,
    S: RunnerMinimizer<Runner, Data>,
    Data: InputData,
    Runner: InputRunner<Data>,
{
    type IterResult<'a,IE> = ComposedIterator<'a, F, S,IE, Runner, Data>
    where
        Runner: 'a,
        Self: 'a,
        IE: Iterator<Item = Evaluation> + Clone+'a;

    fn generate_iterator<'a, IE>(
        &'a self,
        runner: &'a Runner,
        last_evaluations: IE,
    ) -> Self::IterResult<'a, IE>
    where
        IE: Iterator<Item = Evaluation> + Clone + 'a,
    {
        ComposedIterator::new(&self.first, &self.second, runner, last_evaluations)
    }

    fn buget(&self) -> u64 {
        self.first.buget() + self.second.buget()
    }
}

pub struct ComposedIterator<'a, F, S, EI, Runner, Data>
where
    F: RunnerMinimizer<Runner, Data> + 'a,
    S: RunnerMinimizer<Runner, Data> + 'a,
    Data: InputData,
    Runner: InputRunner<Data> + 'a,
    EI: Iterator<Item = Evaluation> + Clone + 'a,
{
    first_iterator: F::IterResult<'a, EI>,
    second_minimizer: &'a S,
    second_iterator:
        Option<S::IterResult<'a, std::iter::Chain<EI, std::vec::IntoIter<Evaluation>>>>,
    evaluations: Option<EI>,
    first_iterator_evaluations: Option<Vec<Evaluation>>,
    runner: &'a Runner,
}

impl<'a, F, S, EI, Runner, Data> ComposedIterator<'a, F, S, EI, Runner, Data>
where
    F: RunnerMinimizer<Runner, Data> + 'a,
    S: RunnerMinimizer<Runner, Data> + 'a,
    Data: InputData,
    Runner: InputRunner<Data> + 'a,
    EI: Iterator<Item = Evaluation> + Clone,
{
    fn new(first: &'a F, second: &'a S, runner: &'a Runner, last_evaluations: EI) -> Self {
        let first_iterator = first.generate_iterator(runner, last_evaluations.clone());
        Self {
            first_iterator,
            second_minimizer: second,
            second_iterator: None,
            evaluations: Some(last_evaluations),
            runner,
            first_iterator_evaluations: Some(Vec::new()),
        }
    }
}

impl<'a, F, S, EI, Runner, Data> Iterator for ComposedIterator<'a, F, S, EI, Runner, Data>
where
    F: RunnerMinimizer<Runner, Data>,
    S: RunnerMinimizer<Runner, Data>,
    Data: InputData,
    Runner: InputRunner<Data>,
    EI: Iterator<Item = Evaluation> + Clone + 'a,
{
    type Item = (Vec<Value>, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((input, output)) = self.first_iterator.next() {
            self.first_iterator_evaluations
                .as_mut()
                .unwrap()
                .push(Evaluation {
                    input: input.iter().map(|v| v.to_f64()).collect(),
                    output,
                });
            return Some((input, output));
        }
        if let Some(ref mut iter) = self.second_iterator {
            return iter.next();
        }
        let first_iterator_evaluations =
            Option::take(&mut self.first_iterator_evaluations).unwrap();

        let evaluations = Option::take(&mut self.evaluations)
            .unwrap()
            .chain(first_iterator_evaluations.into_iter());
        self.second_iterator = Some(
            self.second_minimizer
                .generate_iterator(self.runner, evaluations),
        );
        self.second_iterator.as_mut().unwrap().next()
    }
}
