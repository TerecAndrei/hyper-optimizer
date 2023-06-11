use std::{collections::HashSet, marker::PhantomData};

use crate::{
    library::{Domain, InputData, InputDataExt, InputRunner, Value},
    output::Evaluation,
    runners::ClonableIterator,
};
use itertools::Itertools;

use super::RunnerMinimizer;

pub struct BoxedRunner<Runner, Data> {
    _data: PhantomData<(Runner, Data)>,
}

impl<Runner, Data> BoxedRunner<Runner, Data>
where
    Runner: InputRunner<Data>,
    Data: InputData + 'static,
{
    pub fn new(minimizer: impl RunnerMinimizer<Runner, Data>) -> Self {
        Self { _data: PhantomData }
    }
}

impl<Runner, Data> RunnerMinimizer<Runner, Data> for BoxedRunner<Runner, Data>
where
    Runner: InputRunner<Data>,
    Data: InputData + 'static,
{
    type IterResult<'a, IE> = ClonableIterator<(Vec<Value>, f64)>
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
        todo!()
    }

    fn buget(&self) -> u64 {
        0
    }
}
