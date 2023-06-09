// use std::collections::HashSet;

// use crate::{
//     library::{Domain, InputData, InputDataExt, InputRunner, Value},
//     output::Evaluation,
//     runners::ClonableIterator,
// };
// use itertools::Itertools;

// use super::RunnerMinimizer;

// pub enum GridRunnerStep {
//     One(usize),
//     ForEach(Vec<usize>),
// }

// impl GridRunnerStep {
//     pub fn get_steps_size(&self, domains: &[Domain]) -> Vec<usize> {
//         match self {
//             Self::One(step) => vec![*step; domains.len()],
//             Self::ForEach(v) => {
//                 if v.len() != domains.len() {
//                     panic!(
//                         "Expected a domains size of {}. Received {} in grid search",
//                         v.len(),
//                         domains.len()
//                     );
//                 }
//                 v.clone()
//             }
//         }
//     }
// }

// impl From<usize> for GridRunnerStep {
//     fn from(value: usize) -> Self {
//         Self::One(value)
//     }
// }

// impl From<Vec<usize>> for GridRunnerStep {
//     fn from(value: Vec<usize>) -> Self {
//         Self::ForEach(value)
//     }
// }

// pub struct BoxedRunner<Runner, Data> {
//     minimizer: Box<
//         dyn RunnerMinimizer<
//             Runner,
//             Data,
//             IterResult<'static, Vec<Evaluation>> = ClonableIterator<(Vec<Value>, f64)>,
//         >,
//     >,
// }

// impl<Runner, Data> BoxedRunner<Runner, Data>
// where
//     Runner: InputRunner<Data>,
//     Data: InputData + 'static,
// {
//     pub fn new(minimizer: impl RunnerMinimizer<Runner, Data>) -> Self {
//         Self {
//             minimizer: Box::new(minimizer),
//         }
//     }
// }

// impl<Runner, Data> RunnerMinimizer<Runner, Data> for BoxedRunner<Runner, Data>
// where
//     Runner: InputRunner<Data>,
//     Data: InputData + 'static,
// {
//     type IterResult<'a, IE> = Box<dyn Iterator<Item = (Vec<Value>, f64)> + 'a>
//         where
//             Runner: 'a,
//             Self: 'a,
//             IE: Iterator<Item = Evaluation> + Clone + 'a,;
//     fn generate_iterator<'a, IE>(
//         &'a self,
//         runner: &'a Runner,
//         _evaluations: IE,
//     ) -> Self::IterResult<'a, IE>
//     where
//         IE: Iterator<Item = Evaluation> + Clone + 'a,
//     {
//     }

//     fn buget(&self) -> u64 {
//         0
//     }
// }
