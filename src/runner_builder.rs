// use std::{io::Write, marker::PhantomData, path::Path};

// use rand::{rngs::ThreadRng, Rng};

// use crate::{
//     library::{InputData, InputRunner},
//     runner::TestRunner,
// };

// pub struct NoOutputFile;
// pub struct OutputFile<P: AsRef<Path>> {
//     path: P,
//     overwrite: bool,
// }

// pub struct TestRunnerBuilder<R, Data, RngGenerator = ThreadRng, P = NoOutputFile>
// where
//     R: InputRunner<Data>,
//     Data: InputData,
//     RngGenerator: Rng + Clone,
// {
//     runner: R,
//     output_file: P,
//     _phantom_data: PhantomData<Data>,
//     rng: RngGenerator,
// }

// impl<R, Data> TestRunnerBuilder<R, Data>
// where
//     R: InputRunner<Data>,
//     Data: InputData,
// {
//     pub fn new(runner: R) -> Self {
//         Self {
//             runner,
//             output_file: NoOutputFile,
//             _phantom_data: PhantomData,
//             rng: rand::thread_rng(),
//         }
//     }
// }

// impl<R, Data, RngGenerator, MaybeOutputFile>
//     TestRunnerBuilder<R, Data, RngGenerator, MaybeOutputFile>
// where
//     R: InputRunner<Data>,
//     Data: InputData,
//     RngGenerator: Rng + Clone,
// {
//     pub fn output<P: AsRef<Path>>(
//         self,
//         path: P,
//     ) -> TestRunnerBuilder<R, Data, RngGenerator, OutputFile<P>> {
//         TestRunnerBuilder {
//             output_file: OutputFile {
//                 path,
//                 overwrite: false,
//             },
//             runner: self.runner,
//             rng: self.rng,
//             _phantom_data: PhantomData,
//         }
//     }
// }

// impl<R, Data, RngGenerator, P> TestRunnerBuilder<R, Data, RngGenerator, OutputFile<P>>
// where
//     R: InputRunner<Data>,
//     Data: InputData,
//     RngGenerator: Rng + Clone,
//     P: AsRef<Path>,
// {
//     pub fn overwrite(
//         mut self,
//         overwrite: bool,
//     ) -> TestRunnerBuilder<R, Data, RngGenerator, OutputFile<P>> {
//         self.output_file.overwrite = overwrite;
//         self
//     }
// }

// impl<R, Data, RngGenerator> TestRunnerBuilder<R, Data, RngGenerator, NoOutputFile>
// where
//     R: InputRunner<Data>,
//     Data: InputData,
//     RngGenerator: Rng + Clone,
// {
//     pub fn build(self) -> TestRunner<R, Data, RngGenerator> {
//         TestRunner::new(self.runner, self.rng)
//     }
// }

// impl<R, Data, RngGenerator, P> TestRunnerBuilder<R, Data, RngGenerator, OutputFile<P>>
// where
//     R: InputRunner<Data>,
//     Data: InputData,
//     RngGenerator: Rng + Clone,
//     P: AsRef<Path>,
// {
//     pub fn build(self) -> TestRunner<R, Data, RngGenerator> {
//         let file = std::fs::File::options()
//             .write(true)
//             .create(true)
//             .create_new(!self.output_file.overwrite)
//             .truncate(true)
//             .open(self.output_file.path)
//             .unwrap(); //TODO: Remove this unwrap call

//         TestRunner::new_with_output_file(self.runner, self.rng, file)
//     }
// }
