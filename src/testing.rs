use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
use ndarray::{array, Array2, ArrayView2};
struct MyProblem {}

// Implement `CostFunction` for `MyProblem`
impl CostFunction for MyProblem {
    // [...]
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock_2d(p, 1.0, 100.0))
    }
}

// Implement `Gradient` for `MyProblem`
impl Gradient for MyProblem {
    // [...]
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Gradient = Vec<f64>;

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
    }
}

fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
    (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
}

use egobox_ego::EgorBuilder;
fn run() -> Result<(), Error> {
    let res = EgorBuilder::optimize(xsinx)
        .min_within(&array![[0.0, 25.0]])
        .n_iter(10)
        .run()
        .expect("xsinx minimized");

    // print result
    println!("{:?}", res);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test1() {
        if let Err(ref e) = run() {
            println!("{}", e);
            std::process::exit(1);
        }
    }
}
