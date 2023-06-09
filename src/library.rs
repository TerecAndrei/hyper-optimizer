use std::ops::{Bound, Range, RangeBounds};

use rand::Rng;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum Value {
    Float(f64),
    Integer(i32),
}

impl Value {
    pub fn to_f64(&self) -> f64 {
        match self {
            Self::Float(v) => *v as f64,
            Self::Integer(v) => *v as f64,
        }
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Float(value)
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Value::Integer(value)
    }
}

pub struct DomainBuilder {
    domains: Vec<Domain>,
    names: Vec<String>,
}

impl DomainBuilder {
    pub(crate) fn new() -> Self {
        DomainBuilder {
            domains: Vec::new(),
            names: Vec::new(),
        }
    }

    pub fn add_f64_range(mut self, range: Range<f64>, name: String) -> Self {
        self.domains.push(Domain::Continuos(range));
        self.names.push(name);
        self
    }

    pub fn add_i32_range(mut self, range: Range<i32>, name: String) -> Self {
        self.domains.push(Domain::Discrete(range));
        self.names.push(name);
        self
    }

    pub(crate) fn into_inner(self) -> (Vec<Domain>, Vec<String>) {
        (self.domains, self.names)
    }
}

pub struct InputDeserializer<T: Iterator<Item = Value>> {
    values: T,
}

impl<T: Iterator<Item = Value>> InputDeserializer<T> {
    pub(crate) fn new(values: T) -> Self {
        Self { values }
    }
    pub fn next_i32(&mut self) -> i32 {
        match self.values.next().unwrap() {
            Value::Integer(v) => v,
            Value::Float(x) => panic!("Expected an i32 value. Received {}", x),
        }
    }

    pub fn next_f64(&mut self) -> f64 {
        match self.values.next().unwrap() {
            Value::Float(v) => v,
            Value::Integer(x) => panic!("Expected an f64 value. Received {}", x),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "type", content = "range")]
pub enum Domain {
    Discrete(Range<i32>),
    Continuos(Range<f64>),
}

impl Domain {
    pub(crate) fn random_value(&self, r: &mut impl Rng) -> Value {
        match self.clone() {
            Domain::Continuos(range) => Value::Float(r.gen_range(range)),
            Domain::Discrete(range) => Value::Integer(r.gen_range(range)),
        }
    }

    pub(crate) fn range(&self) -> Range<f64> {
        match self {
            Domain::Continuos(range) => range.clone(),
            Domain::Discrete(range) => range.start as f64..range.end as f64,
        }
    }
}

pub trait InputData: Clone {
    fn from_deserializer<T: Iterator<Item = Value>>(deserializer: InputDeserializer<T>) -> Self;
    fn get_domains(domains: DomainBuilder) -> DomainBuilder;
}

pub(crate) trait InputDataExt: InputData {
    fn get_domains_ext() -> Vec<Domain>;
    fn get_names_ext() -> Vec<String>;
    fn get_domains_and_names() -> (Vec<Domain>, Vec<String>);
    fn from_values<I>(values: I) -> Self
    where
        I: IntoIterator<Item = Value>;
    fn from_f64<I>(values: I) -> Self
    where
        I: IntoIterator<Item = f64>;
}

impl<Data> InputDataExt for Data
where
    Data: InputData,
{
    fn from_values<I>(values: I) -> Self
    where
        I: IntoIterator<Item = Value>,
    {
        let deserializer = InputDeserializer::new(values.into_iter());
        let data = Data::from_deserializer(deserializer);
        data
    }

    fn from_f64<I>(values: I) -> Self
    where
        I: IntoIterator<Item = f64>,
    {
        let domains = Data::get_domains_ext();
        let iter = values.into_iter().zip(domains).map(|(v, d)| match d {
            Domain::Continuos(_) => Value::Float(v),
            Domain::Discrete(_) => Value::Integer(v as i32),
        });
        let deserializer = InputDeserializer::new(iter);
        let data = Data::from_deserializer(deserializer);
        data
    }

    fn get_domains_ext() -> Vec<Domain> {
        Data::get_domains(DomainBuilder::new()).into_inner().0
    }

    fn get_names_ext() -> Vec<String> {
        Data::get_domains(DomainBuilder::new()).into_inner().1
    }
    fn get_domains_and_names() -> (Vec<Domain>, Vec<String>) {
        Data::get_domains(DomainBuilder::new()).into_inner()
    }
}

pub trait InputRunner<Data: InputData> {
    fn run(&self, data: Data) -> f64;
}

impl<T, Data> InputRunner<Data> for T
where
    T: Fn(Data) -> f64,
    Data: InputData,
{
    fn run(&self, data: Data) -> f64 {
        self(data)
    }
}

pub fn test_runner<R: InputRunner<Data>, Data: InputData>(r: R) {
    let domains = Data::get_domains(DomainBuilder::new()).into_inner().0;

    for _ in 0..100 {
        let data = domains
            .iter()
            .map(|d| {
                let mut r = rand::thread_rng();
                match d.clone() {
                    Domain::Continuos(range) => Value::Float(r.gen_range(range)),
                    Domain::Discrete(range) => Value::Integer(r.gen_range(range)),
                }
            })
            .collect::<Vec<_>>();

        let data = Data::from_deserializer(InputDeserializer {
            values: data.into_iter(),
        });
        let _output = r.run(data);
    }
}
