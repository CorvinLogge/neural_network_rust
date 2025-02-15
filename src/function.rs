use crate::error::Error;
use nalgebra::DMatrix;
use num_traits::Pow;
use rand::{random, thread_rng, Rng};
use rocket::serde::{Deserialize, Serialize};
use std::clone::Clone;
use std::f32::consts::E;

const L_RELU: fn(f32) -> f32 = |x| if x >= 0f32 { x } else { 0.025 * x };
const L_RELU_P: fn(f32) -> f32 = |x| if x < 0.0 { 0.025 } else { 1.0 };

const RELU: fn(f32) -> f32 = |x| f32::max(0.0, x);
const RELU_P: fn(f32) -> f32 = |x| {
    return if x <= 0.0 { 0.0 } else { 1.0 };
};
const HE_INIT: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> =
    |ins| Box::new(move |_, _| ((random::<f32>() * 2.0) - 1.0) * f32::sqrt(2.0 / ins as f32));

const RAND_0_1: fn(usize, usize) -> f32 = |_, _| random::<f32>();
const RAND_0_1_: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> =
    |_| Box::new(move |_, _| random::<f32>());
const RAND_1_1: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> =
    |_| Box::new(move |_, _| thread_rng().gen_range(-1..1) as f32);
const ZERO: fn(usize, usize) -> f32 = |_, _| 0.0;

const SIGMOID: fn(f32) -> f32 = |x| 1f32 / (1f32 + E.pow(-x));
const SIGMOID_P: fn(f32) -> f32 = |x| SIGMOID(x) * (1f32 - SIGMOID(x));
const XAVIER_INIT: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> = |ins| {
    Box::new(move |_, _| {
        thread_rng().gen_range(-(1f32 / f32::sqrt(ins as f32))..(1f32 / f32::sqrt(ins as f32)))
    })
};

pub fn softmax(mat: &DMatrix<f32>) -> Box<dyn Fn(f32) -> f32> {
    let sum: f32 = mat.iter().map(|v| E.pow(v)).sum();
    Box::new(move |v| E.pow(v) / sum)
}

pub fn equ(mat: &DMatrix<f32>) -> Box<dyn Fn(f32) -> f32> {
    let sum: f32 = mat.sum();
    Box::new(move |v| v / sum)
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum ErrorFunction {
    #[serde(alias = "mse")]
    MSE = 0,
    #[serde(alias = "cross_entropy")]
    CrossEntropy = 1,
    #[serde(alias = "l1")]
    L1 = 2,
    #[serde(alias = "l2")]
    L2 = 3,
}

impl TryFrom<u8> for ErrorFunction {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Error> {
        match value {
            0 => Ok(ErrorFunction::MSE),
            1 => Ok(ErrorFunction::CrossEntropy),
            2 => Ok(ErrorFunction::L1),
            3 => Ok(ErrorFunction::L2),
            _ => Err(Error::unknown_variant(vec![0, 1, 2, 3], value)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[rustfmt::skip]
pub enum ActivationFunction {
    #[serde(alias = "relu")] RELU = 0,
    #[serde(alias = "lrelu")] LRELU = 1,
    #[serde(alias = "sigmoid")] SIGMOID = 2,
}

impl ActivationFunction {
    pub(crate) fn derivative(&self) -> Box<impl Fn(f32) -> f32> {
        match self {
            ActivationFunction::RELU => Box::new(RELU_P),
            ActivationFunction::LRELU => Box::new(L_RELU_P),
            ActivationFunction::SIGMOID => Box::new(SIGMOID_P),
        }
    }

    pub(crate) fn original(&self) -> Box<impl Fn(f32) -> f32> {
        match self {
            ActivationFunction::RELU => Box::new(RELU),
            ActivationFunction::LRELU => Box::new(L_RELU),
            ActivationFunction::SIGMOID => Box::new(SIGMOID),
        }
    }

    pub(crate) fn weight_initialization(&self, ins: usize) -> Box<impl Fn(usize, usize) -> f32> {
        match self {
            ActivationFunction::RELU => Box::new(HE_INIT(ins)),
            ActivationFunction::LRELU => Box::new(HE_INIT(ins)),
            ActivationFunction::SIGMOID => Box::new(XAVIER_INIT(ins)),
        }
    }

    pub(crate) fn bias_initialization(&self) -> Box<impl Fn(usize, usize) -> f32> {
        match self {
            ActivationFunction::RELU => Box::new(RAND_0_1),
            ActivationFunction::LRELU => Box::new(ZERO),
            ActivationFunction::SIGMOID => Box::new(ZERO),
        }
    }
}

impl TryFrom<u8> for ActivationFunction {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ActivationFunction::RELU),
            1 => Ok(ActivationFunction::LRELU),
            2 => Ok(ActivationFunction::SIGMOID),
            _ => Err(Error::unknown_variant(vec![0, 1, 2], value)),
        }
    }
}
