use std::clone::Clone;
use std::f32::consts::E;

use nalgebra::DMatrix;
use num_traits::Pow;
use rand::{random, Rng, thread_rng};
use rocket::serde::{Deserialize, Serialize};

const L_RELU: fn(f32) -> f32 = |x| {
    if x >= 0f32 {
        x
    } else {
        0.025 * x
    }
};
const L_RELU_P: fn(f32) -> f32 = |x| {
    return if x < 0.0 { 0.025 } else { 1.0 };
};

const RELU: fn(f32) -> f32 = |x| f32::max(0.0, x);
const RELU_P: fn(f32) -> f32 = |x| { return if x <= 0.0 { 0.0 } else { 1.0 }; };
const HE_INIT: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> = |ins| Box::new(move |_, _| ((random::<f32>() * 2.0) - 1.0) * f32::sqrt(2.0 / ins as f32));
const RAND_0_1: fn(usize, usize) -> f32 = |_, _| random::<f32>();
const RAND_0_1_: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> = |_| Box::new(move |_, _| random::<f32>());
const RAND_1_1: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> = |_| Box::new(move |_, _| thread_rng().gen_range(-1..1) as f32);


const SIGMOID: fn(f32) -> f32 = |x| 1f32 / (1f32 + E.pow(-x));
const SIGMOID_P: fn(f32) -> f32 = |x| SIGMOID(x) * (1f32 - SIGMOID(x));
const XAVIER_INIT: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> = |ins| Box::new(move |_, _| thread_rng().gen_range(-(1f32 / f32::sqrt(ins as f32))..(1f32 / f32::sqrt(ins as f32))));
const ZERO: fn(usize, usize) -> f32 = |_, _| 0.0;

pub fn softmax(mat: &DMatrix<f32>) -> Box<dyn Fn(f32) -> f32> {
    let sum: f32 = mat.iter().map(|v| { E.pow(v) }).sum();
    Box::new(move |v| E.pow(v) / sum)
}

pub fn equ(mat: &DMatrix<f32>) -> Box<dyn Fn(f32) -> f32> {
    let sum: f32 = mat.sum();
    Box::new(move |v| v / sum)
}

#[derive(Clone, Copy)]
#[derive(Debug)]
#[derive(Serialize, Deserialize)]
#[derive(PartialEq)]
pub enum ErrorFunction {
    MSE,
    CrossEntropy,
}

impl ErrorFunction {
    pub fn derivative(&self, outputs: &DMatrix<f32>, targets: &DMatrix<f32>) -> DMatrix<f32> {
        match self {
            ErrorFunction::MSE => { outputs - targets }
            ErrorFunction::CrossEntropy => {
                let left = targets.clone().component_div(&outputs.clone());
                let right = targets.clone().map(move |v| 1f32 - v).component_div(&outputs.clone().map(move |v| 1f32 - v));

                right - left
            }
        }
    }
}

#[derive(Clone, Copy)]
#[derive(Debug)]
#[derive(Serialize, Deserialize)]
#[derive(PartialEq)]
pub enum ActivationFunction {
    RELU,
    LRELU,
    SIGMOID,
}

impl ActivationFunction {
    pub(crate) fn derivative(&self) -> Box<impl Fn(f32) -> f32> {
        match self {
            ActivationFunction::RELU => { Box::new(RELU_P) }
            ActivationFunction::LRELU => { Box::new(L_RELU_P) }
            ActivationFunction::SIGMOID => { Box::new(SIGMOID_P) }
        }
    }

    pub(crate) fn original(&self) -> Box<impl Fn(f32) -> f32> {
        match self {
            ActivationFunction::RELU => { Box::new(RELU) }
            ActivationFunction::LRELU => { Box::new(L_RELU) }
            ActivationFunction::SIGMOID => { Box::new(SIGMOID) }
        }
    }

    pub(crate) fn weight_initialization(&self, ins: usize) -> Box<impl Fn(usize, usize) -> f32> {
        match self {
            ActivationFunction::RELU => { Box::new(HE_INIT(ins)) }
            ActivationFunction::LRELU => { Box::new(HE_INIT(ins)) }
            ActivationFunction::SIGMOID => { Box::new(XAVIER_INIT(ins)) }
        }
    }

    pub(crate) fn bias_initialization(&self, ins: usize) -> Box<impl Fn(usize, usize) -> f32> {
        match self {
            ActivationFunction::RELU => { Box::new(RAND_0_1) }
            ActivationFunction::LRELU => { Box::new(ZERO) }
            ActivationFunction::SIGMOID => { Box::new(ZERO) }
        }
    }
}