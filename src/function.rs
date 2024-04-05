use std::clone::Clone;
use std::f32::consts::E;

use nalgebra::DMatrix;
use num_traits::Pow;
use rand::{random, Rng, thread_rng};
use serde::{Deserialize, Serialize};

const RELU: fn(f32) -> f32 = |x| f32::max(0.0, x);
const RELU_P: fn(f32) -> f32 = |x| { return if x <= 0.0 { 0.0 } else { 1.0 }; };
const HE_INIT: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> = |ins| Box::new(move |_, _| ((random::<f32>() * 2.0) - 1.0) * f32::sqrt(2.0 / ins as f32));

const SIGMOID: fn(f32) -> f32 = |x| 1.0 / (1.0 + f32::pow(E, x));
const SIGMOID_P: fn(f32) -> f32 = |x| SIGMOID(x) * (1.0 - SIGMOID(x));
const XAVIER_INIT: fn(usize) -> Box<dyn Fn(usize, usize) -> f32> = |ins| Box::new(move |_, _| thread_rng().gen_range(-(1.0 / f32::sqrt(ins as f32))..(1.0 / f32::sqrt(ins as f32))));

pub fn softmax(mat: &DMatrix<f32>) -> Box<dyn Fn(f32) -> f32> {
    let sum: f32 = mat.iter().map(|v| { E.pow(v) }).sum();
    Box::new(move |v| (E.pow(v) / sum))
}

pub trait Function {
    fn derivative(&self) -> Box<impl Fn(f32) -> f32>;
    fn original(&self) -> Box<impl Fn(f32) -> f32>;
    fn initialization(&self, ins: usize) -> Box<impl Fn(usize, usize) -> f32>;
}

#[derive(Clone, Copy)]
#[derive(Debug)]
#[derive(Serialize, Deserialize)]
pub enum ActivationFunction {
    RELU,
    SIGMOID,
}

impl Function for ActivationFunction {
    fn derivative(&self) -> Box<impl Fn(f32) -> f32> {
        match self {
            ActivationFunction::RELU => { Box::new(RELU_P) }
            ActivationFunction::SIGMOID => { Box::new(SIGMOID_P) }
        }
    }

    fn original(&self) -> Box<impl Fn(f32) -> f32> {
        match self {
            ActivationFunction::RELU => { Box::new(RELU) }
            ActivationFunction::SIGMOID => { Box::new(SIGMOID) }
        }
    }

    fn initialization(&self, ins: usize) -> Box<impl Fn(usize, usize) -> f32> {
        match self {
            ActivationFunction::RELU => { Box::new(HE_INIT(ins)) }
            ActivationFunction::SIGMOID => { Box::new(XAVIER_INIT(ins)) }
        }
    }
}