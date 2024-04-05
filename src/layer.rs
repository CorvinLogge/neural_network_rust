use std::ops::Add;

use nalgebra::DMatrix;
use rand::random;

use crate::function::{ActivationFunction, Function};

#[derive(Debug)]
#[derive(Clone)]
pub(crate) struct Layer {
    weights: DMatrix<f32>,
    biases: DMatrix<f32>,
    net: DMatrix<f32>,
    outputs: DMatrix<f32>,
    activation: ActivationFunction,
}

impl Layer {
    pub fn new(ins: usize, outs: usize, activation: ActivationFunction) -> Layer {
        Layer {
            weights: DMatrix::<f32>::from_fn(ins, outs, *activation.initialization(ins)),
            biases: DMatrix::<f32>::from_fn(outs, 1, |_, _| random::<f32>()),
            net: DMatrix::<f32>::zeros(outs, 1),
            outputs: DMatrix::<f32>::zeros(outs, 1),
            activation,
        }
    }

    pub fn from(weights: DMatrix<f32>, biases: DMatrix<f32>, activation: ActivationFunction) -> Layer {
        Layer {
            weights: weights.clone(),
            biases: biases.clone(),
            net: DMatrix::<f32>::zeros(weights.ncols(), 1),
            outputs: DMatrix::<f32>::zeros(weights.ncols(), 1),
            activation,
        }
    }

    pub(crate) fn feedforward(&mut self, inp: DMatrix<f32>) -> DMatrix<f32> {
        let weights_t = self.get_weights().transpose();

        let mut out = weights_t * inp;
        out = out.add(self.get_biases());

        self.net = out.clone();

        out = out.map(*self.activation.original());

        self.outputs = out.clone();

        return out.clone();
    }

    pub(crate) fn update_weights(&mut self, delta_weights: DMatrix<f32>) {
        self.weights = self.get_weights().add(delta_weights);
    }

    pub(crate) fn update_biases(&mut self, delta_biases: DMatrix<f32>) {
        self.biases = self.get_biases().add(delta_biases);
    }

    pub(crate) fn get_weights(&self) -> DMatrix<f32> {
        return self.weights.clone();
    }
    pub(crate) fn get_biases(&self) -> DMatrix<f32> {
        return self.biases.clone();
    }
    pub(crate) fn get_net(&self) -> DMatrix<f32> {
        return self.net.clone();
    }
    pub(crate) fn get_outputs(&self) -> DMatrix<f32> {
        return self.outputs.clone();
    }
}