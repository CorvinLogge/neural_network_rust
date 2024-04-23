use nalgebra::DMatrix;

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
            weights: DMatrix::<f32>::from_fn(ins, outs, *activation.weight_initialization(ins)),
            biases: DMatrix::<f32>::from_fn(outs, 1, *activation.bias_initialization(ins)),
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
        let mut out = self.get_weights().transpose() * inp;

        if out.shape() == self.get_biases().shape() {
            out += self.get_biases();
        } else {
            out += Self::expand_columns(out.ncols(), self.get_biases());
        }

        self.net = out.clone();

        out = out.map(*self.activation.original());

        self.outputs = out.clone();

        return out;
    }

    pub fn guess(&self, inp: DMatrix<f32>) -> DMatrix<f32> {
        let mut out = self.get_weights().transpose() * inp;

        if out.shape() == self.get_biases().shape() {
            out += self.get_biases();
        } else {
            out += Self::expand_columns(out.ncols(), self.get_biases());
        }

        out = out.map(*self.activation.original());

        return out;
    }

    pub(crate) fn update_weights(&mut self, delta_weights: DMatrix<f32>) {
        self.weights += delta_weights;
    }

    pub(crate) fn update_biases(&mut self, delta_biases: DMatrix<f32>) {
        self.biases += delta_biases;
    }

    pub(crate) fn get_weights(&self) -> DMatrix<f32> {
        return self.weights.clone();
    }
    pub(crate) fn get_biases(&self) -> DMatrix<f32> {
        return self.biases.clone();
    }
    pub(crate) fn get_net(&self) -> DMatrix<f32> {
        //DMatrix::<f32>::from_vec(self.net.nrows(), 1, self.net.column_mean().as_slice().to_vec())
        self.net.clone()
    }
    pub(crate) fn get_outputs(&self) -> DMatrix<f32> {
        //DMatrix::<f32>::from_vec(self.outputs.nrows(), 1, self.outputs.column_mean().as_slice().to_vec())
        self.outputs.clone()
    }
    pub fn get_activation(&self) -> ActivationFunction {
        self.activation.clone()
    }

    fn expand_columns(ncols: usize, mat: DMatrix<f32>) -> DMatrix<f32> {
        let vec = mat.as_slice().to_vec();

        let mut new_vec = Vec::new();

        for _ in 0..ncols {
            new_vec.append(&mut vec.clone());
        }

        DMatrix::<f32>::from_vec(vec.len(), ncols, new_vec)
    }
}

pub fn column_mean(mat: DMatrix<f32>) -> DMatrix<f32> {
    DMatrix::<f32>::from_vec(mat.nrows(), 1, mat.column_mean().as_slice().to_vec())
}