use derive_getters::Getters;
use nalgebra::DMatrix;
use rand::distributions::{Bernoulli, Distribution};
use rand::thread_rng;
use rocket::http::Status;
use rocket::serde::{Deserialize, Serialize};

use crate::function::ActivationFunction;
use crate::function::ActivationFunction::RELU;
use crate::layer::LayerType::*;
use crate::utils::Error;

#[derive(Clone, Debug, Getters)]
pub struct Layer {
    weights: DMatrix<f32>,
    weight_velocity: DMatrix<f32>,
    weight_moment: DMatrix<f32>,
    biases: DMatrix<f32>,
    bias_velocity: DMatrix<f32>,
    net: DMatrix<f32>,
    outputs: DMatrix<f32>,
    activation: ActivationFunction,
    layer_type: LayerType,
}

impl Default for Layer {
    fn default() -> Self {
        Layer {
            weights: DMatrix::<f32>::zeros(0, 0),
            weight_velocity: DMatrix::<f32>::zeros(0, 0),
            weight_moment: DMatrix::<f32>::zeros(0, 0),
            biases: DMatrix::<f32>::zeros(0, 0),
            bias_velocity: DMatrix::<f32>::zeros(0, 0),
            net: DMatrix::<f32>::zeros(0, 0),
            outputs: DMatrix::<f32>::zeros(0, 0),
            activation: RELU,
            layer_type: FullyConnected,
        }
    }
}

impl Layer {
    pub fn new(ins: usize, outs: usize, activation: ActivationFunction, layer_type: LayerType) -> Layer {
        Layer {
            weights: DMatrix::<f32>::from_fn(ins, outs, *activation.weight_initialization(ins)),
            weight_velocity: DMatrix::<f32>::zeros(ins, outs),
            weight_moment: DMatrix::<f32>::zeros(ins, outs),
            biases: DMatrix::<f32>::from_fn(outs, 1, *activation.bias_initialization()),
            bias_velocity: DMatrix::<f32>::zeros(outs, 1),
            net: DMatrix::<f32>::zeros(outs, 1),
            outputs: DMatrix::<f32>::zeros(outs, 1),
            activation,
            layer_type,
        }
    }

    pub fn from_vec(layer_specs: &[usize], activation_functions: Vec<ActivationFunction>, layer_types: Vec<LayerType>) -> Vec<Layer> {
        let mut layers: Vec<Layer> = Vec::new();

        for index in 0..layer_specs.len() - 1 {
            if index == layer_specs.len() - 2 {
                layers.push(Layer::new(
                    layer_specs[index],
                    layer_specs[index + 1],
                    activation_functions[index],
                    layer_types[index],
                ));
            } else {
                layers.push(Layer::new(layer_specs[index], layer_specs[index + 1], RELU, layer_types[index]));
            }
        }

        layers
    }

    pub fn from(weights: DMatrix<f32>, biases: DMatrix<f32>, activation: ActivationFunction, layer_type: LayerType) -> Layer {
        Layer {
            weights: weights.clone(),
            weight_velocity: DMatrix::<f32>::zeros(weights.nrows(), weights.ncols()),
            weight_moment: DMatrix::<f32>::zeros(weights.nrows(), weights.ncols()),
            biases: biases.clone(),
            bias_velocity: DMatrix::<f32>::zeros(biases.nrows(), biases.ncols()),
            net: DMatrix::<f32>::zeros(weights.ncols(), 1),
            outputs: DMatrix::<f32>::zeros(weights.ncols(), 1),
            activation,
            layer_type,
        }
    }

    pub(crate) fn feedforward(&mut self, inp: DMatrix<f32>) -> DMatrix<f32> {
        return match self.layer_type {
            FullyConnected => {
                let mut out = self.weights().transpose() * inp;

                if out.shape() == self.biases().shape() {
                    out += self.biases();
                } else {
                    out += expand_columns(out.ncols(), self.biases());
                }

                self.net = out.clone();

                out = out.map(*self.activation.original());

                self.outputs = out.clone();

                out
            }
            Dropout => {
                let bernoulli = Bernoulli::new(0.8).unwrap();
                let bernoulli_mat = DMatrix::<f32>::from_fn(self.weights.nrows(), self.weights.ncols(), |_, _| f32::from(bernoulli.sample(&mut thread_rng())));

                let dropped_weights: DMatrix<f32> = self.weights.component_mul(&bernoulli_mat).cast();

                let mut out = dropped_weights.transpose() * inp;

                if out.shape() == self.biases().shape() {
                    out += self.biases();
                } else {
                    out += expand_columns(out.ncols(), self.biases());
                }

                self.net = out.clone();

                out = out.map(*self.activation.original());

                self.outputs = out.clone();

                out
            }
            Convolutional => {
                //batch size 1 only

                

                inp
            }
            PassThrough => {
                inp
            }
        };
    }

    pub fn guess(&self, inp: DMatrix<f32>) -> DMatrix<f32> {
        let mut out = self.weights().transpose() * inp;

        if out.shape() == self.biases().shape() {
            out += self.biases();
        } else {
            out += expand_columns(out.ncols(), self.biases());
        }

        out = out.map(*self.activation.original());

        return out;
    }

    pub fn update_weights(&mut self, delta_weights: &DMatrix<f32>) {
        self.weights += delta_weights;
    }

    pub fn update_biases(&mut self, delta_biases: &DMatrix<f32>) {
        self.biases += delta_biases;
    }

    pub fn set_weight_velocity(&mut self, weight_velocity: &DMatrix<f32>) {
        self.weight_velocity = weight_velocity.clone();
    }

    pub fn set_weight_moment(&mut self, weight_moment: &DMatrix<f32>) {
        self.weight_moment = weight_moment.clone();
    }

    pub fn set_bias_velocity(&mut self, bias_velocity: &DMatrix<f32>) {
        self.bias_velocity = bias_velocity.clone();
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum LayerType {
    FullyConnected = 0,
    Convolutional = 1,
    PassThrough = 2,
    Dropout = 3,
}

impl LayerType {
    pub fn from_u8(u: u8) -> Result<LayerType, Error> {
        match u {
            0 => { Ok(FullyConnected) }
            1 => { Ok(Convolutional) }
            2 => { Ok(PassThrough) }
            3 => { Ok(Dropout) }
            _ => { Err(Error::new(format!("unknown variant '{}', expected one of '0', '1', '2'", u), Status::BadRequest)) }
        }
    }
}

pub fn expand_columns(ncols: usize, mat: &DMatrix<f32>) -> DMatrix<f32> {
    let vec = mat.as_slice().to_vec();

    let mut new_vec = Vec::new();

    for _ in 0..ncols {
        new_vec.append(&mut vec.clone());
    }

    DMatrix::<f32>::from_vec(vec.len(), ncols, new_vec)
}

pub fn expand_rows(nrows: usize, mat: &DMatrix<f32>) -> DMatrix<f32> {
    let vec = mat.as_slice().to_vec();

    let mut new_vec = Vec::new();

    for _ in 0..nrows {
        new_vec.append(&mut vec.clone());
    }

    DMatrix::<f32>::from_vec(nrows, vec.len(), new_vec)
}

pub fn column_mean(mat: DMatrix<f32>) -> DMatrix<f32> {
    DMatrix::<f32>::from_vec(mat.nrows(), 1, mat.column_mean().as_slice().to_vec())
}

pub fn row_mean(mat: DMatrix<f32>) -> DMatrix<f32> {
    DMatrix::<f32>::from_vec(1, mat.ncols(), mat.row_mean().as_slice().to_vec())
}
