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

#[derive(Clone, Debug, Getters, Deserialize)]
pub struct Layer {
    #[serde(skip, default = "default_matrix")] weights: DMatrix<f32>,
    #[serde(skip, default = "default_matrix")] weight_velocity: DMatrix<f32>,
    #[serde(skip, default = "default_matrix")] weight_moment: DMatrix<f32>,
    #[serde(skip, default = "default_matrix")] biases: DMatrix<f32>,
    #[serde(skip, default = "default_matrix")] bias_velocity: DMatrix<f32>,
    #[serde(skip, default = "default_matrix")] net: DMatrix<f32>,
    #[serde(skip, default = "default_matrix")] outputs: DMatrix<f32>,
    ins: usize,
    outs: usize,
    activation: ActivationFunction,
    layer_type: LayerType,
}

fn default_matrix() -> DMatrix<f32> {
    DMatrix::<f32>::zeros(0, 0)
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
            ins: 0,
            outs: 0,
            activation: RELU,
            layer_type: FullyConnected,
        }
    }
}

impl Layer {
    pub fn init(&mut self) {
        self.weights = DMatrix::<f32>::from_fn(self.ins, self.outs, *self.activation.weight_initialization(self.ins));
        self.weight_velocity = DMatrix::<f32>::zeros(self.ins, self.outs);
        self.weight_moment = DMatrix::<f32>::zeros(self.ins, self.outs);
        self.biases = DMatrix::<f32>::from_fn(self.outs, 1, *self.activation.bias_initialization());
        self.bias_velocity = DMatrix::<f32>::zeros(self.outs, 1);
        self.net = DMatrix::<f32>::zeros(self.outs, 1);
        self.outputs = DMatrix::<f32>::zeros(self.outs, 1);
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
            ins: weights.nrows(),
            outs: weights.ncols(),
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
    #[serde(alias = "fc", alias = "fully_connected")] FullyConnected = 0,
    #[serde(alias = "conv", alias = "convolutional")] Convolutional = 1,
    #[serde(alias = "pass", alias = "pass_through")] PassThrough = 2,
    #[serde(alias = "drop", alias = "dropout")] Dropout = 3,
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
