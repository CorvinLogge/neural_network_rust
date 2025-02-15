use crate::data_sets::{read_emnist_test, read_emnist_train, DataPoint, DataSet};
use crate::error::{Error, ErrorKind};
use crate::function::ErrorFunction::*;
use crate::function::{ActivationFunction, ErrorFunction};
use crate::layer::{column_mean, expand_columns, row_mean, Layer};
use crate::optimizers::Optimizer;
use crate::NET_ID_PATTERN;
use byteorder::{BigEndian, ByteOrder, WriteBytesExt};
use indicatif::ProgressIterator;
use nalgebra::{DMatrix, Dyn, VecStorage, Vector, U1};
use num_traits::{Float, Pow};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::ParallelIterator;
use rocket::http::Status;
use rocket::serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::fs::File;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Network {
    layers: Vec<Layer>,
    lr: f32,
    #[serde(skip)]
    id: String,
    error: ErrorFunction,
    optimizer: Optimizer,
    data_set: DataSet,
}

impl Network {
    pub fn init(&mut self) {
        self.layers.iter_mut().for_each(|mut layer| layer.init());
        self.id = chrono::offset::Local::now()
            .format(NET_ID_PATTERN)
            .to_string();
    }

    pub fn feedforward(&mut self, mut inputs: &DMatrix<f32>) -> DMatrix<f32> {
        let mut inputs = inputs.clone();
        for layer in self.layers.iter_mut() {
            inputs = layer.feedforward(&inputs);
        }

        inputs
    }

    pub fn guess(&self, inputs: &Vec<f32>) -> DMatrix<f32> {
        let mut out: DMatrix<f32> = DMatrix::<f32>::from_vec(inputs.len(), 1, inputs.clone());

        for layer in &self.layers {
            out = layer.guess(out);
        }

        out
    }

    pub fn backpropagation_batches(
        &mut self,
        batch: &[DataPoint],
        beta_1: f32,
        beta_2: f32,
        t: usize,
    ) -> Result<(), Error> {
        let (inputs, targets) = combine_batch(batch);

        let mut error: DMatrix<f32> = DMatrix::<f32>::zeros(0, 0);

        let outputs: DMatrix<f32> = self.feedforward(&inputs);

        for index in (0..self.layers.len()).rev() {
            let activation_derivative = self.layers[index].activation().derivative();
            let net = self.layers[index].net();
            let prime_activated_net = net.map(activation_derivative);

            let mut bias_gradient: DMatrix<f32>;
            let mut weight_gradient: DMatrix<f32>;
            let mut delta_bias: DMatrix<f32> = DMatrix::<f32>::zeros(0, 0);
            let mut delta_weights: DMatrix<f32> = DMatrix::<f32>::zeros(0, 0);

            //Only output layer
            if index == self.layers.len() - 1 {
                let loss_derivative = match self.error {
                    MSE => &outputs - &targets,
                    CrossEntropy => {
                        let left = targets.clone().component_div(&outputs.clone());
                        let right = targets
                            .clone()
                            .map(move |v| 1f32 - v)
                            .component_div(&outputs.clone().map(move |v| 1f32 - v));

                        right - left
                    }
                    L1 => {
                        let weights = self.layers.last().unwrap().weights().clone();
                        let abs_sum = weights.clone().apply_into(|val| *val = val.abs()).sum();
                        let norm_weights = (1.0 / abs_sum) * &weights;
                        let row_mean = row_mean(norm_weights);
                        let expanded_columns = expand_columns(outputs.ncols(), &row_mean);

                        let error =
                            &outputs - &targets + (1.0 / batch.len() as f32) * expanded_columns;

                        error
                    }
                    L2 => {
                        let weights = self.layers.last().unwrap().weights();
                        let abs_sum = weights
                            .clone()
                            .apply_into(|val| *val = val.pow(2))
                            .sum()
                            .sqrt();
                        let norm_weights = (1.0 / abs_sum) * weights;
                        let row_mean = row_mean(norm_weights);
                        let expanded_columns = expand_columns(outputs.ncols(), &row_mean);

                        let error =
                            &outputs - &targets + (1.0 / batch.len() as f32) * expanded_columns;

                        error
                    }
                };

                error = loss_derivative.component_mul(&prime_activated_net).cast();

                bias_gradient = column_mean(error.clone());

                weight_gradient = self.layers[index - 1].outputs() * error.transpose();
            } else {
                let left_side: DMatrix<f32> = self.layers[index + 1].weights() * &error;

                error = left_side.component_mul(&prime_activated_net);

                bias_gradient = column_mean(error.clone());

                if index == 0 {
                    weight_gradient = inputs.clone() * error.transpose();
                } else {
                    weight_gradient = self.layers[index - 1].outputs() * error.transpose();
                }
            }

            let layer = self.layers.get_mut(index).ok_or(Error::back_prop())?;

            let weight_velocity = layer.weight_velocity();
            let bias_velocity = layer.bias_velocity();

            let weight_moment = layer.weight_moment();

            match self.optimizer {
                Optimizer::Sgd => {
                    delta_weights = weight_gradient;
                    delta_bias = bias_gradient;
                }
                Optimizer::SgdM => {
                    let new_weight_velocity =
                        beta_1 * weight_velocity + (1.0 - beta_1) * weight_gradient;
                    let new_bias_velocity = beta_1 * bias_velocity + (1.0 - beta_1) * bias_gradient;

                    layer.set_weight_velocity(&new_weight_velocity);
                    layer.set_bias_velocity(&new_bias_velocity);

                    delta_weights = new_weight_velocity;
                    delta_bias = new_bias_velocity;
                }
                Optimizer::RmsProp => {
                    let weight_gradient_squared: DMatrix<f32> =
                        weight_gradient.component_mul(&weight_gradient);
                    let mut new_weight_velocity =
                        beta_1 * weight_velocity + (1.0 - beta_1) * &weight_gradient_squared;
                    new_weight_velocity.apply(|val| *val = val.sqrt() + 10e-8);

                    delta_weights = weight_gradient.component_div(&new_weight_velocity);
                    delta_bias = bias_gradient;

                    layer.set_weight_velocity(&new_weight_velocity);
                }
                Optimizer::Adam => {
                    let new_weight_moment =
                        beta_1 * weight_moment + (1.0 - beta_1) * &weight_gradient;
                    let new_weight_moment_hat =
                        new_weight_moment.scale(1.0 / (1.0 - beta_1.pow(t as f32)));

                    let weight_gradient_squared: DMatrix<f32> =
                        weight_gradient.component_mul(&weight_gradient);
                    let new_weight_velocity =
                        beta_2 * weight_velocity + (1.0 - beta_2) * &weight_gradient_squared;
                    let mut new_weight_velocity_hat =
                        new_weight_velocity.scale(1.0 / (1.0 - beta_2.pow(t as f32)));
                    new_weight_velocity_hat.apply(|val| *val = val.sqrt() + 10e-8);

                    delta_weights = new_weight_moment_hat.component_div(&new_weight_velocity_hat);
                    delta_bias = bias_gradient;

                    layer.set_weight_velocity(&new_weight_velocity);
                    layer.set_weight_moment(&new_weight_moment);
                }
            }

            delta_weights *= -&self.lr / (batch.len() as f32);
            delta_bias *= -&self.lr / (batch.len() as f32);

            layer.update_weights(&delta_weights);
            layer.update_biases(&delta_bias);
        }

        Ok(())
    }

    pub fn train_batches(&mut self, epochs: usize, batch_size: usize) -> Result<(), Error> {
        self.train_batches_plot(epochs, batch_size)?;

        Ok(())
    }

    pub fn train_batches_plot(
        &mut self,
        epochs: usize,
        batch_size: usize,
    ) -> Result<Vec<(f32, f32)>, Error> {
        let data_path = self.data_set.path();
        let num_classes = self.data_set.num_classes();

        let mut plot_points: Vec<(f32, f32)> = Vec::with_capacity(epochs);

        let mut train_data = read_emnist_train(&data_path, num_classes)?;
        let test_data = read_emnist_test(&data_path, num_classes)?;

        let mut t = 0;

        for epoch in (0..epochs).progress() {
            train_data.shuffle(&mut thread_rng());

            let batches = train_data.chunks(batch_size);

            for batch in batches {
                if batch.len() == batch_size {
                    t += 1;
                    self.backpropagation_batches(
                        batch,
                        0.9 / batch_size as f32,
                        0.99 / batch_size as f32,
                        t,
                    )?
                }
            }

            plot_points.push((
                epoch as f32,
                self.profile(0.95, &test_data)?.accuracy * 100.0,
            ));
        }

        Ok(plot_points)
    }

    pub fn profile(
        &self,
        tolerance: f32,
        testing_data: &Vec<DataPoint>,
    ) -> Result<ProfileResult, Error> {
        let mut successes = 0;
        let mut fails = 0;

        for data_point in testing_data.iter() {
            let mut out = self.guess(&data_point.input());

            let expected = data_point.target() as usize;

            if out.get(expected).ok_or(Error::profile())? > &tolerance {
                successes += 1;
            } else {
                fails += 1;
            }
        }

        Ok(ProfileResult::new(
            self.id.clone(),
            tolerance,
            successes,
            fails,
            successes as f32 / testing_data.len() as f32,
        ))
    }

    pub fn from_file(file_path: &String) -> Result<Network, Error> {
        let bytes = Self::get_file_bytes(file_path)?;
        Network::try_from(bytes)
    }
    fn get_file_bytes(file_path: &String) -> Result<VecDeque<u8>, Error> {
        let network_path = format!("./{file_path}");
        let bytes = fs::read(&network_path).map_err(|e| {
            Error::new(
                ErrorKind::IO,
                format!("{}. Could not find file {}", e.to_string(), &network_path),
                Status::InternalServerError,
                Some(Box::new(e)),
            )
        })?;
        Ok(VecDeque::from(bytes))
    }

    /// Saves the entire network to a file in a custom format (for now)
    ///
    /// File Format:
    ///
    /// First 4 Bytes -> number of layers + 1 as u32
    ///
    /// Next 4 * (number of layers + 1) Bytes -> number of inp
    ///
    /// Next (inputs * outputs) Bytes -> Weights of first layer
    ///
    /// Next (outputs * 1) Bytes -> Biases of first layer
    ///
    /// Next byte is a u8 that deserializes to an ActivationFunction
    ///
    /// Repeat for all layers
    ///
    /// Next 4 Bytes -> learn rate as f32
    ///
    /// Next byte is a u8 that deserializes to an Error Function
    ///
    /// Next byte is a u8 that deserializes to an Optimizer
    ///
    /// Next byte is a u8 that deserializes to a DataSet
    ///
    pub fn save_to_path(&self, dir_path: &str) -> Result<String, Error> {
        let folder_path_string = format!("{dir_path}/{}", self.id);
        let folder_path = Path::new(folder_path_string.as_str());
        fs::create_dir_all(folder_path)?;

        let network_path_string = format!("{folder_path_string}/network");
        let network_path = Path::new(network_path_string.as_str());

        let mut network_file = File::create(network_path)?;

        let mut v = vec![0; self.layers.len() + 1];
        let mut layer_specs = v.as_mut_slice();

        for (index, layer) in self.layers.iter().enumerate() {
            layer_specs[index] = *layer.ins();
            layer_specs[index + 1] = *layer.outs();
        }

        network_file.write_u32::<BigEndian>(layer_specs.len() as u32)?;

        for spec in layer_specs {
            network_file.write_u32::<BigEndian>(*spec as u32)?;
        }

        for layer in self.layers.iter() {
            for value in layer.weights().data.as_slice() {
                network_file.write_f32::<BigEndian>(*value)?
            }

            for value in layer.biases().data.as_slice() {
                network_file.write_f32::<BigEndian>(*value)?
            }

            network_file.write_u8(*layer.activation() as u8)?;
        }

        network_file.write_f32::<BigEndian>(self.lr)?;

        network_file.write_u8(self.error as u8)?;

        network_file.write_u8(self.optimizer as u8)?;

        network_file.write_u8(self.data_set as u8)?;

        Ok(self.id.clone())
    }

    pub fn save_to_def_path(&self) -> Result<String, Error> {
        Ok(self.save_to_path("./resources/models")?)
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    pub fn error(&self) -> ErrorFunction {
        self.error
    }

    pub fn optimizer(&self) -> Optimizer {
        self.optimizer
    }

    pub fn data_set(&self) -> DataSet {
        self.data_set
    }
}

impl TryFrom<VecDeque<u8>> for Network {
    type Error = Error;

    fn try_from(mut value: VecDeque<u8>) -> Result<Self, Self::Error> {
        let layer_specs_len =
            byteorder::BigEndian::read_u32(value.drain(..4).collect::<Vec<u8>>().as_slice());

        let mut layer_specs = Vec::with_capacity(layer_specs_len as usize);
        let mut layers = Vec::with_capacity((layer_specs_len - 1) as usize);

        for _ in 0..layer_specs_len {
            layer_specs.push(byteorder::BigEndian::read_u32(
                value.drain(..4).collect::<Vec<u8>>().as_slice(),
            ));
        }

        for index in 0..layer_specs.len() - 1 {
            let num_ins = *layer_specs.get(index).ok_or(Error::io(
                format!("Failed reading number of inputs for layer {index}"),
                None,
            ))?;
            let num_outs = *layer_specs.get(index + 1).ok_or(Error::io(
                format!("Failed reading number of outputs for layer {index}"),
                None,
            ))?;

            let mut weight_data: Vec<f32> = Vec::with_capacity((num_ins * num_outs) as usize);
            let mut bias_data: Vec<f32> = Vec::with_capacity(num_outs as usize);

            for _ in 0..(num_ins * num_outs) {
                weight_data.push(byteorder::BigEndian::read_f32(
                    value.drain(..4).collect::<Vec<u8>>().as_slice(),
                ));
            }

            for _ in 0..num_outs {
                bias_data.push(byteorder::BigEndian::read_f32(
                    value.drain(..4).collect::<Vec<u8>>().as_slice(),
                ));
            }

            let weights =
                DMatrix::<f32>::from_vec(num_ins as usize, num_outs as usize, weight_data);
            let biases = DMatrix::<f32>::from_vec(num_outs as usize, 1, bias_data);

            let activation: ActivationFunction =
                ActivationFunction::try_from(value.pop_front().ok_or(Error::io(
                    "Failed reading activation function from file".to_string(),
                    None,
                ))?)?;

            layers.push(Layer::from(weights, biases, activation));
        }

        let lr = byteorder::BigEndian::read_f32(value.drain(..4).collect::<Vec<u8>>().as_slice());

        let error: ErrorFunction = ErrorFunction::try_from(value.pop_front().ok_or(Error::io(
            "Failed reading error function from file".to_string(),
            None,
        ))?)?;

        let optimizer: Optimizer = Optimizer::try_from(value.pop_front().ok_or(Error::io(
            "Failed reading optimizer from file".to_string(),
            None,
        ))?)?;

        let data_set: DataSet = DataSet::try_from(value.pop_front().ok_or(Error::io(
            "Failed reading data set from file".to_string(),
            None,
        ))?)?;

        Ok(Network {
            layers,
            lr,
            id: "embedded".to_string(),
            error,
            data_set,
            optimizer,
        })
    }
}

fn combine_batch(batch: &[DataPoint]) -> (DMatrix<f32>, DMatrix<f32>) {
    let len_inputs = batch.first().unwrap().input().len();
    let len_target = batch.first().unwrap().target_vec().len();

    let mut inputs: DMatrix<f32> = DMatrix::<f32>::zeros(len_inputs, batch.len());
    let mut targets: DMatrix<f32> = DMatrix::<f32>::zeros(len_target, batch.len());

    let mut inputs = inputs;
    let mut targets = targets;

    for (index, data_point) in batch.iter().enumerate() {
        let input = Vector::from_vec_storage(VecStorage::new(
            Dyn(len_inputs),
            U1,
            data_point.input().as_slice().to_vec(),
        ));
        inputs.set_column(index, &input);

        let input = Vector::from_vec_storage(VecStorage::new(
            Dyn(len_target),
            U1,
            data_point.target_vec().as_slice().to_vec(),
        ));
        targets.set_column(index, &input);
    }

    (inputs, targets)
}

#[derive(Debug, Serialize)]
#[serde(crate = "rocket::serde")]
pub struct ProfileResult {
    model: String,
    tolerance: f32,
    successes: u32,
    failures: u32,
    accuracy: f32,
}

impl ProfileResult {
    pub fn new(
        model: String,
        tolerance: f32,
        successes: u32,
        failures: u32,
        accuracy: f32,
    ) -> ProfileResult {
        ProfileResult {
            model,
            tolerance,
            successes,
            failures,
            accuracy,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[rustfmt::skip]
pub enum ActivationMode {
    #[serde(alias = "sigmoid")] Sigmoid,
    #[serde(alias = "relu")] Relu,
    #[serde(alias = "relu_sig")] ReluSig,
}
