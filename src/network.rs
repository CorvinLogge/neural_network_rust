use std::collections::VecDeque;
use std::fs;
use std::fs::File;
use std::path::Path;

use byteorder::{BigEndian, ByteOrder, WriteBytesExt};
use indicatif::ProgressIterator;
use nalgebra::{DMatrix, Dyn, U1, VecStorage, Vector};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rocket::yansi::Paint;

use crate::data_sets::{DataPoint, read_emnist};
use crate::function::{ActivationFunction, equ, Function, softmax};
use crate::function::ActivationFunction::{RELU, SIGMOID};
use crate::layer::{column_mean, Layer};
use crate::plotter::FilePlotter;

#[derive(Clone, Debug)]
pub(crate) struct Network {
    pub(crate) layers: Vec<Layer>,
    layer_specs: Vec<usize>,
    lr: f32,
    id: String,
}

impl Network {
    pub(crate) fn new(layer_specs: &[usize], lr: f32, activation: ActivationFunction) -> Network {
        let mut layers: Vec<Layer> = Vec::new();

        for index in 0..layer_specs.len() - 1 {
            layers.push(Layer::new(layer_specs[index], layer_specs[index + 1], activation));
        }

        Network {
            layers,
            layer_specs: layer_specs.to_vec(),
            lr,
            id: chrono::offset::Local::now().format("%d%m%Y%H%M%S").to_string(),
        }
    }

    pub fn relu_sig(layer_specs: &[usize], lr: f32) -> Network {
        let mut layers: Vec<Layer> = Vec::new();

        for index in 0..layer_specs.len() - 1 {
            if index == layer_specs.len() - 2 {
                layers.push(Layer::new(layer_specs[index], layer_specs[index + 1], SIGMOID));
            } else {
                layers.push(Layer::new(layer_specs[index], layer_specs[index + 1], RELU));
            }
        }

        Network {
            layers,
            layer_specs: layer_specs.to_vec(),
            lr,
            id: chrono::offset::Local::now().format("%d%m%Y%H%M%S").to_string(),
        }
    }

    pub(crate) fn feedforward(&mut self, inputs: &Vec<f32>) -> DMatrix<f32> {
        let mut out: DMatrix<f32> = DMatrix::<f32>::from_vec(inputs.len(), 1, inputs.clone());

        for layer in self.layers.iter_mut() {
            out = layer.feedforward(out);
        }

        return out;
    }

    pub(crate) fn feedforward_batch(&mut self, mut inputs: DMatrix<f32>) -> DMatrix<f32> {
        for layer in self.layers.iter_mut() {
            inputs = layer.feedforward(inputs);
        }

        return inputs;
    }

    pub fn guess(&self, inputs: &Vec<f32>) -> DMatrix<f32> {
        let mut out: DMatrix<f32> = DMatrix::<f32>::from_vec(inputs.len(), 1, inputs.clone());

        for layer in &self.layers {
            out = layer.guess(out);
        }

        return out;
    }

    pub(crate) fn backpropagation(&mut self, inputs: &Vec<f32>, targets: &Vec<f32>) {
        let input_values: DMatrix<f32> = DMatrix::<f32>::from_vec(inputs.len(), 1, inputs.clone());
        let target_values: DMatrix<f32> = DMatrix::<f32>::from_vec(targets.len(), 1, targets.clone());

        let outputs: DMatrix<f32> = self.feedforward(&inputs);
        let mut error: DMatrix<f32> = DMatrix::<f32>::zeros(0, 0);

        for index in (0..self.layers.len()).rev() {
            let prime_activated_net: DMatrix<f32> = self.layers.get(index).unwrap().get_net().map(self.layers.get(index).unwrap().get_activation().derivative());

            let mut delta_biases: DMatrix<f32>;
            let mut delta_weights: DMatrix<f32>;

            if index == self.layers.len() - 1 {
                let loss_derivative: DMatrix<f32> = &outputs - &target_values;

                error = loss_derivative.clone().component_mul(&prime_activated_net);

                delta_biases = error.clone();

                let error_t: DMatrix<f32> = error.transpose();

                delta_weights = self.layers.get(index - 1).unwrap().get_outputs() * error_t;
            } else {
                let left_side: DMatrix<f32> = self.layers.get(index + 1).unwrap().get_weights() * error;

                error = left_side.component_mul(&prime_activated_net);

                delta_biases = error.clone();

                let error_t: DMatrix<f32> = error.transpose();

                if index == 0 {
                    delta_weights = &input_values * error_t;
                } else {
                    delta_weights = self.layers.get(index - 1).unwrap().get_outputs() * error_t;
                }
            }

            delta_weights = delta_weights.scale(-&self.lr);
            delta_biases = delta_biases.scale(-&self.lr);

            self.layers.get_mut(index).unwrap().update_weights(delta_weights);
            self.layers.get_mut(index).unwrap().update_biases(delta_biases);
        }
    }

    pub(crate) fn backpropagation_batches(&mut self, batch: &[DataPoint]) {
        let mut inputs = DMatrix::<f32>::zeros(batch.get(0).unwrap().input.len(), batch.len());
        let mut targets = DMatrix::<f32>::zeros(batch.get(0).unwrap().target.len(), batch.len());

        for (index, data_point) in batch.iter().enumerate() {
            let input = Vector::from_vec_storage(VecStorage::new(Dyn(batch.get(0).unwrap().input.len()), U1, data_point.input.clone()));
            inputs.set_column(index, &input);

            let input = Vector::from_vec_storage(VecStorage::new(Dyn(batch.get(0).unwrap().target.len()), U1, data_point.target.clone()));
            targets.set_column(index, &input);
        }

        let outputs: DMatrix<f32> = self.feedforward_batch(inputs.clone());
        let mut error = DMatrix::<f32>::zeros(0, 0);

        for index in (0..self.layers.len()).rev() {
            let prime_activated_net = self.layers.get(index).unwrap().get_net().map(self.layers.get(index).unwrap().get_activation().derivative());

            let mut delta_biases: DMatrix<f32>;
            let mut delta_weights: DMatrix<f32>;

            if index == self.layers.len() - 1 {
                let loss_derivative = (&outputs - &targets)/* * (1f32 / batch.len() as f32)*/;

                error = loss_derivative.clone().component_mul(&prime_activated_net).cast();

                delta_biases = column_mean(error.clone());

                delta_weights = self.layers.get(index - 1).unwrap().get_outputs() * error.transpose();
            } else {
                let left_side: DMatrix<f32> = self.layers.get(index + 1).unwrap().get_weights() * error;

                error = left_side.component_mul(&prime_activated_net);

                delta_biases = column_mean(error.clone());

                if index == 0 {
                    delta_weights = inputs.clone() * error.transpose();
                } else {
                    delta_weights = self.layers.get(index - 1).unwrap().get_outputs() * error.transpose();
                }
            }

            delta_weights *= -&self.lr / (batch.len() as f32);
            delta_biases *= -&self.lr / (batch.len() as f32);

            self.layers.get_mut(index).unwrap().update_weights(delta_weights);
            self.layers.get_mut(index).unwrap().update_biases(delta_biases);
        }
    }

    pub fn train_batches(&mut self, data: String, epochs: usize, batch_size: usize) {
        let mut data_points = read_emnist(format!("{data}train/images"), format!("{data}train/labels"));

        data_points.shuffle(&mut thread_rng());

        for _ in (0..epochs).progress() {
            let batches = data_points.chunks(batch_size);

            for batch in batches.progress() {
                self.backpropagation_batches(batch)
            }

            println!("{}", self.profile_str(0.95, data.clone()));
        }
    }

    pub fn train_batches_plot(&mut self, data: String, epochs: usize, batch_size: usize) -> Vec<u8> {
        let mut file_plotter = FilePlotter::new(self.id.clone(), epochs as u32);
        let mut buffer: Vec<u8> = vec![0u8; 640 * 480 * 3];

        let mut data_points = read_emnist(format!("{data}train/images"), format!("{data}train/labels"));

        data_points.shuffle(&mut thread_rng());

        for epoch in (0..epochs).progress() {
            let batches = data_points.chunks(batch_size);

            for batch in batches.progress() {
                self.backpropagation_batches(batch)
            }

            file_plotter.plot((epoch as f32, self.profile(0.95, data.clone()).accuracy * 100.0), &mut *buffer);
        }

        buffer
    }

    pub(crate) fn train(&mut self, data: String, iterations: usize) {
        let mut data_points = read_emnist(format!("{data}train/images"), format!("{data}train/labels"));

        for _ in (0..iterations).progress() {
            data_points.shuffle(&mut thread_rng());

            for data_point in data_points.iter() {
                self.backpropagation(&data_point.input, &data_point.target);
            }
        }
    }

    pub(crate) fn train_supervised(&mut self, data: String, iterations: usize, interval: usize) {
        let mut data_points = read_emnist(format!("{data}train/images"), format!("{data}train/labels"));

        for iteration in (0..iterations).progress() {
            data_points.shuffle(&mut thread_rng());

            for data_point in data_points.iter().progress() {
                self.backpropagation(&data_point.input, &data_point.target);
            }

            if (iteration % interval) == 0 {
                println!("{}", self.profile_str(0.95, data.clone()));
            }
        }
        println!("{}", self.profile_str(0.95, data.clone()));
    }

    pub fn profile(&self, tolerance: f32, data: String) -> ProfileResult {
        let test_images_path = format!("{data}test/images");
        let test_labels_path = format!("{data}test/labels");

        let testing_data = read_emnist(test_images_path, test_labels_path);

        let mut successes = 0;
        let mut fails = 0;

        for data_point in testing_data.iter() {
            let mut out = self.guess(&data_point.input);

            out = out.map(equ(&out));

            let expected = DMatrix::<f32>::from_vec(data_point.target.len(), 1, data_point.target.clone());

            let ex_index = expected.iter().position(|x| { *x == 1.0 }).unwrap();

            if out.get(ex_index).unwrap() > &tolerance {
                successes += 1;
            } else { fails += 1; }
        }

        ProfileResult::new(self.id.clone(), tolerance, successes, fails, successes as f32 / (fails + successes) as f32)
    }

    pub fn profile_str(&mut self, tolerance: f32, data: String) -> String {
        let profile_result = self.profile(tolerance, data);

        format!("Profiling Result for model: '{}'\n\
                                      with tolerance: {}\n\
                                      Successes: {}\n\
                                      Failures: {}\n\
                                      Accuracy: {}", profile_result.model, profile_result.tolerance, profile_result.successes, profile_result.failures, profile_result.accuracy)
    }

    pub fn relu_sig_from_file(file_path: String) -> Network {
        let network_path = format!("{file_path}/network");
        let mut network_bytes = VecDeque::from(fs::read(network_path).unwrap());

        let layer_specs_len = byteorder::BigEndian::read_u32(network_bytes.drain(..4).collect::<Vec<u8>>().as_slice());

        let mut layer_specs = Vec::new();
        let mut layers = Vec::new();

        for _ in 0..layer_specs_len {
            layer_specs.push(byteorder::BigEndian::read_u32(network_bytes.drain(..4).collect::<Vec<u8>>().as_slice()));
        }


        for index in 0..layer_specs.len() - 1 {
            let num_ins = *layer_specs.get(index).unwrap();
            let num_outs = *layer_specs.get(index + 1).unwrap();

            let mut weight_data: Vec<f32> = Vec::new();
            let mut bias_data: Vec<f32> = Vec::new();

            for _ in 0..(num_ins * num_outs) {
                weight_data.push(byteorder::BigEndian::read_f32(network_bytes.drain(..4).collect::<Vec<u8>>().as_slice()));
            }

            for _ in 0..num_outs {
                bias_data.push(byteorder::BigEndian::read_f32(network_bytes.drain(..4).collect::<Vec<u8>>().as_slice()));
            }

            let weights = DMatrix::<f32>::from_vec(num_ins as usize, num_outs as usize, weight_data);
            let biases = DMatrix::<f32>::from_vec(num_outs as usize, 1, bias_data);

            if index == layer_specs.len() - 2 {
                layers.push(Layer::from(weights, biases, SIGMOID));
            } else {
                layers.push(Layer::from(weights, biases, RELU));
            }
        }

        let lr = byteorder::BigEndian::read_f32(network_bytes.drain(..4).collect::<Vec<u8>>().as_slice());


        Network {
            layers,
            layer_specs: layer_specs.iter().map(move |x| { *x as usize }).collect(),
            lr,
            id: Path::new(file_path.as_str()).file_name().unwrap().to_str().unwrap().to_string(),
        }
    }

    // File Format:
    // First 4 Bytes -> number of layers + 1 as u32
    // Next 4 * (number of layers + 1) Bytes -> number of inputs/outputs to layer
    // Next (inputs * outputs) Bytes -> Weights of first layer
    // Next (outputs * 1) Bytes -> Biases of first layer
    // Repeat for all layers
    // Last 4 Bytes -> learn rate as f32
    pub fn save_to_path(&self, dir_path: &str) -> String {
        let id = &self.id;
        let folder_path_string = format!("{dir_path}/{id}");
        let folder_path = Path::new(folder_path_string.as_str());
        fs::create_dir_all(folder_path).unwrap();

        let network_path_string = format!("{folder_path_string}/network");
        let network_path = Path::new(network_path_string.as_str());

        let mut network_file = File::create(network_path).unwrap();

        network_file.write_u32::<BigEndian>(self.layer_specs.len() as u32).unwrap();

        for spec in &self.layer_specs {
            network_file.write_u32::<BigEndian>(*spec as u32).unwrap();
        }

        for layer in self.layers.iter() {
            for value in layer.get_weights().data.as_slice() {
                network_file.write_f32::<BigEndian>(*value).unwrap()
            }

            for value in layer.get_biases().data.as_slice() {
                network_file.write_f32::<BigEndian>(*value).unwrap()
            }
        }

        network_file.write_f32::<BigEndian>(self.lr).unwrap();

        self.id.clone()
    }


    pub fn save_to_def_path(&self) -> String {
        self.save_to_path("./resources/models")
    }
}

struct ProfileResult {
    model: String,
    tolerance: f32,
    successes: u32,
    failures: u32,
    accuracy: f32,
}

impl ProfileResult {
    pub fn new(model: String, tolerance: f32, successes: u32, failures: u32, accuracy: f32) -> ProfileResult {
        ProfileResult {
            model,
            tolerance,
            successes,
            failures,
            accuracy,
        }
    }
}