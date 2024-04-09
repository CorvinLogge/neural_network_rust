use std::path::Path;
use std::collections::VecDeque;
use std::fs;
use std::fs::File;

use byteorder::{BigEndian, ByteOrder, WriteBytesExt};
use indicatif::ProgressIterator;
use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rocket::yansi::Paint;

use crate::data_point::DataPoint;
use crate::emnist_parser::read_emnist;
use crate::function::{ActivationFunction, Function};
use crate::function::ActivationFunction::RELU;
use crate::layer::Layer;

const EMNIST_PATH: &str = "./resources/emnist_digits/";

#[derive(Clone, Debug)]
pub(crate) struct Network {
    pub(crate) layers: Vec<Layer>,
    layer_specs: Vec<usize>,
    lr: f32,
    activation: ActivationFunction,
    id: String,
    training_iterations: usize,
}

impl Network {
    pub(crate) fn new(layer_specs: &[usize], lr: f32, activation: ActivationFunction) -> Self {
        let mut layers: Vec<Layer> = Vec::new();

        for index in 0..layer_specs.len() - 1 {
            layers.push(Layer::new(layer_specs[index] as usize, layer_specs[index + 1] as usize, activation))
        }

        Self {
            layers,
            layer_specs: layer_specs.to_vec(),
            lr,
            activation,
            id: chrono::offset::Local::now().format("%d.%m.%Y_%H-%M-%S").to_string(),
            training_iterations: 0,
        }
    }

    pub(crate) fn feedforward(&mut self, inputs: &Vec<f32>) -> DMatrix<f32> {
        let mut out: DMatrix<f32> = DMatrix::<f32>::from_vec(inputs.len(), 1, inputs.clone());

        for layer in self.layers.iter_mut() {
            out = layer.feedforward(out);
        }

        return out;
    }


    pub(crate) fn backpropagation(&mut self, inputs: &Vec<f32>, targets: &Vec<f32>) {
        let input_values: DMatrix<f32> = DMatrix::<f32>::from_vec(inputs.len(), 1, inputs.clone());
        let target_values: DMatrix<f32> = DMatrix::<f32>::from_vec(targets.len(), 1, targets.clone());

        let outputs: DMatrix<f32> = self.feedforward(&inputs);
        let mut error: DMatrix<f32> = DMatrix::<f32>::zeros(0, 0);

        for index in (0..self.layers.len()).rev() {
            let prime_activated_net: DMatrix<f32> = self.layers.get(index).unwrap().get_net().map(self.activation.derivative());

            let mut delta_biases: DMatrix<f32>;
            let mut delta_weights: DMatrix<f32>;

            if index == self.layers.len() - 1 {
                let cost_derivative: DMatrix<f32> = &outputs - &target_values;

                error = cost_derivative.clone().component_mul(&prime_activated_net);

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

    pub(crate) fn train(&mut self, data_points: &mut Vec<DataPoint>, iterations: usize) {
        self.training_iterations += iterations;

        for _ in (0..iterations).progress() {
            data_points.shuffle(&mut thread_rng());

            for data_point in data_points.iter() {
                self.backpropagation(&data_point.input, &data_point.target);
            }
        }
    }

    pub(crate) fn profile_save(&mut self, tolerance: f32) {
        let mut test_images_path = EMNIST_PATH.to_owned();
        test_images_path.push_str("test/images");
        let mut test_labels_path = EMNIST_PATH.to_owned();
        test_labels_path.push_str("test/labels");

        let testing_data = read_emnist(&*test_images_path, &*test_labels_path);

        let mut successes = 0;
        let mut fails = 0;

        for data_point in testing_data.iter().progress() {
            let mut out = self.feedforward(&data_point.input);

            let expected = DMatrix::<f32>::from_vec(data_point.target.len(), 1, data_point.target.clone());

            let ex_index = expected.iter().position(|x| { *x == 1.0 }).unwrap();

            if out.get(ex_index).unwrap() > &tolerance {
                successes += 1;
            } else { fails += 1; }
        }

        let profile = format!("Profiling Result for model: '{}'\n\
                                      with tolerance: {tolerance}\n\
                                      trained with:\n\
                                        learn rate: {}\n\
                                        iterations: {}\n\
                                        layer specs: {:?}\n\
                                      \n\
                                      Successes: {successes}\n\
                                      Failures: {fails}\n\
                                      Accuracy: {}", self.id, self.lr, self.training_iterations, self.layer_specs, successes as f32 / (fails + successes) as f32);

        let profile_path = format!("./resources/profiles/{}.txt", self.id);

        fs::write(profile_path, &profile).unwrap();

        println!("{}", &profile);
    }

    pub fn profile_str(&mut self, tolerance: f32) -> String {
        let mut test_images_path = EMNIST_PATH.to_owned();
        test_images_path.push_str("test/images");
        let mut test_labels_path = EMNIST_PATH.to_owned();
        test_labels_path.push_str("test/labels");

        let testing_data = read_emnist(&*test_images_path, &*test_labels_path);

        let mut successes = 0;
        let mut fails = 0;

        for data_point in testing_data.iter().progress() {
            let mut out = self.feedforward(&data_point.input);

            let expected = DMatrix::<f32>::from_vec(data_point.target.len(), 1, data_point.target.clone());

            let ex_index = expected.iter().position(|x| { *x == 1.0 }).unwrap();

            if out.get(ex_index).unwrap() > &tolerance {
                successes += 1;
            } else { fails += 1; }
        }

        format!("Profiling Result for model: '{}'\n\
                                      with tolerance: {tolerance}\n\
                                      trained with:\n\
                                        learn rate: {}\n\
                                        iterations: {}\n\
                                        layer specs: {:?}\n\
                                      \n\
                                      Successes: {successes}\n\
                                      Failures: {fails}\n\
                                      Accuracy: {}", self.id, self.lr, self.training_iterations, self.layer_specs, successes as f32 / (fails + successes) as f32)
    }

    pub fn from_file(file_path: &str) -> Network {
        let paths = fs::read_dir(file_path).unwrap();

        let mut layers: Vec<Layer> = Vec::new();
        let mut layer_specs: Vec<usize> = Vec::new();

        let its = ((&paths.count() - 1) / 2);

        for index in 0..its {
            let weights_path_string = format!("{file_path}/weights_{index}");
            let mut weight_bytes = VecDeque::from(fs::read(weights_path_string).unwrap());

            let num_rows_w = byteorder::BigEndian::read_u32(weight_bytes.drain(..4).collect::<Vec<u8>>().as_slice());
            let num_cols_w = byteorder::BigEndian::read_u32(weight_bytes.drain(..4).collect::<Vec<u8>>().as_slice());

            let mut weight_data: Vec<f32> = Vec::new();

            while !weight_bytes.is_empty() {
                weight_data.push(byteorder::BigEndian::read_f32(weight_bytes.drain(..4).collect::<Vec<u8>>().as_slice()));
            }

            let weights = DMatrix::<f32>::from_vec(num_rows_w as usize, num_cols_w as usize, weight_data);

            let biases_path_string = format!("{file_path}/biases_{index}");
            let mut biases_bytes = VecDeque::from(fs::read(biases_path_string).unwrap());

            let num_rows_b = byteorder::BigEndian::read_u32(biases_bytes.drain(..4).collect::<Vec<u8>>().as_slice());
            let num_cols_b = byteorder::BigEndian::read_u32(biases_bytes.drain(..4).collect::<Vec<u8>>().as_slice());

            let mut bias_data: Vec<f32> = Vec::new();

            while !biases_bytes.is_empty() {
                bias_data.push(byteorder::BigEndian::read_f32(biases_bytes.drain(..4).collect::<Vec<u8>>().as_slice()));
            }

            let biases = DMatrix::<f32>::from_vec(num_rows_b as usize, num_cols_b as usize, bias_data);

            layers.push(Layer::from(weights, biases, RELU));

            layer_specs.push(num_rows_w as usize);

            if index == its - 1 { layer_specs.push(num_rows_b as usize) }
        }


        let lr_path = format!("{file_path}/lr");
        let lr = byteorder::BigEndian::read_f32(fs::read(lr_path).unwrap().as_slice());

        Network {
            layers,
            layer_specs,
            lr,
            activation: RELU,
            id: Path::new(file_path).file_name().unwrap().to_str().unwrap().to_string(),
            training_iterations: 0,
        }
    }

    // File format: first 4 bytes are number of rows, next 4 are number of columns every next 4 bytes are the values
    pub fn save_to_path(&self, dir_path: &str) -> String
    {
        let id = &self.id;
        let folder_path_string = format!("{dir_path}/{id}");
        let folder_path = Path::new(folder_path_string.as_str());
        fs::create_dir_all(folder_path).unwrap();

        for (index, layer) in self.layers.iter().enumerate().progress() {
            let weights_path_string = format!("{folder_path_string}/weights_{index}");
            let biases_path_string = format!("{folder_path_string}/biases_{index}");
            let weights_path = Path::new(weights_path_string.as_str());
            let biases_path = Path::new(biases_path_string.as_str());

            let mut weights_file = File::create(weights_path).unwrap();
            let mut biases_file = File::create(biases_path).unwrap();

            weights_file.write_u32::<BigEndian>(layer.get_weights().shape().0 as u32).unwrap();
            weights_file.write_u32::<BigEndian>(layer.get_weights().shape().1 as u32).unwrap();

            for value in layer.get_weights().data.as_slice() {
                weights_file.write_f32::<BigEndian>(*value).unwrap()
            }

            biases_file.write_u32::<BigEndian>(layer.get_biases().shape().0 as u32).unwrap();
            biases_file.write_u32::<BigEndian>(layer.get_biases().shape().1 as u32).unwrap();

            for value in layer.get_biases().data.as_slice() {
                biases_file.write_f32::<BigEndian>(*value).unwrap()
            }
        }

        let lr_path_string = format!("{folder_path_string}/lr");
        let mut lr_file = File::create(Path::new(lr_path_string.as_str())).unwrap();

        lr_file.write_f32::<BigEndian>(self.lr).unwrap();

        self.id.clone()
    }
    pub fn save_to_def_path(&self) -> String {
        self.save_to_path("./resources/models")
    }
}