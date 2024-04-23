#[macro_use]
extern crate rocket;

use std::fmt::format;
use std::ops::Add;
use std::string::ToString;

use image::{ImageBuffer, Rgb};
use nalgebra::DMatrix;
use rocket::{Request, Response};
use rocket::fairing::{Fairing, Info, Kind};
use rocket::form::FromForm;
use rocket::http::Header;
use rocket::response::status::Accepted;
use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};

use crate::function::ActivationFunction::{LRELU, RELU, SIGMOID};
use crate::function::Function;
use crate::image_processor::ImageProcessor;
use crate::network::Network;

mod layer;
mod network;
mod data_sets;
mod function;
mod image_processor;
mod plotter;

const EMNIST_PATH: &str = "./resources/emnist_digits/";
const MNIST_PATH: &str = "./resources/mnist/";

#[derive(FromForm)]
struct TrainReq {
    iterations: usize,
    lr: f32,
    layer_specs: Vec<usize>,
}

#[derive(FromForm)]
struct TrainBatchReq {
    batch_size: usize,
    epochs: usize,
    lr: f32,
    layer_specs: Vec<usize>,
    activation_mode: String,
    data: String,
}

#[derive(Deserialize, Serialize)]
#[serde(crate = "rocket::serde")]
#[derive(Debug)]
struct GuessReq {
    network_id: String,
    image: String,
}

#[get("/train?<r>")]
fn train(r: TrainReq) -> Accepted<String> {
    let mut network = Network::new(r.layer_specs.as_slice(), r.lr, RELU);

    let data = EMNIST_PATH.to_string();

    network.train_supervised(data, r.iterations, 100_000);

    let id = network.save_to_def_path();

    Accepted(format!("Successfully trained model: {id}"))
}

#[get("/train/batches?<r>")]
fn train_batches(r: TrainBatchReq) -> Accepted<String> {
    let mut network;
    let mut data;

    match r.data.as_str() {
        "mnist_digits" => data = MNIST_PATH.to_owned(),
        "e_mnist_digits" => data = EMNIST_PATH.to_owned(),
        _ => data = EMNIST_PATH.to_owned(),
    }

    match r.activation_mode.as_str() {
        "relu_sig" => network = Network::relu_sig(r.layer_specs.as_slice(), r.lr),
        "relu" => network = Network::new(r.layer_specs.as_slice(), r.lr, RELU),
        "sigmoid" => network = Network::new(r.layer_specs.as_slice(), r.lr, SIGMOID),
        _ => network = Network::relu_sig(r.layer_specs.as_slice(), r.lr),
    }

    network.train_batches(data, r.epochs, r.batch_size);

    let id = network.save_to_def_path();

    Accepted(format!("Successfully trained model: {id}"))
}

#[get("/train/batches/plot?<r>")]
fn train_batches_plot(r: TrainBatchReq) -> Accepted<String> {
    let mut network;
    let mut data;

    match r.data.as_str() {
        "mnist_digits" => data = MNIST_PATH.to_owned(),
        "e_mnist_digits" => data = EMNIST_PATH.to_owned(),
        _ => data = EMNIST_PATH.to_owned(),
    }

    match r.activation_mode.as_str() {
        "relu_sig" => network = Network::relu_sig(r.layer_specs.as_slice(), r.lr),
        "relu" => network = Network::new(r.layer_specs.as_slice(), r.lr, RELU),
        "sigmoid" => network = Network::new(r.layer_specs.as_slice(), r.lr, SIGMOID),
        _ => network = Network::relu_sig(r.layer_specs.as_slice(), r.lr),
    }

    let plot = network.train_batches_plot(data, r.epochs, r.batch_size);

    let id = network.save_to_def_path();

    let image: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_vec(640, 480, plot).unwrap();
    image.save(format!("./resources/models/{id}/plot.png")).unwrap();


    Accepted(format!("Successfully trained model: {id}"))
}

#[get("/profile?<network_id>&<tolerance>&<data>")]
fn profile(network_id: &str, tolerance: f32, data: String) -> String {
    let mut data_path;

    match data.as_str() {
        "mnist_digits" => data_path = MNIST_PATH.to_owned(),
        "e_mnist_digits" => data_path = EMNIST_PATH.to_owned(),
        _ => data_path = EMNIST_PATH.to_owned(),
    }

    Network::relu_sig_from_file(format!("./resources/models/{network_id}")).profile_str(tolerance, data_path)
}

#[post("/guess", data = "<r>")]
fn guess(r: Json<GuessReq>) -> Json<Vec<f32>> {
    let mut network = Network::relu_sig_from_file(format!("./resources/models/{}", r.network_id));
    let guess = network.guess(&ImageProcessor::from_data_url(&*r.image));
    let data = guess.data.as_vec();
    Json::from(data.clone())
}

#[get("/test")]
fn test() -> String {
    "Connection worked\n".to_string()
}

#[options("/<_..>")]
fn all_options() {
    /* Intentionally left empty */
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/network", routes![train_batches, train_batches_plot, profile, test, guess, all_options]).attach(CORS)
}

// fn main() {
//     let mut a = DMatrix::<u32>::from_vec(3, 1, vec![0,1,2]);
//     let mut b = DMatrix::<u32>::from_vec(3, 1, vec![2,1,0]);
//
//     let mut c = a + b;
//     let d = -5;
//
//     println!("{}", d)
// }

pub struct CORS;

#[rocket::async_trait]
impl Fairing for CORS {
    fn info(&self) -> Info {
        Info {
            name: "Add CORS headers to responses",
            kind: Kind::Response,
        }
    }

    async fn on_response<'r>(&self, _request: &'r Request<'_>, response: &mut Response<'r>) {
        response.set_header(Header::new("Access-Control-Allow-Origin", "*"));
        response.set_header(Header::new("Access-Control-Allow-Methods", "POST, GET, PATCH, OPTIONS"));
        response.set_header(Header::new("Access-Control-Allow-Headers", "*"));
        response.set_header(Header::new("Access-Control-Allow-Credentials", "true"));
    }
}


// fn main() {
//     let mut training_data = emnist_parser::read_emnist("C:/Users/logge/RustroverProjects/neural_network/resources/emnist_digits/train/images", "C:/Users/logge/RustroverProjects/neural_network/resources/emnist_digits/train/labels");
//
//     let iterations = 60;
//     let tolerance = 0.95;
//     let layer_specs = &[784, 30, 10];
//     let lr = 0.001;
//
//     let mut network = Network::new(layer_specs, lr, RELU);
//
//     network.train(&mut training_data, iterations);
//
//     network.profile_save(tolerance, lr, iterations as u32, layer_specs);
//
//     network.save_to_def_path();
// }