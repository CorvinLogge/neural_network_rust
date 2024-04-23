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

use crate::function::ActivationFunction::{RELU, SIGMOID};
use crate::function::ErrorFunction::{CrossEntropy, MSE};
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
    epochs: usize,
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
        "relu_sig" => network = Network::relu_sig(r.layer_specs.as_slice(), r.lr, MSE),
        "relu" => network = Network::new(r.layer_specs.as_slice(), r.lr, RELU, MSE),
        "sigmoid" => network = Network::new(r.layer_specs.as_slice(), r.lr, SIGMOID, MSE),
        _ => network = Network::relu_sig(r.layer_specs.as_slice(), r.lr, MSE),
    }

    network.train_batches(data, r.epochs, r.batch_size);

    let id = network.save_to_def_path();

    Accepted(format!("Successfully trained model: {id}"))
}

#[get("/train/batches/plot?<r>")]
fn train_batches_plot(r: TrainBatchReq) -> Result<String, String> {
    let mut network;
    let mut data;

    match r.data.as_str() {
        "mnist_digits" => data = MNIST_PATH.to_owned(),
        "e_mnist_digits" => data = EMNIST_PATH.to_owned(),
        _ => return Err(format!("'{}' is not a valid option for data", r.data)),
    }

    match r.activation_mode.as_str() {
        "relu_sig" => network = Network::relu_sig(r.layer_specs.as_slice(), r.lr, MSE),
        "relu" => network = Network::new(r.layer_specs.as_slice(), r.lr, RELU, MSE),
        "sigmoid" => network = Network::new(r.layer_specs.as_slice(), r.lr, SIGMOID, MSE),
        _ => return Err(format!("'{}' is not a valid option for activation_mode", r.activation_mode)),
    }

    let plot = network.train_batches_plot(data.clone(), r.epochs, r.batch_size);

    println!("{}", network.profile_str(0.95, data.clone()));

    let id = network.save_to_def_path();

    let image: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_vec(640, 480, plot).ok_or("Failed to save plot")?;
    image.save(format!("./resources/models/{id}/plot.png")).unwrap();

    Ok(format!("Successfully trained model: {id}"))
}

#[get("/profile?<network_id>&<tolerance>&<data>")]
fn profile(network_id: &str, tolerance: f32, data: String) -> Result<String, String> {
    let mut data_path;

    match data.as_str() {
        "mnist_digits" => data_path = MNIST_PATH.to_owned(),
        "e_mnist_digits" => data_path = EMNIST_PATH.to_owned(),
        _ => return Err(format!("'{}' is not a valid option for data", data)),
    }

    Ok(Network::relu_sig_from_file(format!("./resources/models/{network_id}")).profile_str(tolerance, data_path))
}

#[get("/profile/digit?<network_id>&<tolerance>&<data>&<digit>")]
fn profile_digit(network_id: &str, tolerance: f32, data: String, digit: usize) -> Result<String, String> {
    let mut data_path;

    match data.as_str() {
        "mnist_digits" => data_path = MNIST_PATH.to_owned(),
        "e_mnist_digits" => data_path = EMNIST_PATH.to_owned(),
        _ => return Err(format!("'{}' is not a valid option for data", data)),
    }

    Ok(format!("{:?}", Network::relu_sig_from_file(format!("./resources/models/{network_id}")).profile_number(tolerance, data_path, digit)))
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
    rocket::build().mount("/network", routes![train_batches, train_batches_plot, profile, profile_digit, test, guess, all_options]).attach(CORS)
}

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