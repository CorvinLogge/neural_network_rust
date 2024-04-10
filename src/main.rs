#[macro_use]
extern crate rocket;

use rocket::{Request, Response};
use rocket::fairing::{Fairing, Info, Kind};
use rocket::form::FromForm;
use rocket::http::Header;
use rocket::response::status::Accepted;
use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};

use crate::function::ActivationFunction::RELU;
use crate::function::Function;
use crate::image_processor::ImageProcessor;
use crate::network::Network;

mod layer;
mod network;
mod emnist_parser;
mod data_point;
mod function;
mod image_processor;

#[derive(FromForm)]
struct TrainReq {
    iterations: usize,
    lr: f32,
    layer_specs: Vec<usize>,
}

#[derive(Deserialize, Serialize)]
#[serde(crate = "rocket::serde")]
#[derive(Debug)]
struct GuessReq {
    network_id: String,
    image: Vec<usize>,
}

#[get("/train?<r>")]
fn train(r: TrainReq) -> Accepted<String> {
    let mut training_data = emnist_parser::read_emnist("./resources/emnist_digits/train/images", "./resources/emnist_digits/train/labels");

    let mut network = Network::new(r.layer_specs.as_slice(), r.lr, RELU);

    network.train(&mut training_data, r.iterations);

    let id = network.save_to_def_path();

    Accepted(format!("Successfully trained model: {id}"))
}

#[get("/profile?<network_id>&<tolerance>")]
fn profile(network_id: &str, tolerance: f32) -> String {
    Network::from_file(format!("./resources/models/{network_id}").as_str()).profile_str(tolerance)
}

#[post("/guess", data = "<r>")]
fn guess(r: Json<GuessReq>) -> Json<Vec<f32>> {

    // Preprocess incoming image

    let binding = Network::from_file(format!("./resources/models/{}", r.network_id).as_str()).feedforward(&ImageProcessor::from_vec(r.image.clone()));
    let data = binding.data.as_vec();
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
    rocket::build().mount("/network", routes![train, profile, test, guess, all_options]).attach(CORS)
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