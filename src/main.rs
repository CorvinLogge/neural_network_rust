#![allow(unused)]
#![allow(non_camel_case_types)]

#[macro_use]
extern crate rocket;

use std::string::ToString;

use image::{ImageBuffer, Rgb};
use rocket::{Build, Request, Response, Rocket};
use rocket::fairing::{Fairing, Info, Kind};
use rocket::http::{Header, Status};
use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::data_sets::{DataSet, read_emnist_test};
use crate::function::ActivationFunction::{SIGMOID};
use crate::function::ErrorFunction;
use crate::image_processor::ImageProcessor;
use crate::layer::LayerType;
use crate::network::{ActivationMode, Network, ProfileResult};
use crate::optimizers::Optimizer;
use crate::plotter::file_plot;
use crate::utils::Error;

mod utils;
mod data_sets;
mod function;
mod image_processor;
mod network;
mod plotter;
mod optimizers;
mod layer;

static mut DEBUG: bool = false;
static NET_ID_PATTERN: &str = "%Y%m%d%H%M%S";


#[derive(Deserialize)]
struct TrainBatchReq {
    batch_size: usize,
    epochs: usize,
    lr: f32,
    layer_specs: Vec<usize>,
    layer_types: Vec<LayerType>,
    activation_mode: ActivationMode,
    optimizer: Optimizer,
    data: DataSet,
    error_function: ErrorFunction,
}

#[derive(Deserialize, Serialize)]
#[derive(Debug)]
struct GuessReq {
    network_id: String,
    image: String,
}

#[post("/train/batches", data = "<request>", format = "application/json")]
fn train_batches(request: Json<TrainBatchReq>) -> Result<Json<Value>, Error> {
    let mut network = Network::with_mode(request.layer_specs.as_slice(), request.lr, request.error_function, request.activation_mode, request.layer_types.clone(), request.optimizer)?;

    network.train_batches(request.data, request.epochs, request.batch_size)?;

    let id = network.save_to_def_path()?;

    Ok(Json::from(json!(
        {
            "id": id
        }
    )))
}

#[post("/train/batches/plot", data = "<request>", format = "application/json")]
fn train_batches_plot(request: Json<TrainBatchReq>) -> Result<Json<Value>, Error> {
    let mut network = Network::with_mode(request.layer_specs.as_slice(), request.lr, request.error_function, request.activation_mode, request.layer_types.clone(), request.optimizer)?;

    let plot = network.train_batches_plot(request.data, request.epochs, request.batch_size)?;

    debug_only!(
        for (i, layer) in network.layers.iter().enumerate() {
            println!("weights layer {i}: {}", layer.weights());
            println!("biases layer {i}: {}", layer.biases());
    });

    let id = network.save_to_def_path()?;

    let plot_w = 640;
    let plot_h = 480;

    let mut plot_buffer: Vec<u8> = vec![0u8; plot_w * plot_h * 3];
    file_plot(plot, &mut plot_buffer, id.clone(), request.epochs);

    let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_vec(plot_w as u32, plot_h as u32, plot_buffer).ok_or(Error::new(
            "Failed to save plot".to_string(),
            Status::new(500),
        ))?;
    image.save(format!("./resources/models/{id}/plot.png"))?;

    Ok(Json::from(json!(
        {
          "id": id
        }
    )))
}

#[get("/profile?<network_id>&<tolerance>&<data_set>")]
fn profile(network_id: &str, tolerance: f32, data_set: DataSet) -> Result<Json<ProfileResult>, Error> {
    let data_path = data_set.path();
    let num_classes = data_set.num_classes();
    let testing_data = read_emnist_test(&data_path, num_classes);

    let network = Network::from_file(format!("./resources/models/{network_id}"))?;
    let profile = network.profile(tolerance, &testing_data)?;

    Ok(Json::from(profile))
}

#[get("/profile/digit?<network_id>&<tolerance>&<data_set>&<digit>")]
fn profile_digit(
    network_id: &str,
    tolerance: f32,
    data_set: DataSet,
    digit: usize,
) -> Result<String, Error> {
    Ok(format!(
        "{:?}",
        Network::from_file(format!("./resources/models/{network_id}"))?
            .profile_digit(tolerance, data_set, digit)?
    ))
}

#[post("/guess", data = "<request>")]
fn guess(request: Json<GuessReq>) -> Result<Json<Vec<f32>>, Error> {
    let mut network = Network::from_file(format!("./resources/models/{}", request.network_id))?;
    let guess = network.guess(&ImageProcessor::from_data_url(&*request.image));
    let data = guess.data.as_vec();
    Ok(Json::from(data.clone()))
}

#[get("/images?<data_set>")]
fn generate_images_(data_set: DataSet) -> Result<(), Error> {
    data_sets::generate_images(data_set, 20..30)?;

    Ok(())
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
fn rocket() -> Rocket<Build> {
    let rocket = rocket::build();
    let figment = rocket.figment();
    let profile = figment.profile();

    let mut routes;
    let mut debug = false;

    match profile.to_string().as_str() {
        "debug" | "local" => unsafe {
            DEBUG = true;
            debug = true;
        },
        _ => {}
    }

    match debug {
        true => {
            routes = routes![
                guess,
                all_options,
                profile,
                profile_digit,
                train_batches,
                train_batches_plot,
                test,
                generate_images_
            ]
        }
        false => routes = routes![guess, all_options, test],
    }

    rocket.mount("/network", routes).attach(CORS)
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
        response.set_header(Header::new(
            "Access-Control-Allow-Methods",
            "POST, GET, OPTIONS",
        ));
        response.set_header(Header::new("Access-Control-Allow-Headers", "*"));
        response.set_header(Header::new("Access-Control-Allow-Credentials", "true"));
    }
}
