#![allow(unused)]

#[macro_use]
extern crate rocket;

use std::string::ToString;

use image::{ImageBuffer, Rgb};
use rocket::fairing::{Fairing, Info, Kind};
use rocket::http::{Header, Status};
use rocket::serde::json::Json;
use rocket::{Build, Data, Request, Response, Rocket};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::data_sets::{read_emnist_test, DataSet};
use crate::function::ActivationFunction::SIGMOID;
use crate::function::ErrorFunction;
use crate::image_processor::ImageProcessor;
use crate::layer::Layer;
use crate::network::{ActivationMode, Network, ProfileResult};
use crate::optimizers::Optimizer;
use crate::plotter::file_plot;
use crate::utils::Error;

mod data_sets;
mod function;
mod image_processor;
mod layer;
mod network;
mod optimizers;
mod plotter;
mod utils;

static mut DEBUG: bool = false;
static NET_ID_PATTERN: &str = "%Y%m%d%H%M%S";

#[derive(Deserialize)]
struct TrainBatchReq {
    batch_size: usize,
    epochs: usize,
    network: Network,
}

#[derive(Deserialize)]
struct ProfileReq {
    network_id: String,
    tolerance: f32,
    data_set: DataSet,
}

#[derive(Deserialize, Serialize, Debug)]
struct GuessReq {
    network_id: String,
    image: String,
}

#[post("/train/batches", data = "<request>", format = "application/json")]
fn train_batches(mut request: Json<TrainBatchReq>) -> Result<Json<Value>, Error> {
    let mut network = request.network.clone();

    network.init();

    network.train_batches(request.epochs, request.batch_size)?;

    let id = network.save_to_def_path()?;

    Ok(Json::from(json!(
        {
            "id": id
        }
    )))
}

#[post("/train/batches/plot", data = "<request>", format = "application/json")]
fn train_batches_plot(request: Json<TrainBatchReq>) -> Result<Json<Value>, Error> {
    let mut network = request.network.clone();

    network.init();

    let id = network.save_to_def_path()?;

    let plot = network.train_batches_plot(request.epochs, request.batch_size)?;

    let plot_w = 640;
    let plot_h = 480;

    let mut plot_buffer: Vec<u8> = vec![0u8; plot_w * plot_h * 3];
    file_plot(plot, &mut plot_buffer, id.clone(), request.epochs);

    let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_vec(plot_w as u32, plot_h as u32, plot_buffer).ok_or(Error::new(
            "Failed to save plot".to_string(),
            Status::InternalServerError,
        ))?;

    image.save(format!("./resources/models/{id}/plot.png"))?;

    Ok(Json::from(json!(
        {
          "id": id
        }
    )))
}

#[post("/profile", data = "<request>", format = "application/json")]
fn profile(request: Json<ProfileReq>) -> Result<Json<ProfileResult>, Error> {
    let data_path = request.data_set.path();
    let num_classes = request.data_set.num_classes();
    let testing_data = read_emnist_test(&data_path, num_classes);

    let network_id = &request.network_id;
    let network = Network::from_file(&format!("./resources/models/{network_id}"))?;
    let profile = network.profile(request.tolerance, &testing_data)?;

    Ok(Json::from(profile))
}

#[post("/guess", data = "<request>")]
fn guess(request: Json<GuessReq>) -> Result<Json<Vec<f32>>, Error> {
    let mut network = Network::from_file(&request.network_id)?;
    let guess = network.guess(&ImageProcessor::from_rle(&request.image)?);
    let data = guess.data.as_vec();
    Ok(Json::from(data.clone()))
}

#[get("/images?<data_set>")]
fn generate_images_(data_set: DataSet) -> Result<(), Error> {
    data_sets::generate_images(data_set, 20..30)?;

    Ok(())
}

#[get("/test")]
fn test() -> &'static str {
    "Connection worked\n"
}

#[options("/<_..>")]
fn all_options() {
    /* Intentionally left empty */
}

#[launch]
fn rocket() -> Rocket<Build> {
    let mut debug = false;

    #[cfg(feature = "local")]
    {
        unsafe {
            DEBUG = true;
        }
        debug = true;
    }

    let rocket = rocket::build();

    let routes = match debug {
        true => {
            routes![
                guess,
                all_options,
                profile,
                train_batches,
                train_batches_plot,
                test,
                generate_images_
            ]
        }
        false => routes![guess, all_options, test],
    };

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
