#[macro_use]
extern crate rocket;

use conv::ConvUtil;
use rocket::form::Form;

use crate::function::ActivationFunction::RELU;
use crate::function::Function;
use crate::network::Network;

mod layer;
mod network;
mod emnist_parser;
mod data_point;
mod function;

// #[get("/world")]
// fn world() -> &'static str {
//     "Hello, world!"
// }

#[derive(FromForm)]
struct TestForm {
    layer_specs: Vec<usize>,
}

#[get("/train?<iterations>&<lr>&<layerspecs>")]
fn train(iterations: usize, lr: f32, layerspecs: Form<TestForm>) {
    let mut training_data = emnist_parser::read_emnist("C:/Users/logge/RustroverProjects/neural_network/resources/emnist_digits/train/images", "C:/Users/logge/RustroverProjects/neural_network/resources/emnist_digits/train/labels");

    let mut network = Network::new(layerspecs.layer_specs.as_slice(), lr, RELU);

    network.train(&mut training_data, iterations);

    network.save_to_def_path();
}

#[get("/profile?<network_id>&<tolerance>")]
fn profile(network_id: &str, tolerance: f32) -> String {
    Network::from_file(format!("C:/Users/logge/RustroverProjects/neural_network/resources/models/{network_id}").as_str()).profile_str(tolerance)
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/network", routes![train, profile])
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