use std::collections::{HashMap, VecDeque};
use std::fmt::{Debug};
use std::fs;
use std::ops::Range;

use image::{ImageBuffer, Rgb};
use num_integer::Roots;
use rocket::form::validate::Len;
use rocket::http::Status;
use serde::{Deserialize, Serialize};
use crate::data_sets::DataSet::*;
use crate::utils::Error;

#[derive(Clone)]
pub(crate) struct DataPoint {
    pub(crate) input: Vec<f32>,
    pub(crate) target: Vec<f32>,
}

#[derive(Serialize, Deserialize, FromFormField, Debug, Copy, Clone)]
#[serde(crate = "rocket::serde")]
pub enum DataSet {
    #[serde(alias = "mnist_digits")] MnistDigits = 0,
    #[serde(alias = "emnist_digits")] EMnistDigits = 1,
    #[serde(alias = "emnist_letters")] EMnistLetters = 2,
    #[serde(alias = "emnist_byclass")] EMnistByClass = 3,
    #[serde(alias = "emnist_balanced")] EMnistBalanced = 4,
}

impl DataSet {
    pub fn path(&self) -> String {
        match self {
            MnistDigits => "./resources/mnist/".to_string(),
            EMnistDigits => "./resources/emnist_digits/".to_string(),
            EMnistLetters => "./resources/emnist_letters/".to_string(),
            EMnistByClass => "./resources/emnist_byclass/".to_string(),
            EMnistBalanced => "./resources/emnist_balanced/".to_string(),
        }
    }

    pub fn num_classes(&self) -> usize {
        match self {
            MnistDigits => 10,
            EMnistDigits => 10,
            EMnistLetters => 37,
            EMnistByClass => 62,
            EMnistBalanced => 47,
        }
    }

    pub fn num_inputs(&self) -> usize {
        match self {
            _ => 784
        }
    }

    pub fn from_u8(u: u8) -> Result<DataSet, Error> {
        match u {
            0 => { Ok(MnistDigits) }
            1 => { Ok(EMnistDigits) }
            2 => { Ok(EMnistLetters) }
            3 => { Ok(EMnistByClass) }
            4 => { Ok(EMnistBalanced) }
            _ => { Err(Error::new(format!("unknown variant '{}', expected one of '0', '1', '2', '3', '4'", u), Status::BadRequest)) }
        }
    }
}

fn read_emnist(inputs_path: String, targets_path: String, num_classes: usize) -> Vec<DataPoint> {
    let mut inputs_as_bytes: VecDeque<u8> = VecDeque::from(fs::read(inputs_path).unwrap());
    let mut targets_as_bytes: VecDeque<u8> = VecDeque::from(fs::read(targets_path).unwrap());

    inputs_as_bytes.drain(0..16);
    targets_as_bytes.drain(0..8);

    let mut data_points: Vec<DataPoint> = Vec::with_capacity(targets_as_bytes.len());

    while !targets_as_bytes.is_empty() {
        let target_val = targets_as_bytes.pop_front().unwrap() as f32;
        let mut target_vec = vec![0.0; num_classes];
        let _ = std::mem::replace(&mut target_vec[target_val as usize], 1_f32);
        data_points.push(DataPoint {
            input: mirror(&inputs_as_bytes.drain(..784).collect())
                .iter()
                .map(|x| *x as f32 / 255.0)
                .collect(),
            target: target_vec,
        });
    }

    data_points.clone()
}

//Mirrors a quadratic vector along the diagonal from top left to bottom right
fn mirror(vec: &Vec<u8>) -> Vec<u8> {
    let mut mirrored = vec![0; vec.len()];

    let w = vec.len().sqrt();

    for index in 0..vec.len() {
        let x = (index as f32 / w as f32).floor();
        let y = (index as f32 % w as f32).floor();

        let n_index = x + y * 28.0;

        *mirrored.get_mut(n_index as usize).unwrap() = *vec.get(index).unwrap();
    }

    mirrored
}

pub fn read_emnist_test(data_path: &String, num_classes: usize) -> Vec<DataPoint> {
    read_emnist(
        format!("{data_path}test/images"),
        format!("{data_path}test/labels"),
        num_classes,
    )
}

pub fn read_emnist_train(data_path: &String, num_classes: usize) -> Vec<DataPoint> {
    read_emnist(
        format!("{data_path}train/images"),
        format!("{data_path}train/labels"),
        num_classes,
    )
}

pub fn generate_images(data_set: DataSet, indices: Range<usize>) -> Result<(), serde_json::Error> {
    let data_path = data_set.path();
    let num_classes = data_set.num_classes();

    let data_points = read_emnist_train(&data_path, num_classes);

    for index in indices {
        generate_image(data_points.get(index).unwrap().clone(), index);
    }

    Ok(())
}

fn generate_image(data_point: DataPoint, index: usize) {
    let mut image = ImageBuffer::new(28, 28);

    for (index, pix) in data_point.input.iter().enumerate() {
        let x = index as f32 % 28.0;
        let y = index as f32 / 28.0;

        image.put_pixel(
            x.floor() as u32,
            y.floor() as u32,
            Rgb([
                (*pix * 255.0) as u8,
                (*pix * 255.0) as u8,
                (*pix * 255.0) as u8,
            ]),
        );
    }

    println!("{:?}", data_point.target);
    image = image::imageops::rotate90(&image);
    image = image::imageops::flip_horizontal(&image);

    image.save(format!("test_index_{index}.png")).unwrap();
}

pub struct XOR {
    pub(crate) data_points: Vec<DataPoint>,
}

impl XOR {
    pub fn get() -> XOR {
        let mut data_points: Vec<DataPoint> = Vec::new();
        data_points.push(DataPoint {
            input: vec![0.0, 1.0],
            target: vec![1.0],
        });
        data_points.push(DataPoint {
            input: vec![1.0, 0.0],
            target: vec![1.0],
        });
        data_points.push(DataPoint {
            input: vec![0.0, 0.0],
            target: vec![0.0],
        });
        data_points.push(DataPoint {
            input: vec![1.0, 1.0],
            target: vec![0.0],
        });

        XOR { data_points }
    }
}
