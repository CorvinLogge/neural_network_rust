use std::collections::VecDeque;
use std::fs;

use image::{ImageBuffer, Rgb};

#[derive(Clone)]
pub(crate) struct DataPoint {
    pub(crate) input: Vec<f32>,
    pub(crate) target: Vec<f32>,
}

pub fn read_emnist(inputs_path: String, targets_path: String) -> Vec<DataPoint> {
    let mut inputs_as_bytes: VecDeque<u8> = VecDeque::from(fs::read(inputs_path).unwrap());
    let mut targets_as_bytes: VecDeque<u8> = VecDeque::from(fs::read(targets_path).unwrap());

    inputs_as_bytes.drain(0..16);
    targets_as_bytes.drain(0..8);

    let mut data_points: Vec<DataPoint> = Vec::new();

    while !targets_as_bytes.is_empty() {
        let target_val = targets_as_bytes.pop_front().unwrap() as f32;
        let mut target_arr = vec![0.0; 10];
        let _ = std::mem::replace(&mut target_arr[target_val as usize], 1_f32);
        data_points.push(
            DataPoint {
                input: inputs_as_bytes.drain(..784).map(|x| x as f32 / 255.0).collect(),
                target: target_arr,
            }
        );
    }

    return data_points.clone();
}

fn generate_image(data_point: DataPoint) {
    let mut image = ImageBuffer::new(28, 28);

    for (index, pix) in data_point.input.iter().enumerate() {
        let x = index as f32 % 28.0;
        let y = index as f32 / 28.0;

        image.put_pixel(x.floor() as u32, y.floor() as u32, Rgb([(*pix * 255.0) as u8, (*pix * 255.0) as u8, (*pix * 255.0) as u8]));
    }

    println!("{:?}", data_point.target);
    image = image::imageops::rotate90(&image);
    image = image::imageops::flip_horizontal(&image);

    image.save("test.png").unwrap();
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

        XOR {
            data_points
        }
    }
}