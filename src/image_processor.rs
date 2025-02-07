use std::cmp::max;
use std::collections::VecDeque;
use std::ops::Add;
use std::ops::Deref;

use image::{ImageBuffer, Rgb};
use image::imageops::FilterType;
use regex::Regex;

use crate::DEBUG;
use crate::debug_only;
use crate::utils::Error;

macro_rules! preprocess {
    ($image:expr, $($function:ident $(($($args:expr),*))?),*) => {{
        let mut image: Image = $image;

        $(
            image = $function(&image $($(, $args)*)?);
            debug_only!({
                let method_name = type_name_of($function);
                let regex = Regex::new(".*::").unwrap();
                let filename = "logs/".to_string().add(&*regex.replace_all(method_name, "").deref().to_string().add(".png"));
                image.save(filename).unwrap();
            });
        )*

        image
    }};
}

// https://stackoverflow.com/a/40234666
fn type_name_of<T>(_: T) -> &'static str {
    std::any::type_name::<T>()
}

type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;

pub struct ImageProcessor {}

impl ImageProcessor {
    pub fn from_rle(input: &String) -> Result<Vec<f32>, Error> {
        let mut vec: Vec<u8> = Vec::new();

        let chunks = input.split(";");

        for chunk in chunks {
            if chunk.len() == 0 { continue; }
            let num = (&chunk[0..(chunk.len() - 1)]).parse::<usize>()?;
            let val = (&chunk[(chunk.len() - 1)..]).parse::<u8>()?;

            vec.append(&mut vec![val; num]);
        }

        let width = f32::sqrt(vec.len() as f32) as u32;
        let mut image = ImageBuffer::new(width, width);

        for (index, val) in vec.iter().enumerate() {
            let x = index as f32 % width as f32;
            let y = index as f32 / width as f32;

            image.put_pixel(x.floor() as u32, y.floor() as u32, Rgb([*val * 255, *val * 255, *val * 255]));
        }

        debug_only!(image.save("logs/original_image.png")?);

        let clusters = get_clusters(&image, 100);
        let max_cluster = get_max_cluster(clusters);

        image = preprocess!(
            image,
            denoise(&max_cluster),
            gaussian_blur(1f32),
            crop(&max_cluster),
            resize(24, 24, FilterType::Gaussian),
            center
        );

        Ok(image.pixels().map(|pix| pix.0[0] as f32 / 255f32).collect())
    }
}

fn resize(image: &Image, width: u32, height: u32, filter_type: FilterType) -> Image {
    image::imageops::resize(image, width, height, filter_type)
}

fn center(image: &Image) -> Image {
    let mut centered_image: Image = Image::new(image.width() + 4, image.height() + 4);

    for x in 0..image.width() {
        for y in 0..image.height() {
            centered_image.get_pixel_mut(x + 2, y + 2).0 = image.get_pixel(x, y).0
        }
    }

    centered_image
}

fn fix_rotation(image: &Image) -> Image {
    let mut fixed_rotation_image = image.clone();

    fixed_rotation_image = image::imageops::rotate90(&fixed_rotation_image);
    fixed_rotation_image = image::imageops::flip_horizontal(&fixed_rotation_image);

    fixed_rotation_image
}

fn gaussian_blur(image: &Image, sigma: f32) -> Image {
    image::imageops::blur(image, sigma)
}

fn get_clusters(image: &Image, threshold: u8) -> Vec<Vec<u32>> {
    let mut visited = vec![false; (image.width() * image.height()) as usize];

    let mut clusters = Vec::new();

    for (index, _) in image.pixels().enumerate() {
        if visited[index] {
            continue;
        }

        let mut queue = VecDeque::new();
        queue.push_front(index as u32);

        let mut cluster = Vec::new();

        while !queue.is_empty() {
            let index = queue.pop_front().unwrap();

            if visited[index as usize] {
                continue;
            }

            let x = index % image.width();
            let y = index / image.width();

            for row in -1..=1_isize {
                for col in -1..=1_isize {
                    let n_x = max(0, x as isize + row) as u32;
                    let n_y = max(0, y as isize + col) as u32;

                    if n_x >= image.width() || n_y >= image.height() {
                        continue;
                    }

                    let n_index = n_x + n_y * image.width();

                    if image.get_pixel(n_x, n_y).0[1] > threshold && !visited[n_index as usize] {
                        cluster.push(n_index);
                        queue.push_front(n_index)
                    }
                }
            }

            visited[index as usize] = true;
        }

        if !cluster.is_empty() {
            clusters.push(cluster.clone());
        }
    }

    clusters
}

fn get_max_cluster(clusters: Vec<Vec<u32>>) -> Vec<u32> {
    clusters
        .iter()
        .max_by(|vec1, vec2| vec1.len().cmp(&vec2.len()))
        .unwrap()
        .clone()
}

fn denoise(image: &Image, max_cluster: &Vec<u32>) -> Image {
    let mut denoised_image: Image = ImageBuffer::new(image.width(), image.height());

    for index in max_cluster {
        let x = index % denoised_image.width();
        let y = index / denoised_image.width();

        denoised_image.get_pixel_mut(x, y).0 = image.get_pixel(x, y).0;
    }

    denoised_image
}

fn crop(image: &Image, max_cluster: &Vec<u32>) -> Image {
    if max_cluster.is_empty() {
        return image.clone();
    }

    let xs: Vec<u32> = max_cluster
        .iter()
        .map(|index| index % image.width())
        .collect();
    let ys: Vec<u32> = max_cluster
        .iter()
        .map(|index| index / image.width())
        .collect();

    let left_bound = *xs.iter().min().unwrap();
    let right_bound = *xs.iter().max().unwrap();
    let top_bound = *ys.iter().min().unwrap();
    let bottom_bound = *ys.iter().max().unwrap();

    let crop_w = right_bound - left_bound;
    let crop_h = bottom_bound - top_bound;

    let max_wh = max(crop_h, crop_w);

    let mut cropped_image: Image = Image::new(max_wh, max_wh);

    let adj_x = (max_wh - crop_w) / 2;
    let adj_y = (max_wh - crop_h) / 2;

    for x in 0..crop_w {
        for y in 0..crop_h {
            cropped_image.get_pixel_mut(x + adj_x, y + adj_y).0 =
                image.get_pixel(x + left_bound, y + top_bound).0;
        }
    }

    cropped_image
}
