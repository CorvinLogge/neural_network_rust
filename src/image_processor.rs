use image::{ImageBuffer, Rgb};
use image::imageops::FilterType;
use rocket::yansi::Paint;

pub struct ImageProcessor {}

impl ImageProcessor {
    pub fn from_vec(vec: Vec<usize>) -> Vec<f32> {
        let width = f32::sqrt(vec.len() as f32) as u32;
        let mut image = ImageBuffer::new(width, width);

        for (index, val) in vec.iter().enumerate() {
            let x = index as f32 % width as f32;
            let y = index as f32 / width as f32;

            image.put_pixel(x.floor() as u32, y.floor() as u32, Rgb([(*val) as u8, (*val) as u8, (*val) as u8]));
        }

        image.save("original_image.png").unwrap();

        image = image::imageops::resize(&image, 28, 28, FilterType::Lanczos3);

        image.save("resized_image.png").unwrap();

        let mut sum: u32 = 0;
        let mut w_sum: u32 = 0;

        for (index, pix) in image.rows().enumerate() {
            sum += pix.clone().map(|p| { p.0[1] as u32 }).sum::<u32>();
            w_sum += pix.map(|p| { p.0[1] as u32 }).sum::<u32>() * (index as u32 + 1);
        }

        let middle_y = w_sum / sum;

        image = image::imageops::rotate90(&image);

        sum = 0;
        w_sum = 0;

        for (index, pix) in image.rows().enumerate() {
            sum += pix.clone().map(|p| { p.0[1] as u32 }).sum::<u32>();
            w_sum += pix.map(|p| { p.0[1] as u32 }).sum::<u32>() * (index as u32 + 1);
        }

        let middle_x = w_sum / sum;

        image = image::imageops::rotate270(&image);

        let true_middle = 28 / 2;

        let transform_x: i32 = true_middle - middle_x as i32;
        let transform_y: i32 = true_middle - middle_y as i32;

        let mut centered_image = ImageBuffer::new(28, 28);

        for (index, pix) in image.clone().pixels().enumerate() {
            let x = (index as f32 % 28f32) as i32;
            let y = (index as f32 / 28f32) as i32;

            let new_x = x + transform_x;
            let new_y = y + transform_y;

            if (new_x as u32) < 28u32 && (new_y as u32) < 28u32 {
                centered_image.put_pixel(new_x as u32, new_y as u32, pix.clone());
            }
        }

        centered_image.save("centered_image.png").unwrap();

        centered_image = image::imageops::rotate270(&centered_image);
        centered_image = image::imageops::flip_horizontal(&centered_image);

        centered_image.pixels().map(|pix| { pix.0[0] as f32 / 255f32 }).collect()
    }
}