use std::sync::{Arc, Mutex};

// use eframe::egui::{self};
// use egui_plot::{Legend, Line, PlotPoints};
use plotters::prelude::*;

pub struct FilePlotter {
    points: Vec<(f32, f32)>,
    caption: String,
    epochs: u32,
}

impl FilePlotter {
    pub fn new(caption: String, epochs: u32) -> FilePlotter {
        FilePlotter {
            points: vec![],
            caption,
            epochs,
        }
    }

    pub fn plot(&mut self, point: (f32, f32), buffer: &mut [u8]) {
        let root = BitMapBackend::with_buffer(buffer, (640, 480)).into_drawing_area();

        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(35)
            .y_label_area_size(40)
            .margin(5)
            .caption(&self.caption, ("sans-serif", 50.0))
            .build_cartesian_2d(0f32..self.epochs as f32, 0f32..100f32).unwrap();

        chart
            .configure_mesh()
            .disable_mesh()
            .bold_line_style(WHITE.mix(0.3))
            .y_desc("Accuracy")
            .x_desc("Epoch")
            .x_labels(10)
            .y_labels(10)
            .axis_desc_style(("sans-serif", 15))
            .draw().unwrap();

        self.points.push(point);

        chart
            .draw_series(LineSeries::new(
                self.points.clone(),
                &RED,
            )).unwrap();

        chart
            .draw_series(PointSeries::of_element(
                self.points.last().map(|(x, y)| (*x, *y)),
                2,
                ShapeStyle::from(&RED).filled(),
                &|coord, size, style| {
                    EmptyElement::at(coord)
                        + Circle::new((0, 0), size, style)
                        + Text::new(format!("{}%", coord.1), (0, 15), ("sans-serif", 15))
                },
            )).unwrap();

        root.present().unwrap();
    }
}


// pub(crate) struct LivePlotter {
//     lock_x: bool,
//     lock_y: bool,
//     ctrl_to_zoom: bool,
//     shift_to_horizontal: bool,
//     zoom_speed: f32,
//     scroll_speed: f32,
//     points: Arc<Mutex<Vec<[f64; 2]>>>,
//     epochs: f64,
// }
//
// impl LivePlotter {
//     pub fn new(points: Arc<Mutex<Vec<[f64; 2]>>>, epochs: f64) -> LivePlotter {
//         LivePlotter {
//             lock_x: false,
//             lock_y: false,
//             ctrl_to_zoom: false,
//             shift_to_horizontal: false,
//             zoom_speed: 1.0,
//             scroll_speed: 1.0,
//             points,
//             epochs,
//         }
//     }
// }
//
// impl Default for LivePlotter {
//     fn default() -> Self {
//         Self {
//             lock_x: false,
//             lock_y: false,
//             ctrl_to_zoom: false,
//             shift_to_horizontal: false,
//             zoom_speed: 1.0,
//             scroll_speed: 1.0,
//             points: Arc::new(Mutex::new(Vec::new())),
//             epochs: 100f64,
//         }
//     }
// }
//
// impl eframe::App for LivePlotter {
//     fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
//         egui::CentralPanel::default().show(ctx, |ui| {
//             egui_plot::Plot::new("plot")
//                 .y_axis_label("Accuracy")
//                 .x_axis_label("Epochs")
//                 .include_x(self.epochs)
//                 .include_y(100)
//                 .allow_zoom(false)
//                 .allow_drag(false)
//                 .allow_scroll(false)
//                 .allow_boxed_zoom(false)
//                 .allow_double_click_reset(false)
//                 .legend(Legend::default())
//                 .show(ui, |plot_ui| {
//                     let acc_points = PlotPoints::new(self.points.lock().unwrap().clone());
//                     plot_ui.line(Line::new(acc_points).name("Accuracy"));
//                 });
//         });
//
//         ctx.request_repaint();
//     }
// }