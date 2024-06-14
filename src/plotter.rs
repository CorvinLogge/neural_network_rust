use plotters::prelude::*;

pub fn file_plot(points: Vec<(f32, f32)>, buffer: &mut [u8], caption: String, epochs: usize) {
    let root = BitMapBackend::with_buffer(buffer, (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption(&caption, ("sans-serif", 50.0))
        .build_cartesian_2d(0f32..epochs as f32, 0f32..100f32)
        .unwrap();

    chart
        .configure_mesh()
        .disable_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Accuracy")
        .x_desc("Epoch")
        .x_labels(10)
        .y_labels(10)
        .axis_desc_style(("sans-serif", 15))
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(points.clone(), &RED))
        .unwrap();

    chart
        .draw_series(PointSeries::of_element(
            points.last().map(|(x, y)| (*x, *y)),
            2,
            ShapeStyle::from(&RED).filled(),
            &|coord, size, style| {
                EmptyElement::at(coord)
                    + Circle::new((0, 0), size, style)
                    + Text::new(format!("{}%", coord.1), (-10, 15), ("sans-serif", 15))
            },
        ))
        .unwrap();

    root.present().unwrap();
}