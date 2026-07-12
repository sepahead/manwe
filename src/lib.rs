// pub mod audio;
// pub mod bs1770;
pub mod coco_classes;
pub mod model;
pub mod secure_io;
pub mod stream_url;
// pub mod imagenet;
// pub mod token_output_stream;
// pub mod wav;

use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Result, Tensor};
use image::DynamicImage;

const MAX_SOURCE_DIMENSION: usize = 32_768;
const MAX_SOURCE_PIXELS: usize = 64 * 1024 * 1024;
const MAX_INFERENCE_DIMENSION: usize = 4_096;
const COCO_CLASS_COUNT: usize = 80;
const MAX_REPORT_PREDICTIONS: usize = 100_000;
const MAX_REPORT_CANDIDATES: usize = 2_000;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// Computes the isotropically scaled content size for a square inference canvas.
///
/// The returned dimensions are not stride-aligned: the surrounding square canvas
/// is aligned instead. This avoids stretching narrow images to one full stride.
pub fn resize_dimensions(
    original_width: usize,
    original_height: usize,
    longest: usize,
    multiple: usize,
) -> Result<(usize, usize)> {
    validate_source_dimensions(original_width, original_height)?;
    if multiple == 0 || longest < multiple || !longest.is_multiple_of(multiple) {
        candle::bail!("longest must be a non-zero multiple of the alignment")
    }
    if longest > MAX_INFERENCE_DIMENSION {
        candle::bail!("longest must not exceed {MAX_INFERENCE_DIMENSION}")
    }

    let scaled = |short: usize, long: usize| -> Result<usize> {
        let numerator = (short as u128) * (longest as u128);
        let denominator = long as u128;
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;
        let distance_to_next = denominator - remainder;
        let rounded = if remainder < distance_to_next
            || (remainder == distance_to_next && quotient.is_multiple_of(2))
        {
            quotient
        } else {
            quotient + 1
        };
        usize::try_from(rounded.max(1))
            .map_err(|_| candle::Error::Msg("scaled image dimension overflowed".to_string()))
    };
    if original_width >= original_height {
        Ok((longest, scaled(original_height, original_width)?))
    } else {
        Ok((scaled(original_width, original_height)?, longest))
    }
}

fn validate_source_dimensions(width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 {
        candle::bail!("image dimensions must be non-zero")
    }
    if width > MAX_SOURCE_DIMENSION || height > MAX_SOURCE_DIMENSION {
        candle::bail!("image dimensions must not exceed {MAX_SOURCE_DIMENSION}")
    }
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| candle::Error::Msg("image pixel count overflowed".to_string()))?;
    if pixels > MAX_SOURCE_PIXELS {
        candle::bail!("image must not exceed {MAX_SOURCE_PIXELS} pixels")
    }
    Ok(())
}

/// Geometry needed to invert square-letterbox preprocessing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageTransform {
    pub original_width: usize,
    pub original_height: usize,
    pub resized_width: usize,
    pub resized_height: usize,
    pub canvas_width: usize,
    pub canvas_height: usize,
    pub pad_left: usize,
    pub pad_top: usize,
}

impl ImageTransform {
    fn validate_for_image(&self, image: &DynamicImage) -> Result<()> {
        if self.original_width == 0
            || self.original_height == 0
            || self.resized_width == 0
            || self.resized_height == 0
            || self.canvas_width == 0
            || self.canvas_height == 0
            || self.canvas_width > MAX_INFERENCE_DIMENSION
            || self.canvas_height > MAX_INFERENCE_DIMENSION
            || self.original_width != image.width() as usize
            || self.original_height != image.height() as usize
            || self.resized_width > self.canvas_width
            || self.resized_height > self.canvas_height
            || self.canvas_width != self.canvas_height
            || self.pad_left != (self.canvas_width - self.resized_width) / 2
            || self.pad_top != (self.canvas_height - self.resized_height) / 2
        {
            candle::bail!("image transform is inconsistent with the source image")
        }
        Ok(())
    }

    fn source_x(&self, value: f32) -> f32 {
        (value - self.pad_left as f32) * self.original_width as f32 / self.resized_width as f32
    }

    fn source_y(&self, value: f32) -> f32 {
        (value - self.pad_top as f32) * self.original_height as f32 / self.resized_height as f32
    }
}

/// Isotropically resizes and letterboxes RGB into a `[1, 3, S, S]` FP32 tensor.
pub fn prepare_image(
    image: &DynamicImage,
    longest: usize,
    multiple: usize,
    device: &Device,
) -> Result<(Tensor, ImageTransform)> {
    let (width, height) = resize_dimensions(
        image.width() as usize,
        image.height() as usize,
        longest,
        multiple,
    )?;
    let resized = image
        .resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgb8();
    let pad_left = (longest - width) / 2;
    let pad_top = (longest - height) / 2;
    let mut canvas =
        image::RgbImage::from_pixel(longest as u32, longest as u32, image::Rgb([114, 114, 114]));
    image::imageops::overlay(&mut canvas, &resized, pad_left as i64, pad_top as i64);
    let data = canvas.into_raw();
    let tensor = Tensor::from_vec(data, (longest, longest, 3), device)?.permute((2, 0, 1))?;
    let tensor = (tensor.unsqueeze(0)?.to_dtype(candle::DType::F32)? * (1.0 / 255.0))?;
    Ok((
        tensor,
        ImageTransform {
            original_width: image.width() as usize,
            original_height: image.height() as usize,
            resized_width: width,
            resized_height: height,
            canvas_width: longest,
            canvas_height: longest,
            pad_left,
            pad_top,
        },
    ))
}

// Keypoints as reported by ChatGPT :)
// Nose
// Left Eye
// Right Eye
// Left Ear
// Right Ear
// Left Shoulder
// Right Shoulder
// Left Elbow
// Right Elbow
// Left Wrist
// Right Wrist
// Left Hip
// Right Hip
// Left Knee
// Right Knee
// Left Ankle
// Right Ankle
pub const KP_CONNECTIONS: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

use candle::IndexOp;
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};

fn validate_probability(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        candle::bail!("{name} must be finite and between 0 and 1")
    }
    Ok(())
}

fn validate_legend_size(value: u32) -> Result<()> {
    if value > 256 {
        candle::bail!("legend size must not exceed 256 pixels")
    }
    Ok(())
}

fn class_name(class_index: usize) -> &'static str {
    crate::coco_classes::NAMES
        .get(class_index)
        .copied()
        .unwrap_or("unknown")
}

/// Controls whether annotation helpers emit per-object details.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReportOutput {
    /// Render annotations without writing detection data to standard output.
    #[default]
    Quiet,
    /// Write each retained detection to standard output while rendering.
    Stdout,
}

/// Renders COCO detections without emitting per-object output.
pub fn report_detect(
    pred: &Tensor,
    img: DynamicImage,
    transform: &ImageTransform,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
) -> Result<DynamicImage> {
    report_detect_with_output(
        pred,
        img,
        transform,
        confidence_threshold,
        nms_threshold,
        legend_size,
        ReportOutput::Quiet,
    )
}

/// Renders COCO detections with an explicit per-object output policy.
pub fn report_detect_with_output(
    pred: &Tensor,
    img: DynamicImage,
    transform: &ImageTransform,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
    output: ReportOutput,
) -> Result<DynamicImage> {
    validate_source_dimensions(img.width() as usize, img.height() as usize)?;
    transform.validate_for_image(&img)?;
    validate_probability("confidence threshold", confidence_threshold)?;
    validate_probability("NMS threshold", nms_threshold)?;
    validate_legend_size(legend_size)?;
    let (pred_size, npreds) = pred.dims2()?;
    if pred_size != 4 + COCO_CLASS_COUNT {
        candle::bail!(
            "COCO detection reporter requires 4 box rows and {COCO_CLASS_COUNT} class rows"
        )
    }
    if npreds > MAX_REPORT_PREDICTIONS {
        candle::bail!("prediction count must not exceed {MAX_REPORT_PREDICTIONS}")
    }
    let pred = pred.to_device(&Device::Cpu)?;
    let nclasses = COCO_CLASS_COUNT;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    let mut candidate_count = 0_usize;
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        if pred.iter().any(|value| !value.is_finite()) {
            continue;
        }
        let Some(confidence) = pred[4..].iter().copied().max_by(f32::total_cmp) else {
            candle::bail!("prediction contains no class scores")
        };
        if confidence > confidence_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                if candidate_count >= MAX_REPORT_CANDIDATES {
                    candle::bail!(
                        "detection candidate count must not exceed {MAX_REPORT_CANDIDATES}"
                    )
                }
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox);
                candidate_count += 1;
            }
        }
    }

    non_maximum_suppression(&mut bboxes, nms_threshold);

    // Annotate the original image and optionally print box information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let mut img = img.to_rgb8();
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = ab_glyph::FontRef::try_from_slice(&font).map_err(candle::Error::wrap)?;
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            if output == ReportOutput::Stdout {
                println!("{}: {:?}", class_name(class_index), b);
            }
            let xmin = transform
                .source_x(b.xmin)
                .clamp(0.0, initial_w.saturating_sub(1) as f32);
            let ymin = transform
                .source_y(b.ymin)
                .clamp(0.0, initial_h.saturating_sub(1) as f32);
            let xmax = transform.source_x(b.xmax).clamp(0.0, initial_w as f32);
            let ymax = transform.source_y(b.ymax).clamp(0.0, initial_h as f32);
            if xmax <= xmin || ymax <= ymin {
                continue;
            }
            let xmin = xmin as i32;
            let ymin = ymin as i32;
            let dx = ((xmax - xmin as f32).ceil() as u32).max(1);
            let dy = ((ymax - ymin as f32).ceil() as u32).max(1);
            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(xmin, ymin).of_size(dx, dy),
                image::Rgb([255, 0, 0]),
            );
            if legend_size > 0 {
                imageproc::drawing::draw_filled_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(xmin, ymin).of_size(dx, legend_size.min(dy)),
                    image::Rgb([170, 0, 0]),
                );
                let legend = format!("{}   {:.0}%", class_name(class_index), 100. * b.confidence);
                imageproc::drawing::draw_text_mut(
                    &mut img,
                    image::Rgb([255, 255, 255]),
                    xmin,
                    ymin,
                    ab_glyph::PxScale {
                        x: legend_size as f32 - 1.,
                        y: legend_size as f32 - 1.,
                    },
                    &font,
                    &legend,
                )
            }
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

/// Renders COCO-17 poses without emitting per-object output.
pub fn report_pose(
    pred: &Tensor,
    img: DynamicImage,
    transform: &ImageTransform,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
) -> Result<DynamicImage> {
    report_pose_with_output(
        pred,
        img,
        transform,
        confidence_threshold,
        nms_threshold,
        legend_size,
        ReportOutput::Quiet,
    )
}

/// Renders COCO-17 poses with an explicit per-object output policy.
pub fn report_pose_with_output(
    pred: &Tensor,
    img: DynamicImage,
    transform: &ImageTransform,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
    output: ReportOutput,
) -> Result<DynamicImage> {
    validate_source_dimensions(img.width() as usize, img.height() as usize)?;
    transform.validate_for_image(&img)?;
    validate_probability("confidence threshold", confidence_threshold)?;
    validate_probability("NMS threshold", nms_threshold)?;
    validate_legend_size(legend_size)?;
    let (pred_size, npreds) = pred.dims2()?;
    if pred_size != 17 * 3 + 4 + 1 {
        candle::bail!("pose reporter requires one class and 17 keypoints with x/y/visibility");
    }
    if npreds > MAX_REPORT_PREDICTIONS {
        candle::bail!("prediction count must not exceed {MAX_REPORT_PREDICTIONS}")
    }
    let pred = pred.to_device(&Device::Cpu)?;
    let mut bboxes = vec![];
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        if pred.iter().any(|value| !value.is_finite()) {
            continue;
        }
        let confidence = pred[4];
        if confidence > confidence_threshold {
            if bboxes.len() >= MAX_REPORT_CANDIDATES {
                candle::bail!("pose candidate count must not exceed {MAX_REPORT_CANDIDATES}")
            }
            let keypoints = (0..17)
                .map(|i| KeyPoint {
                    x: pred[3 * i + 5],
                    y: pred[3 * i + 6],
                    mask: pred[3 * i + 7],
                })
                .collect::<Vec<_>>();
            let bbox = Bbox {
                xmin: pred[0] - pred[2] / 2.,
                ymin: pred[1] - pred[3] / 2.,
                xmax: pred[0] + pred[2] / 2.,
                ymax: pred[1] + pred[3] / 2.,
                confidence,
                data: keypoints,
            };
            bboxes.push(bbox)
        }
    }

    let mut bboxes = vec![bboxes];
    non_maximum_suppression(&mut bboxes, nms_threshold);
    let bboxes = &bboxes[0];
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font: ab_glyph::FontRef =
        ab_glyph::FontRef::try_from_slice(&font).map_err(candle::Error::wrap)?;

    // Annotate the original image and optionally print box information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let mut img = img.to_rgb8();
    for b in bboxes.iter() {
        if output == ReportOutput::Stdout {
            println!("{b:?}");
        }
        let xmin_f = transform
            .source_x(b.xmin)
            .clamp(0.0, initial_w.saturating_sub(1) as f32);
        let ymin_f = transform
            .source_y(b.ymin)
            .clamp(0.0, initial_h.saturating_sub(1) as f32);
        let xmax = transform.source_x(b.xmax).clamp(0.0, initial_w as f32);
        let ymax = transform.source_y(b.ymax).clamp(0.0, initial_h as f32);
        if xmax > xmin_f && ymax > ymin_f {
            let xmin = xmin_f as i32;
            let ymin = ymin_f as i32;
            let dx = ((xmax - xmin_f).ceil() as u32).max(1);
            let dy = ((ymax - ymin_f).ceil() as u32).max(1);
            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(xmin, ymin).of_size(dx, dy),
                image::Rgb([255, 0, 0]),
            );

            if legend_size > 0 {
                let legend = format!("{}:{:.2}%", class_name(0), 100. * b.confidence);
                imageproc::drawing::draw_text_mut(
                    &mut img,
                    image::Rgb([255, 255, 255]),
                    xmin,
                    ymin,
                    ab_glyph::PxScale::from(legend_size as f32),
                    &font,
                    &legend,
                )
            }
        }
        for kp in b.data.iter() {
            if kp.mask < 0.6 {
                continue;
            }
            let x = transform
                .source_x(kp.x)
                .clamp(0.0, initial_w.saturating_sub(1) as f32) as i32;
            let y = transform
                .source_y(kp.y)
                .clamp(0.0, initial_h.saturating_sub(1) as f32) as i32;
            imageproc::drawing::draw_filled_circle_mut(
                &mut img,
                (x, y),
                2,
                image::Rgb([0, 255, 0]),
            );
        }

        for &(idx1, idx2) in KP_CONNECTIONS.iter() {
            let kp1 = &b.data[idx1];
            let kp2 = &b.data[idx2];
            if kp1.mask < 0.6 || kp2.mask < 0.6 {
                continue;
            }
            imageproc::drawing::draw_line_segment_mut(
                &mut img,
                (
                    transform.source_x(kp1.x).clamp(0.0, initial_w as f32),
                    transform.source_y(kp1.y).clamp(0.0, initial_h as f32),
                ),
                (
                    transform.source_x(kp2.x).clamp(0.0, initial_w as f32),
                    transform.source_y(kp2.y).clamp(0.0, initial_h as f32),
                ),
                image::Rgb([255, 255, 0]),
            );
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_dimensions_preserves_extreme_and_large_aspect_ratios() {
        assert_eq!(resize_dimensions(10_000, 1, 640, 32).unwrap(), (640, 1));
        assert_eq!(resize_dimensions(1, 10_000, 640, 32).unwrap(), (1, 640));
        assert!(resize_dimensions(usize::MAX, usize::MAX / 2, 640, 32).is_err());
    }

    #[test]
    fn resize_dimensions_matches_python_ties_to_even_rounding() {
        assert_eq!(resize_dimensions(256, 1, 640, 32).unwrap(), (640, 2));
        assert_eq!(resize_dimensions(256, 3, 640, 32).unwrap(), (640, 8));
        assert_eq!(resize_dimensions(1, 256, 640, 32).unwrap(), (2, 640));
    }

    #[test]
    fn prepare_image_uses_height_width_tensor_order() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(20, 10));
        let (tensor, transform) = prepare_image(&image, 64, 32, &Device::Cpu).unwrap();

        assert_eq!(tensor.dims4().unwrap(), (1, 3, 64, 64));
        assert_eq!(transform.resized_width, 64);
        assert_eq!(transform.resized_height, 32);
        assert_eq!(transform.pad_left, 0);
        assert_eq!(transform.pad_top, 16);
        assert_eq!(transform.source_y(16.0), 0.0);
        assert_eq!(transform.source_y(48.0), 10.0);
    }

    #[test]
    fn prepare_image_records_the_actual_integer_odd_padding_offset() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(640, 639));
        let (_, transform) = prepare_image(&image, 640, 32, &Device::Cpu).unwrap();

        assert_eq!(transform.resized_height, 639);
        assert_eq!(transform.pad_top, 0);
        assert_eq!(transform.canvas_height - transform.resized_height, 1);
        assert_eq!(transform.source_y(transform.pad_top as f32), 0.0);
    }

    #[test]
    fn report_detect_rejects_a_tensor_without_classes() {
        let pred = Tensor::zeros((4, 1), candle::DType::F32, &Device::Cpu).unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));

        let transform = ImageTransform {
            original_width: 8,
            original_height: 8,
            resized_width: 8,
            resized_height: 8,
            canvas_width: 8,
            canvas_height: 8,
            pad_left: 0,
            pad_top: 0,
        };
        let error = report_detect(&pred, image, &transform, 0.25, 0.45, 0).unwrap_err();

        assert!(error.to_string().contains("COCO detection reporter"));
    }

    #[test]
    fn report_detect_rejects_nonfinite_thresholds() {
        let pred = Tensor::zeros((84, 1), candle::DType::F32, &Device::Cpu).unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));

        let transform = ImageTransform {
            original_width: 8,
            original_height: 8,
            resized_width: 8,
            resized_height: 8,
            canvas_width: 8,
            canvas_height: 8,
            pad_left: 0,
            pad_top: 0,
        };
        let error = report_detect(&pred, image, &transform, f32::NAN, 0.45, 0).unwrap_err();

        assert!(error.to_string().contains("confidence threshold"));
    }

    #[test]
    fn report_pose_clips_adversarial_keypoints_before_drawing() {
        let mut values = vec![0.0_f32; 17 * 3 + 5];
        values[0..5].copy_from_slice(&[4.0, 4.0, 2.0, 2.0, 0.9]);
        for index in 0..17 {
            values[5 + index * 3] = if index % 2 == 0 { -1e30 } else { 1e30 };
            values[6 + index * 3] = if index % 2 == 0 { 1e30 } else { -1e30 };
            values[7 + index * 3] = 1.0;
        }
        let pred = Tensor::from_vec(values, (17 * 3 + 5, 1), &Device::Cpu).unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        let transform = ImageTransform {
            original_width: 8,
            original_height: 8,
            resized_width: 8,
            resized_height: 8,
            canvas_width: 8,
            canvas_height: 8,
            pad_left: 0,
            pad_top: 0,
        };

        let rendered = report_pose(&pred, image, &transform, 0.25, 0.45, 0).unwrap();

        assert_eq!((rendered.width(), rendered.height()), (8, 8));
    }

    #[test]
    fn report_rejects_unbounded_legend_sizes() {
        let pred = Tensor::zeros((84, 1), candle::DType::F32, &Device::Cpu).unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        let transform = ImageTransform {
            original_width: 8,
            original_height: 8,
            resized_width: 8,
            resized_height: 8,
            canvas_width: 8,
            canvas_height: 8,
            pad_left: 0,
            pad_top: 0,
        };

        let error = report_detect(&pred, image, &transform, 0.25, 0.45, 257).unwrap_err();

        assert!(error.to_string().contains("legend size"));
    }

    #[test]
    fn report_rejects_content_larger_than_its_declared_canvas() {
        let pred = Tensor::zeros((84, 1), candle::DType::F32, &Device::Cpu).unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        let transform = ImageTransform {
            original_width: 8,
            original_height: 8,
            resized_width: 9,
            resized_height: 8,
            canvas_width: 8,
            canvas_height: 8,
            pad_left: 0,
            pad_top: 0,
        };

        let error = report_detect(&pred, image, &transform, 0.25, 0.45, 0).unwrap_err();

        assert!(error.to_string().contains("image transform"));
    }

    #[test]
    fn report_detect_requires_the_coco_80_schema() {
        let pred = Tensor::zeros((85, 1), candle::DType::F32, &Device::Cpu).unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        let transform = ImageTransform {
            original_width: 8,
            original_height: 8,
            resized_width: 8,
            resized_height: 8,
            canvas_width: 8,
            canvas_height: 8,
            pad_left: 0,
            pad_top: 0,
        };

        let error = report_detect(&pred, image, &transform, 0.25, 0.45, 0).unwrap_err();

        assert!(error.to_string().contains("COCO detection reporter"));
    }

    #[test]
    fn report_pose_requires_the_coco_17_by_3_schema() {
        let pred = Tensor::zeros((57, 1), candle::DType::F32, &Device::Cpu).unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        let transform = ImageTransform {
            original_width: 8,
            original_height: 8,
            resized_width: 8,
            resized_height: 8,
            canvas_width: 8,
            canvas_height: 8,
            pad_left: 0,
            pad_top: 0,
        };

        let error = report_pose(&pred, image, &transform, 0.25, 0.45, 0).unwrap_err();

        assert!(error.to_string().contains("17 keypoints"));
    }

    #[test]
    fn report_detect_rejects_excessive_predictions_and_candidates() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        let transform = ImageTransform {
            original_width: 8,
            original_height: 8,
            resized_width: 8,
            resized_height: 8,
            canvas_width: 8,
            canvas_height: 8,
            pad_left: 0,
            pad_top: 0,
        };
        let too_many_predictions = Tensor::zeros(
            (4 + COCO_CLASS_COUNT, MAX_REPORT_PREDICTIONS + 1),
            candle::DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let error = report_detect(
            &too_many_predictions,
            image.clone(),
            &transform,
            0.25,
            0.45,
            0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("prediction count"));

        let mut values = vec![0.0_f32; (4 + COCO_CLASS_COUNT) * (MAX_REPORT_CANDIDATES + 1)];
        for candidate in 0..=MAX_REPORT_CANDIDATES {
            values[candidate] = 4.0;
            values[(MAX_REPORT_CANDIDATES + 1) + candidate] = 4.0;
            values[2 * (MAX_REPORT_CANDIDATES + 1) + candidate] = 2.0;
            values[3 * (MAX_REPORT_CANDIDATES + 1) + candidate] = 2.0;
            values[4 * (MAX_REPORT_CANDIDATES + 1) + candidate] = 0.9;
        }
        let too_many_candidates = Tensor::from_vec(
            values,
            (4 + COCO_CLASS_COUNT, MAX_REPORT_CANDIDATES + 1),
            &Device::Cpu,
        )
        .unwrap();
        let error =
            report_detect(&too_many_candidates, image, &transform, 0.25, 0.45, 0).unwrap_err();
        assert!(error.to_string().contains("candidate count"));
    }

    #[test]
    fn resize_dimensions_rejects_unbounded_source_pixels_and_canvas() {
        assert!(resize_dimensions(32_768, 32_768, 640, 32).is_err());
        assert!(resize_dimensions(640, 640, 8_192, 32).is_err());
    }
}
