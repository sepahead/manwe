pub mod coco_classes;
pub mod model;
pub mod native_runtime;
pub mod runtime_contract;
pub mod secure_io;
pub mod stream_url;

use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Result, Tensor};
use image::DynamicImage;
use std::io::Write;
use unicode_categories::UnicodeCategories;

const MAX_SOURCE_DIMENSION: usize = 32_768;
const MAX_SOURCE_PIXELS: usize = 64 * 1024 * 1024;
pub(crate) const MAX_INFERENCE_DIMENSION: usize = 4_096;
const COCO_CLASS_COUNT: usize = 80;
const MAX_MODEL_CLASSES: usize = 4_096;
pub(crate) const MAX_NATIVE_MODEL_CLASSES: usize = 1_000;
pub(crate) const MAX_REPORT_PREDICTIONS: usize = 100_000;
const MAX_COCO_OUTPUT_ELEMENTS: usize = (4 + COCO_CLASS_COUNT) * MAX_REPORT_PREDICTIONS;
pub(crate) const MAX_MODEL_OUTPUT_ELEMENTS: usize = 16_000_000;
pub(crate) const MAX_NMS_BOXES: usize = 2_000;
const MAX_NMS_PAIR_WORK: usize = MAX_NMS_BOXES * (MAX_NMS_BOXES - 1) / 2;
const MAX_REPORT_CANDIDATES: usize = MAX_NMS_BOXES;

pub(crate) fn is_contract_printable(value: &str) -> bool {
    value.chars().all(|character| {
        character == ' '
            || character.is_letter()
            || character.is_mark()
            || character.is_number()
            || character.is_punctuation()
            || character.is_symbol()
    })
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
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
        let expected_resized = if self.canvas_width == self.canvas_height
            && self.canvas_width > 0
            && self.canvas_width <= MAX_INFERENCE_DIMENSION
        {
            Some(resize_dimensions(
                image.width() as usize,
                image.height() as usize,
                self.canvas_width,
                1,
            )?)
        } else {
            None
        };
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
            || expected_resized != Some((self.resized_width, self.resized_height))
            || self.pad_left != (self.canvas_width - self.resized_width) / 2
            || self.pad_top != (self.canvas_height - self.resized_height) / 2
        {
            candle::bail!("image transform is inconsistent with the source image")
        }
        Ok(())
    }

    fn source_x(&self, value: f32) -> f64 {
        (f64::from(value) - self.pad_left as f64) * self.original_width as f64
            / self.resized_width as f64
    }

    fn source_y(&self, value: f32) -> f64 {
        (f64::from(value) - self.pad_top as f64) * self.original_height as f64
            / self.resized_height as f64
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

/// Canonical COCO-17 keypoint order used by the native pose adapter.
pub const COCO_KEYPOINT_NAMES: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];

/// Skeleton edges for [`COCO_KEYPOINT_NAMES`].
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

use candle_transformers::object_detection::{Bbox, KeyPoint};

/// Validate a contract-defined, unsqueezed detector output without copying it.
pub fn validate_detection_output_schema(
    output: &Tensor,
    class_count: usize,
    expected_predictions: usize,
) -> Result<()> {
    if !(1..=MAX_MODEL_CLASSES).contains(&class_count) {
        candle::bail!("detector class count must be between 1 and {MAX_MODEL_CLASSES}")
    }
    if !(1..=MAX_REPORT_PREDICTIONS).contains(&expected_predictions) {
        candle::bail!(
            "expected detector prediction count must be between 1 and {MAX_REPORT_PREDICTIONS}"
        )
    }
    if output.rank() != 3 {
        candle::bail!("detection model output must have rank 3")
    }
    let (batch, rows, predictions) = output.dims3()?;
    if batch != 1 || rows != 4 + class_count {
        candle::bail!(
            "detection model output must have shape [1, {}, predictions]",
            4 + class_count
        )
    }
    if predictions != expected_predictions {
        candle::bail!(
            "detection model output has {predictions} predictions, expected {expected_predictions}"
        )
    }
    let elements = rows
        .checked_mul(predictions)
        .ok_or_else(|| candle::Error::Msg("detection output element count overflowed".into()))?;
    if elements > MAX_MODEL_OUTPUT_ELEMENTS {
        candle::bail!("detection model output exceeds the bounded element-count limit")
    }
    if output.dtype() != candle::DType::F32 {
        candle::bail!("detection model output must use the FP32 data type")
    }
    Ok(())
}

/// Validate a contract-defined, unsqueezed single-class pose output.
pub fn validate_pose_output_schema(
    output: &Tensor,
    keypoint_count: usize,
    expected_predictions: usize,
) -> Result<()> {
    if keypoint_count == 0 || keypoint_count > MAX_MODEL_CLASSES {
        candle::bail!("pose keypoint count must be between 1 and {MAX_MODEL_CLASSES}")
    }
    if !(1..=MAX_REPORT_PREDICTIONS).contains(&expected_predictions) {
        candle::bail!(
            "expected pose prediction count must be between 1 and {MAX_REPORT_PREDICTIONS}"
        )
    }
    let feature_count = keypoint_count
        .checked_mul(3)
        .and_then(|count| count.checked_add(5))
        .ok_or_else(|| candle::Error::Msg("pose feature count overflowed".into()))?;
    if output.rank() != 3 {
        candle::bail!("pose model output must have rank 3")
    }
    let (batch, features, predictions) = output.dims3()?;
    if batch != 1 || features != feature_count {
        candle::bail!("pose model output must have shape [1, {feature_count}, predictions]")
    }
    if predictions != expected_predictions {
        candle::bail!(
            "pose model output has {predictions} predictions, expected {expected_predictions}"
        )
    }
    let elements = features
        .checked_mul(predictions)
        .ok_or_else(|| candle::Error::Msg("pose output element count overflowed".into()))?;
    if elements > MAX_MODEL_OUTPUT_ELEMENTS {
        candle::bail!("pose model output exceeds the bounded element-count limit")
    }
    if output.dtype() != candle::DType::F32 {
        candle::bail!("pose model output must use the FP32 data type")
    }
    Ok(())
}

/// Validates the fixed, unsqueezed COCO-80 detection output schema.
///
/// This function reads tensor metadata only. It does not synchronize the device
/// or copy tensor storage.
///
/// # Errors
///
/// Returns an error unless `output` is an FP32 `[1, 84, N]` tensor where `N`
/// equals `expected_predictions`. The expected count must be between 1 and
/// 100,000.
pub fn validate_coco_detection_output_schema(
    output: &Tensor,
    expected_predictions: usize,
) -> Result<()> {
    if !(1..=MAX_REPORT_PREDICTIONS).contains(&expected_predictions) {
        candle::bail!(
            "expected COCO detection prediction count must be between 1 and {MAX_REPORT_PREDICTIONS}"
        )
    }
    if output.rank() != 3 {
        candle::bail!("COCO detection model output must have rank 3")
    }
    let (batch, rows, predictions) = output.dims3()?;
    if batch != 1 {
        candle::bail!("COCO detection model output must have batch size 1")
    }
    if rows != 4 + COCO_CLASS_COUNT {
        candle::bail!(
            "COCO detection model output must have 4 box rows and {COCO_CLASS_COUNT} class rows"
        )
    }
    if predictions > MAX_REPORT_PREDICTIONS {
        candle::bail!(
            "COCO detection model output must not exceed {MAX_REPORT_PREDICTIONS} predictions"
        )
    }
    if predictions != expected_predictions {
        candle::bail!(
            "COCO detection model output has {predictions} predictions, expected {expected_predictions}"
        )
    }
    if output.dtype() != candle::DType::F32 {
        candle::bail!("COCO detection model output must use the FP32 data type")
    }
    Ok(())
}

/// Copies and validates every logical element of a bounded FP32 COCO output.
///
/// A compact copy is first created on the source device, so a small tensor view
/// cannot cause the backing allocation to be copied to the CPU. The returned
/// tensor is contiguous and CPU-resident, allowing callers to reuse the single
/// validated readback for postprocessing.
///
/// This validates content and transfer bounds, not task-specific rank or shape.
/// Empty tensors are accepted.
///
/// # Errors
///
/// Returns an error when the tensor exceeds the inference output bound, is not
/// FP32, cannot be compacted or copied to the CPU, or contains `NaN` or either
/// infinity.
pub fn validate_coco_model_output(output: &Tensor) -> Result<Tensor> {
    let element_count = output.dims().iter().try_fold(1_usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| {
            candle::Error::Msg("COCO model output element count overflowed".to_string())
        })
    })?;
    if output
        .dims()
        .iter()
        .any(|dimension| *dimension > MAX_COCO_OUTPUT_ELEMENTS)
    {
        candle::bail!("COCO model output dimension exceeds the bounded element-count limit")
    }
    if element_count > MAX_COCO_OUTPUT_ELEMENTS {
        candle::bail!("COCO model output exceeds the bounded element-count limit")
    }
    if output.dtype() != candle::DType::F32 {
        candle::bail!("model output must use the FP32 data type")
    }

    let output = output.force_contiguous()?.to_device(&Device::Cpu)?;
    let values = Vec::<f32>::try_from(output.flatten_all()?)?;
    if values.iter().any(|value| !value.is_finite()) {
        candle::bail!("model output must contain only finite values")
    }
    Ok(output)
}

fn copy_validated_model_output(output: &Tensor, max_elements: usize) -> Result<Vec<f32>> {
    let element_count = output.dims().iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| candle::Error::Msg("model output element count overflowed".to_string()))
    })?;
    if max_elements == 0
        || output
            .dims()
            .iter()
            .any(|dimension| *dimension > max_elements)
        || element_count > max_elements
    {
        candle::bail!("model output exceeds the bounded element-count limit")
    }
    if output.dtype() != candle::DType::F32 {
        candle::bail!("model output must use the FP32 data type")
    }
    let output = output.force_contiguous()?.to_device(&Device::Cpu)?;
    let values = Vec::<f32>::try_from(output.flatten_all()?)?;
    if values.iter().any(|value| !value.is_finite()) {
        candle::bail!("model output must contain only finite values")
    }
    Ok(values)
}

fn validate_probability(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        candle::bail!("{name} must be finite and between 0 and 1")
    }
    Ok(())
}

fn continuous_xyxy_iou<D>(left: &Bbox<D>, right: &Bbox<D>) -> f64 {
    let left_width = f64::from(left.xmax) - f64::from(left.xmin);
    let left_height = f64::from(left.ymax) - f64::from(left.ymin);
    let right_width = f64::from(right.xmax) - f64::from(right.xmin);
    let right_height = f64::from(right.ymax) - f64::from(right.ymin);
    let intersection_width =
        (f64::from(left.xmax.min(right.xmax)) - f64::from(left.xmin.max(right.xmin))).max(0.0);
    let intersection_height =
        (f64::from(left.ymax.min(right.ymax)) - f64::from(left.ymin.max(right.ymin))).max(0.0);
    let intersection = intersection_width * intersection_height;
    let union = left_width * left_height + right_width * right_height - intersection;

    if union > 0.0 {
        (intersection / union).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Applies greedy, class-aware NMS to continuous-coordinate `xyxy` boxes.
///
/// Each inner vector is one class. Retained boxes are ordered by descending
/// confidence, with equal-confidence boxes retaining their input order. Boxes
/// are suppressed only when their IoU is strictly greater than `iou_threshold`.
///
/// # Errors
///
/// Returns an error when the threshold or a confidence is not a probability,
/// coordinates are non-finite, a box has non-positive area, or the bounded
/// quadratic-work limit is exceeded. Validation completes before any input is
/// reordered or removed.
pub fn class_aware_non_maximum_suppression<D>(
    bboxes: &mut [Vec<Bbox<D>>],
    iou_threshold: f32,
) -> Result<()> {
    validate_probability("NMS threshold", iou_threshold)?;

    let mut box_count = 0_usize;
    let mut pair_work = 0_usize;
    for class_boxes in bboxes.iter() {
        if class_boxes.len() > MAX_NMS_BOXES - box_count {
            candle::bail!("NMS input exceeds the bounded box-count limit")
        }
        box_count += class_boxes.len();

        let class_pair_work = class_boxes
            .len()
            .checked_mul(class_boxes.len().saturating_sub(1))
            .ok_or_else(|| candle::Error::Msg("NMS pair-work calculation overflowed".into()))?
            / 2;
        if class_pair_work > MAX_NMS_PAIR_WORK - pair_work {
            candle::bail!("NMS input exceeds the bounded quadratic-work limit")
        }
        pair_work += class_pair_work;

        for bbox in class_boxes {
            if [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
                .iter()
                .any(|coordinate| !coordinate.is_finite())
            {
                candle::bail!("NMS boxes must contain only finite coordinates")
            }
            if bbox.xmax <= bbox.xmin || bbox.ymax <= bbox.ymin {
                candle::bail!("NMS requires positive-area xyxy boxes")
            }
            validate_probability("NMS confidence", bbox.confidence)?;
        }
    }

    let iou_threshold = f64::from(iou_threshold);
    for class_boxes in bboxes.iter_mut() {
        class_boxes.sort_by(|left, right| {
            if left.confidence == right.confidence {
                std::cmp::Ordering::Equal
            } else {
                right.confidence.total_cmp(&left.confidence)
            }
        });
        let mut retained = 0_usize;
        for index in 0..class_boxes.len() {
            let should_drop = (0..retained).any(|previous| {
                continuous_xyxy_iou(&class_boxes[previous], &class_boxes[index]) > iou_threshold
            });
            if !should_drop {
                class_boxes.swap(retained, index);
                retained += 1;
            }
        }
        class_boxes.truncate(retained);
    }

    Ok(())
}

fn validate_legend_size(value: u32) -> Result<()> {
    if value > 256 {
        candle::bail!("legend size must not exceed 256 pixels")
    }
    Ok(())
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

/// One retained detector result in original-image pixel coordinates.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct DetectionRecord {
    pub prediction_index: usize,
    /// Index in the contract-declared source taxonomy.
    pub class_index: usize,
    /// Name in the contract-declared source taxonomy.
    pub class_name: String,
    /// Optional mapping into Manwe's five-class airspace taxonomy.
    pub airspace_class: Option<String>,
    pub confidence: f32,
    pub bbox_xyxy: [f32; 4],
}

/// Structured detections and the corresponding annotated image.
#[derive(Debug)]
pub struct DetectionReport {
    pub detections: Vec<DetectionRecord>,
    pub annotated_image: DynamicImage,
}

/// One contract-named pose keypoint in original-image pixel coordinates.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PoseKeypointRecord {
    pub index: usize,
    pub name: String,
    pub xy: [f32; 2],
    pub confidence: f32,
}

/// One retained pose result in original-image pixel coordinates.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PoseRecord {
    pub prediction_index: usize,
    /// Index in the contract-declared source taxonomy.
    pub class_index: usize,
    /// Name in the contract-declared source taxonomy.
    pub class_name: String,
    /// Optional mapping into Manwe's five-class airspace taxonomy.
    pub airspace_class: Option<String>,
    pub confidence: f32,
    pub bbox_xyxy: [f32; 4],
    pub keypoints: Vec<PoseKeypointRecord>,
}

/// Structured poses and the corresponding annotated image.
#[derive(Debug)]
pub struct PoseReport {
    pub poses: Vec<PoseRecord>,
    pub annotated_image: DynamicImage,
}

fn validate_names(names: &[String], kind: &str) -> Result<()> {
    if names.is_empty() || names.len() > MAX_MODEL_CLASSES {
        candle::bail!("{kind} must contain 1..={MAX_MODEL_CLASSES} entries")
    }
    let mut unique = std::collections::HashSet::with_capacity(names.len());
    for name in names {
        if name.is_empty()
            || name.trim() != name
            || name.len() > 256
            || !is_contract_printable(name)
        {
            candle::bail!("{kind} must be bounded, nonempty, printable strings")
        }
        if !unique.insert(name.as_str()) {
            candle::bail!("{kind} must be unique")
        }
    }
    Ok(())
}

fn validate_class_names(class_names: &[String]) -> Result<()> {
    validate_names(class_names, "class names")
}

/// Decode, suppress, structure, and render one contract-defined detector output.
#[allow(clippy::too_many_arguments)]
pub fn report_detect_with_names(
    pred: &Tensor,
    img: DynamicImage,
    transform: &ImageTransform,
    class_names: &[String],
    expected_predictions: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    max_detections: usize,
    legend_size: u32,
) -> Result<DetectionReport> {
    validate_source_dimensions(img.width() as usize, img.height() as usize)?;
    transform.validate_for_image(&img)?;
    validate_class_names(class_names)?;
    validate_probability("confidence threshold", confidence_threshold)?;
    validate_probability("NMS threshold", nms_threshold)?;
    validate_legend_size(legend_size)?;
    if !(1..=MAX_REPORT_PREDICTIONS).contains(&expected_predictions) {
        candle::bail!(
            "expected detector prediction count must be between 1 and {MAX_REPORT_PREDICTIONS}"
        )
    }
    if !(1..=MAX_NMS_BOXES).contains(&max_detections) {
        candle::bail!("max detections must be between 1 and {MAX_NMS_BOXES}")
    }
    let (rows, predictions) = pred.dims2()?;
    if rows != 4 + class_names.len() || predictions != expected_predictions {
        candle::bail!(
            "detection reporter expected [{}, {expected_predictions}], got [{rows}, {predictions}]",
            4 + class_names.len()
        )
    }
    let element_count = rows
        .checked_mul(predictions)
        .ok_or_else(|| candle::Error::Msg("detection output element count overflowed".into()))?;
    if element_count > MAX_MODEL_OUTPUT_ELEMENTS {
        candle::bail!("detection model output exceeds the bounded element-count limit")
    }
    let values = copy_validated_model_output(pred, MAX_MODEL_OUTPUT_ELEMENTS)?;

    let mut candidates: Vec<Vec<Bbox<usize>>> =
        (0..class_names.len()).map(|_| Vec::new()).collect();
    let mut candidate_count = 0_usize;
    for prediction_index in 0..predictions {
        let width = values[2 * predictions + prediction_index];
        let height = values[3 * predictions + prediction_index];
        if width <= 0.0 || height <= 0.0 {
            candle::bail!("detection output must contain positive box widths and heights")
        }
        let mut class_index = 0_usize;
        let mut confidence = values[4 * predictions + prediction_index];
        for index in 0..class_names.len() {
            let score = values[(4 + index) * predictions + prediction_index];
            validate_probability("model class score", score)?;
            if score > confidence {
                confidence = score;
                class_index = index;
            }
        }
        if confidence <= confidence_threshold {
            continue;
        }
        if candidate_count >= MAX_REPORT_CANDIDATES {
            candle::bail!("detection candidate count must not exceed {MAX_REPORT_CANDIDATES}")
        }
        let center_x = values[prediction_index];
        let center_y = values[predictions + prediction_index];
        candidates[class_index].push(Bbox {
            xmin: center_x - width / 2.0,
            ymin: center_y - height / 2.0,
            xmax: center_x + width / 2.0,
            ymax: center_y + height / 2.0,
            confidence,
            data: prediction_index,
        });
        candidate_count += 1;
    }
    class_aware_non_maximum_suppression(&mut candidates, nms_threshold)?;

    let mut retained: Vec<(usize, usize, Bbox<usize>)> = candidates
        .into_iter()
        .enumerate()
        .flat_map(|(class_index, boxes)| {
            boxes
                .into_iter()
                .map(move |bbox| (class_index, bbox.data, bbox))
        })
        .collect();
    retained.sort_by(|left, right| {
        right
            .2
            .confidence
            .total_cmp(&left.2.confidence)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.0.cmp(&right.0))
    });

    let image_height = img.height();
    let image_width = img.width();
    let mut detections = Vec::with_capacity(max_detections.min(retained.len()));
    for (class_index, prediction_index, bbox) in retained {
        let xmin = transform
            .source_x(bbox.xmin)
            .clamp(0.0, image_width.saturating_sub(1) as f64) as f32;
        let ymin = transform
            .source_y(bbox.ymin)
            .clamp(0.0, image_height.saturating_sub(1) as f64) as f32;
        let xmax = transform.source_x(bbox.xmax).clamp(0.0, image_width as f64) as f32;
        let ymax = transform
            .source_y(bbox.ymax)
            .clamp(0.0, image_height as f64) as f32;
        if xmax <= xmin || ymax <= ymin {
            continue;
        }
        detections.push(DetectionRecord {
            prediction_index,
            class_index,
            class_name: class_names[class_index].clone(),
            airspace_class: None,
            confidence: bbox.confidence,
            bbox_xyxy: [xmin, ymin, xmax, ymax],
        });
        if detections.len() == max_detections {
            break;
        }
    }

    let mut rendered = img.to_rgb8();
    let font = if legend_size == 0 {
        None
    } else {
        Some(
            ab_glyph::FontRef::try_from_slice(include_bytes!("roboto-mono-stripped.ttf"))
                .map_err(candle::Error::wrap)?,
        )
    };
    for detection in &detections {
        let [xmin, ymin, xmax, ymax] = detection.bbox_xyxy;
        let x = xmin as i32;
        let y = ymin as i32;
        let width = ((xmax - xmin).ceil() as u32).max(1);
        let height = ((ymax - ymin).ceil() as u32).max(1);
        imageproc::drawing::draw_hollow_rect_mut(
            &mut rendered,
            imageproc::rect::Rect::at(x, y).of_size(width, height),
            image::Rgb([255, 0, 0]),
        );
        if let Some(font) = &font {
            imageproc::drawing::draw_filled_rect_mut(
                &mut rendered,
                imageproc::rect::Rect::at(x, y).of_size(width, legend_size.min(height)),
                image::Rgb([170, 0, 0]),
            );
            let legend = format!(
                "{}   {:.0}%",
                detection.class_name,
                100.0 * detection.confidence
            );
            imageproc::drawing::draw_text_mut(
                &mut rendered,
                image::Rgb([255, 255, 255]),
                x,
                y,
                ab_glyph::PxScale::from(legend_size as f32),
                font,
                &legend,
            );
        }
    }
    Ok(DetectionReport {
        detections,
        annotated_image: DynamicImage::ImageRgb8(rendered),
    })
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
    let (rows, predictions) = pred.dims2()?;
    if rows != 4 + COCO_CLASS_COUNT {
        candle::bail!(
            "COCO detection reporter requires 4 box rows and {COCO_CLASS_COUNT} class rows"
        )
    }
    if predictions > MAX_REPORT_PREDICTIONS {
        candle::bail!("prediction count must not exceed {MAX_REPORT_PREDICTIONS}")
    }
    let class_names = crate::coco_classes::NAMES
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let report = report_detect_with_names(
        pred,
        img,
        transform,
        &class_names,
        predictions,
        confidence_threshold,
        nms_threshold,
        MAX_REPORT_CANDIDATES,
        legend_size,
    )?;
    if output == ReportOutput::Stdout {
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        for detection in &report.detections {
            writeln!(stdout, "{}: {detection:?}", detection.class_name)
                .map_err(candle::Error::wrap)?;
        }
        stdout.flush().map_err(candle::Error::wrap)?;
    }
    Ok(report.annotated_image)
}

/// Decode, suppress, structure, and render one contract-defined pose output.
#[allow(clippy::too_many_arguments)]
pub fn report_pose_with_names(
    pred: &Tensor,
    img: DynamicImage,
    transform: &ImageTransform,
    class_names: &[String],
    keypoint_names: &[String],
    expected_predictions: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    keypoint_confidence_threshold: f32,
    max_detections: usize,
    legend_size: u32,
) -> Result<PoseReport> {
    validate_source_dimensions(img.width() as usize, img.height() as usize)?;
    transform.validate_for_image(&img)?;
    validate_class_names(class_names)?;
    if class_names.len() != 1 {
        candle::bail!("pose reporter requires exactly one class name")
    }
    validate_names(keypoint_names, "keypoint names")?;
    validate_probability("confidence threshold", confidence_threshold)?;
    validate_probability("NMS threshold", nms_threshold)?;
    validate_probability(
        "keypoint confidence threshold",
        keypoint_confidence_threshold,
    )?;
    validate_legend_size(legend_size)?;
    if !(1..=MAX_REPORT_PREDICTIONS).contains(&expected_predictions) {
        candle::bail!(
            "expected pose prediction count must be between 1 and {MAX_REPORT_PREDICTIONS}"
        )
    }
    if !(1..=MAX_NMS_BOXES).contains(&max_detections) {
        candle::bail!("max detections must be between 1 and {MAX_NMS_BOXES}")
    }
    let (features, predictions) = pred.dims2()?;
    let expected_features = keypoint_names
        .len()
        .checked_mul(3)
        .and_then(|count| count.checked_add(5))
        .ok_or_else(|| candle::Error::Msg("pose feature count overflowed".into()))?;
    if features != expected_features || predictions != expected_predictions {
        candle::bail!(
            "pose reporter expected [{expected_features}, {expected_predictions}], got [{features}, {predictions}]"
        )
    }
    let element_count = features
        .checked_mul(predictions)
        .ok_or_else(|| candle::Error::Msg("pose output element count overflowed".into()))?;
    if element_count > MAX_MODEL_OUTPUT_ELEMENTS {
        candle::bail!("pose model output exceeds the bounded element-count limit")
    }
    let values = copy_validated_model_output(pred, MAX_MODEL_OUTPUT_ELEMENTS)?;

    let mut candidates = Vec::new();
    for prediction_index in 0..predictions {
        let width = values[2 * predictions + prediction_index];
        let height = values[3 * predictions + prediction_index];
        if width <= 0.0 || height <= 0.0 {
            candle::bail!("pose output must contain positive box widths and heights")
        }
        let confidence = values[4 * predictions + prediction_index];
        validate_probability("model class score", confidence)?;
        for keypoint_index in 0..keypoint_names.len() {
            let mask_row = 5 + 3 * keypoint_index + 2;
            validate_probability(
                "model keypoint confidence",
                values[mask_row * predictions + prediction_index],
            )?;
        }
        if confidence <= confidence_threshold {
            continue;
        }
        if candidates.len() >= MAX_REPORT_CANDIDATES {
            candle::bail!("pose candidate count must not exceed {MAX_REPORT_CANDIDATES}")
        }
        let keypoints: Vec<KeyPoint> = (0..keypoint_names.len())
            .map(|keypoint_index| {
                let base = 5 + 3 * keypoint_index;
                KeyPoint {
                    x: values[base * predictions + prediction_index],
                    y: values[(base + 1) * predictions + prediction_index],
                    mask: values[(base + 2) * predictions + prediction_index],
                }
            })
            .collect();
        let center_x = values[prediction_index];
        let center_y = values[predictions + prediction_index];
        candidates.push(Bbox {
            xmin: center_x - width / 2.0,
            ymin: center_y - height / 2.0,
            xmax: center_x + width / 2.0,
            ymax: center_y + height / 2.0,
            confidence,
            data: (prediction_index, keypoints),
        });
    }

    let mut classes = vec![candidates];
    class_aware_non_maximum_suppression(&mut classes, nms_threshold)?;
    let mut retained = classes.pop().unwrap_or_default();
    retained.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.data.0.cmp(&right.data.0))
    });

    let image_width = img.width();
    let image_height = img.height();
    let mut poses = Vec::with_capacity(max_detections.min(retained.len()));
    for bbox in retained {
        let xmin = transform
            .source_x(bbox.xmin)
            .clamp(0.0, image_width.saturating_sub(1) as f64) as f32;
        let ymin = transform
            .source_y(bbox.ymin)
            .clamp(0.0, image_height.saturating_sub(1) as f64) as f32;
        let xmax = transform.source_x(bbox.xmax).clamp(0.0, image_width as f64) as f32;
        let ymax = transform
            .source_y(bbox.ymax)
            .clamp(0.0, image_height as f64) as f32;
        if xmax <= xmin || ymax <= ymin {
            continue;
        }
        let keypoints = bbox
            .data
            .1
            .iter()
            .enumerate()
            .map(|(index, keypoint)| PoseKeypointRecord {
                index,
                name: keypoint_names[index].clone(),
                xy: [
                    transform
                        .source_x(keypoint.x)
                        .clamp(0.0, image_width.saturating_sub(1) as f64)
                        as f32,
                    transform
                        .source_y(keypoint.y)
                        .clamp(0.0, image_height.saturating_sub(1) as f64)
                        as f32,
                ],
                confidence: keypoint.mask,
            })
            .collect();
        poses.push(PoseRecord {
            prediction_index: bbox.data.0,
            class_index: 0,
            class_name: class_names[0].clone(),
            airspace_class: None,
            confidence: bbox.confidence,
            bbox_xyxy: [xmin, ymin, xmax, ymax],
            keypoints,
        });
        if poses.len() == max_detections {
            break;
        }
    }

    let canonical_coco_order = keypoint_names.len() == COCO_KEYPOINT_NAMES.len()
        && keypoint_names
            .iter()
            .zip(COCO_KEYPOINT_NAMES)
            .all(|(actual, expected)| actual == expected);
    let font = if legend_size == 0 {
        None
    } else {
        Some(
            ab_glyph::FontRef::try_from_slice(include_bytes!("roboto-mono-stripped.ttf"))
                .map_err(candle::Error::wrap)?,
        )
    };
    let mut rendered = img.to_rgb8();
    for pose in &poses {
        let [xmin, ymin, xmax, ymax] = pose.bbox_xyxy;
        let x = xmin as i32;
        let y = ymin as i32;
        let width = ((xmax - xmin).ceil() as u32).max(1);
        let height = ((ymax - ymin).ceil() as u32).max(1);
        imageproc::drawing::draw_hollow_rect_mut(
            &mut rendered,
            imageproc::rect::Rect::at(x, y).of_size(width, height),
            image::Rgb([255, 0, 0]),
        );
        if let Some(font) = &font {
            let legend = format!("{}   {:.0}%", pose.class_name, 100.0 * pose.confidence);
            imageproc::drawing::draw_text_mut(
                &mut rendered,
                image::Rgb([255, 255, 255]),
                x,
                y,
                ab_glyph::PxScale::from(legend_size as f32),
                font,
                &legend,
            );
        }
        for keypoint in &pose.keypoints {
            if keypoint.confidence < keypoint_confidence_threshold {
                continue;
            }
            imageproc::drawing::draw_filled_circle_mut(
                &mut rendered,
                (keypoint.xy[0] as i32, keypoint.xy[1] as i32),
                2,
                image::Rgb([0, 255, 0]),
            );
        }
        if canonical_coco_order {
            for &(left, right) in &KP_CONNECTIONS {
                let left = &pose.keypoints[left];
                let right = &pose.keypoints[right];
                if left.confidence < keypoint_confidence_threshold
                    || right.confidence < keypoint_confidence_threshold
                {
                    continue;
                }
                imageproc::drawing::draw_line_segment_mut(
                    &mut rendered,
                    (left.xy[0], left.xy[1]),
                    (right.xy[0], right.xy[1]),
                    image::Rgb([255, 255, 0]),
                );
            }
        }
    }
    Ok(PoseReport {
        poses,
        annotated_image: DynamicImage::ImageRgb8(rendered),
    })
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
    let (features, predictions) = pred.dims2()?;
    if features != 17 * 3 + 5 {
        candle::bail!("pose reporter requires one class and 17 keypoints with x/y/visibility")
    }
    if predictions == 0 || predictions > MAX_REPORT_PREDICTIONS {
        candle::bail!("prediction count must be between 1 and {MAX_REPORT_PREDICTIONS}")
    }
    let class_names = vec!["person".to_owned()];
    let keypoint_names = COCO_KEYPOINT_NAMES
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let report = report_pose_with_names(
        pred,
        img,
        transform,
        &class_names,
        &keypoint_names,
        predictions,
        confidence_threshold,
        nms_threshold,
        0.6,
        MAX_REPORT_CANDIDATES,
        legend_size,
    )?;
    if output == ReportOutput::Stdout {
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        for pose in &report.poses {
            writeln!(stdout, "{}: {pose:?}", pose.class_name).map_err(candle::Error::wrap)?;
        }
        stdout.flush().map_err(candle::Error::wrap)?;
    }
    Ok(report.annotated_image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nms_box<D>(coordinates: [f32; 4], confidence: f32, data: D) -> Bbox<D> {
        Bbox {
            xmin: coordinates[0],
            ymin: coordinates[1],
            xmax: coordinates[2],
            ymax: coordinates[3],
            confidence,
            data,
        }
    }

    #[test]
    fn nms_uses_continuous_coordinate_iou() {
        let mut classes = vec![vec![
            nms_box([0.0, 0.0, 1.0, 1.0], 0.9, 0),
            nms_box([0.5, 0.0, 1.5, 1.0], 0.8, 1),
        ]];

        class_aware_non_maximum_suppression(&mut classes, 0.45).unwrap();

        assert_eq!(
            classes[0].iter().map(|bbox| bbox.data).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn nms_does_not_suppress_boxes_from_different_classes() {
        let mut classes = vec![
            vec![nms_box([0.0, 0.0, 1.0, 1.0], 0.9, ())],
            vec![nms_box([0.0, 0.0, 1.0, 1.0], 0.8, ())],
        ];

        class_aware_non_maximum_suppression(&mut classes, 0.0).unwrap();

        assert_eq!(classes.iter().map(Vec::len).collect::<Vec<_>>(), vec![1, 1]);
    }

    #[test]
    fn nms_keeps_boxes_at_the_exact_iou_threshold() {
        let mut classes = vec![vec![
            nms_box([0.0, 0.0, 2.0, 1.0], 0.9, 0),
            nms_box([0.0, 0.0, 1.0, 1.0], 0.8, 1),
        ]];

        class_aware_non_maximum_suppression(&mut classes, 0.5).unwrap();

        assert_eq!(classes[0].len(), 2);
    }

    #[test]
    fn nms_threshold_one_keeps_identical_boxes() {
        let mut classes = vec![vec![
            nms_box([0.0, 0.0, 1.0, 1.0], 0.9, 0),
            nms_box([0.0, 0.0, 1.0, 1.0], 0.8, 1),
        ]];

        class_aware_non_maximum_suppression(&mut classes, 1.0).unwrap();

        assert_eq!(classes[0].len(), 2);
    }

    #[test]
    fn nms_orders_scores_descending_and_preserves_input_order_for_ties() {
        let mut classes = vec![vec![
            nms_box([0.0, 0.0, 1.0, 1.0], 0.2, 0),
            nms_box([2.0, 0.0, 3.0, 1.0], 0.9, 1),
            nms_box([4.0, 0.0, 5.0, 1.0], 0.9, 2),
        ]];

        class_aware_non_maximum_suppression(&mut classes, 0.45).unwrap();

        assert_eq!(
            classes[0].iter().map(|bbox| bbox.data).collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn nms_equal_score_overlap_keeps_the_first_box() {
        let mut classes = vec![vec![
            nms_box([0.0, 0.0, 1.0, 1.0], 0.5, 0),
            nms_box([0.0, 0.0, 1.0, 1.0], 0.5, 1),
        ]];

        class_aware_non_maximum_suppression(&mut classes, 0.45).unwrap();

        assert_eq!((classes[0].len(), classes[0][0].data), (1, 0));
    }

    #[test]
    fn nms_treats_signed_zero_scores_as_a_stable_tie() {
        let mut classes = vec![vec![
            nms_box([0.0, 0.0, 1.0, 1.0], -0.0, 0),
            nms_box([2.0, 0.0, 3.0, 1.0], 0.0, 1),
        ]];

        class_aware_non_maximum_suppression(&mut classes, 0.45).unwrap();

        assert_eq!(
            classes[0].iter().map(|bbox| bbox.data).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn nms_rejects_nonfinite_coordinates_without_mutating_input() {
        let mut classes = vec![
            vec![
                nms_box([2.0, 0.0, 3.0, 1.0], 0.2, 0),
                nms_box([0.0, 0.0, 1.0, 1.0], 0.9, 1),
            ],
            vec![nms_box([f32::NAN, 0.0, 1.0, 1.0], 0.8, 2)],
        ];

        let error = class_aware_non_maximum_suppression(&mut classes, 0.45).unwrap_err();

        assert_eq!(
            (
                error.to_string().contains("finite coordinates"),
                classes[0][0].data,
                classes[0][1].data,
            ),
            (true, 0, 1)
        );
    }

    #[test]
    fn nms_rejects_non_positive_area_boxes() {
        let mut classes = vec![vec![nms_box([1.0, 0.0, 1.0, 1.0], 0.9, ())]];

        let error = class_aware_non_maximum_suppression(&mut classes, 0.45).unwrap_err();

        assert!(error.to_string().contains("positive-area"));
    }

    #[test]
    fn nms_rejects_nonfinite_confidence() {
        let mut classes = vec![vec![nms_box([0.0, 0.0, 1.0, 1.0], f32::NAN, ())]];

        let error = class_aware_non_maximum_suppression(&mut classes, 0.45).unwrap_err();

        assert!(error.to_string().contains("NMS confidence"));
    }

    #[test]
    fn nms_rejects_nonfinite_threshold() {
        let mut classes = vec![vec![nms_box([0.0, 0.0, 1.0, 1.0], 0.9, ())]];

        let error = class_aware_non_maximum_suppression(&mut classes, f32::NAN).unwrap_err();

        assert!(error.to_string().contains("NMS threshold"));
    }

    #[test]
    fn nms_rejects_inputs_above_the_bounded_work_limit() {
        let mut classes = vec![(0..=MAX_NMS_BOXES)
            .map(|index| nms_box([index as f32, 0.0, index as f32 + 0.5, 1.0], 0.9, ()))
            .collect::<Vec<_>>()];

        let error = class_aware_non_maximum_suppression(&mut classes, 0.45).unwrap_err();

        assert!(error.to_string().contains("bounded box-count"));
    }

    #[test]
    fn nms_iou_remains_defined_for_extreme_f32_boxes() {
        let mut classes = vec![vec![
            nms_box([-f32::MAX, -f32::MAX, f32::MAX, f32::MAX], 0.9, 0),
            nms_box([-f32::MAX, -f32::MAX, f32::MAX, f32::MAX], 0.8, 1),
        ]];

        class_aware_non_maximum_suppression(&mut classes, 0.45).unwrap();

        assert_eq!((classes[0].len(), classes[0][0].data), (1, 0));
    }

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
    fn inverse_transform_keeps_extreme_f32_coordinates_finite() {
        let transform = ImageTransform {
            original_width: MAX_SOURCE_DIMENSION,
            original_height: MAX_SOURCE_DIMENSION,
            resized_width: 1,
            resized_height: 1,
            canvas_width: 1,
            canvas_height: 1,
            pad_left: 0,
            pad_top: 0,
        };

        assert!(transform.source_x(f32::MAX).is_finite());
        assert!(transform.source_y(-f32::MAX).is_finite());
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
    fn coco_detection_schema_accepts_the_maximum_prediction_count_without_readback() {
        let output = Tensor::zeros(1, candle::DType::F32, &Device::Cpu)
            .unwrap()
            .broadcast_as((1, 4 + COCO_CLASS_COUNT, MAX_REPORT_PREDICTIONS))
            .unwrap();

        let result = validate_coco_detection_output_schema(&output, MAX_REPORT_PREDICTIONS);

        assert!(result.is_ok(), "fixed COCO schema was rejected: {result:?}");
    }

    #[test]
    fn coco_detection_schema_rejects_a_squeezed_rank_two_tensor() {
        let output =
            Tensor::zeros((4 + COCO_CLASS_COUNT, 1), candle::DType::F32, &Device::Cpu).unwrap();

        let error = validate_coco_detection_output_schema(&output, 1).unwrap_err();

        assert!(error.to_string().contains("rank 3"));
    }

    #[test]
    fn coco_detection_schema_rejects_multiple_batches() {
        let output = Tensor::zeros(
            (2, 4 + COCO_CLASS_COUNT, 1),
            candle::DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let error = validate_coco_detection_output_schema(&output, 1).unwrap_err();

        assert!(error.to_string().contains("batch size 1"));
    }

    #[test]
    fn coco_detection_schema_rejects_the_wrong_row_count() {
        let output = Tensor::zeros(
            (1, 4 + COCO_CLASS_COUNT - 1, 1),
            candle::DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let error = validate_coco_detection_output_schema(&output, 1).unwrap_err();

        assert!(error.to_string().contains("class rows"));
    }

    #[test]
    fn coco_detection_schema_rejects_excessive_predictions_without_readback() {
        let output = Tensor::zeros(1, candle::DType::F32, &Device::Cpu)
            .unwrap()
            .broadcast_as((1, 4 + COCO_CLASS_COUNT, MAX_REPORT_PREDICTIONS + 1))
            .unwrap();

        let error =
            validate_coco_detection_output_schema(&output, MAX_REPORT_PREDICTIONS).unwrap_err();

        assert!(error.to_string().contains("predictions"));
    }

    #[test]
    fn coco_detection_schema_rejects_non_fp32_metadata() {
        let output = Tensor::zeros(
            (1, 4 + COCO_CLASS_COUNT, 1),
            candle::DType::F64,
            &Device::Cpu,
        )
        .unwrap();

        let error = validate_coco_detection_output_schema(&output, 1).unwrap_err();

        assert!(error.to_string().contains("FP32 data type"));
    }

    #[test]
    fn coco_detection_schema_rejects_zero_predictions() {
        let output = Tensor::zeros(
            (1, 4 + COCO_CLASS_COUNT, 0),
            candle::DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let error = validate_coco_detection_output_schema(&output, 1).unwrap_err();

        assert!(error.to_string().contains("expected 1"));
    }

    #[test]
    fn coco_detection_schema_rejects_an_unexpected_bounded_prediction_count() {
        let output = Tensor::zeros(
            (1, 4 + COCO_CLASS_COUNT, 2),
            candle::DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let error = validate_coco_detection_output_schema(&output, 1).unwrap_err();

        assert!(error.to_string().contains("has 2 predictions, expected 1"));
    }

    #[test]
    fn coco_detection_schema_rejects_a_zero_expected_prediction_count() {
        let output = Tensor::zeros(
            (1, 4 + COCO_CLASS_COUNT, 1),
            candle::DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let error = validate_coco_detection_output_schema(&output, 0).unwrap_err();

        assert!(error.to_string().contains("between 1"));
    }

    #[test]
    fn coco_detection_schema_rejects_an_excessive_expected_prediction_count() {
        let output = Tensor::zeros(
            (1, 4 + COCO_CLASS_COUNT, 1),
            candle::DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let error =
            validate_coco_detection_output_schema(&output, MAX_REPORT_PREDICTIONS + 1).unwrap_err();

        assert!(error.to_string().contains("between 1"));
    }

    #[test]
    fn validate_coco_model_output_accepts_only_finite_values() {
        let output = Tensor::from_vec(
            vec![-f32::MAX, -0.0, 0.0, f32::MAX],
            (1, 2, 2),
            &Device::Cpu,
        )
        .unwrap();

        let result = validate_coco_model_output(&output);

        assert!(result.is_ok(), "finite output was rejected: {result:?}");
    }

    #[test]
    fn validate_coco_model_output_accepts_an_empty_tensor() {
        let output = Tensor::zeros(0, candle::DType::F32, &Device::Cpu).unwrap();

        let result = validate_coco_model_output(&output);

        assert!(result.is_ok(), "empty output was rejected: {result:?}");
    }

    #[test]
    fn validate_coco_model_output_accepts_a_noncontiguous_finite_view() {
        let output = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap();

        let result = validate_coco_model_output(&output);

        assert!(
            result.is_ok(),
            "noncontiguous finite output was rejected: {result:?}"
        );
    }

    #[test]
    fn validate_coco_model_output_compacts_a_narrow_backing_view() {
        let mut values = vec![f32::NAN; 4_096];
        values[0] = 3.0;
        let backing = Tensor::from_vec(values, (1, 1, 4_096), &Device::Cpu).unwrap();
        let output = backing.narrow(2, 0, 1).unwrap();

        let validated = validate_coco_model_output(&output).unwrap();
        let values = Vec::<f32>::try_from(validated.flatten_all().unwrap()).unwrap();

        assert_eq!(values, vec![3.0]);
    }

    #[test]
    fn validate_coco_model_output_rejects_a_non_fp32_tensor() {
        let output = Tensor::zeros(1, candle::DType::F64, &Device::Cpu).unwrap();

        let error = validate_coco_model_output(&output).unwrap_err();

        assert!(error.to_string().contains("FP32 data type"));
    }

    #[test]
    fn validate_coco_model_output_rejects_an_excessive_broadcast_before_copying() {
        let output = Tensor::zeros(1, candle::DType::F32, &Device::Cpu)
            .unwrap()
            .broadcast_as(MAX_COCO_OUTPUT_ELEMENTS + 1)
            .unwrap();

        let error = validate_coco_model_output(&output).unwrap_err();

        assert!(error.to_string().contains("element-count limit"));
    }

    #[test]
    fn validate_coco_model_output_rejects_shape_product_overflow_without_panicking() {
        let output = Tensor::zeros(1, candle::DType::F32, &Device::Cpu)
            .unwrap()
            .broadcast_as((usize::MAX, usize::MAX))
            .unwrap();

        let error = validate_coco_model_output(&output).unwrap_err();

        assert!(error.to_string().contains("element count overflowed"));
    }

    #[test]
    fn validate_coco_model_output_rejects_nan_beyond_the_first_value() {
        let output = Tensor::from_vec(vec![0.0, 1.0, f32::NAN], (1, 1, 3), &Device::Cpu).unwrap();

        let error = validate_coco_model_output(&output).unwrap_err();

        assert!(error.to_string().contains("finite values"));
    }

    #[test]
    fn validate_coco_model_output_rejects_positive_infinity() {
        let output = Tensor::from_vec(vec![f32::INFINITY], (1, 1, 1), &Device::Cpu).unwrap();

        let error = validate_coco_model_output(&output).unwrap_err();

        assert!(error.to_string().contains("finite values"));
    }

    #[test]
    fn validate_coco_model_output_rejects_negative_infinity() {
        let output = Tensor::from_vec(vec![f32::NEG_INFINITY], (1, 1, 1), &Device::Cpu).unwrap();

        let error = validate_coco_model_output(&output).unwrap_err();

        assert!(error.to_string().contains("finite values"));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn validate_coco_model_output_rejects_nan_after_metal_synchronization() {
        let device = Device::new_metal(0).unwrap();
        let output = Tensor::from_vec(vec![0.0, f32::NAN], (1, 1, 2), &device).unwrap();

        let error = validate_coco_model_output(&output).unwrap_err();

        assert!(error.to_string().contains("finite values"));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn validate_coco_model_output_compacts_a_narrow_metal_backing_view() {
        let device = Device::new_metal(0).unwrap();
        let mut values = vec![f32::NAN; 4_096];
        values[0] = 7.0;
        let backing = Tensor::from_vec(values, (1, 1, 4_096), &device).unwrap();
        let output = backing.narrow(2, 0, 1).unwrap();

        let validated = validate_coco_model_output(&output).unwrap();
        let values = Vec::<f32>::try_from(validated.flatten_all().unwrap()).unwrap();

        assert_eq!(values, vec![7.0]);
    }

    #[test]
    fn report_detect_rejects_nan_in_a_later_low_confidence_prediction() {
        let mut values = vec![0.0_f32; (4 + COCO_CLASS_COUNT) * 2];
        values[(4 + COCO_CLASS_COUNT) * 2 - 1] = f32::NAN;
        let pred = Tensor::from_vec(values, (4 + COCO_CLASS_COUNT, 2), &Device::Cpu).unwrap();
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

        assert!(error.to_string().contains("finite values"));
    }

    #[test]
    fn report_pose_rejects_infinite_low_confidence_keypoint() {
        let mut values = vec![0.0_f32; 17 * 3 + 5];
        let last = values.len() - 1;
        values[last] = f32::INFINITY;
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

        let error = report_pose(&pred, image, &transform, 0.25, 0.45, 0).unwrap_err();

        assert!(error.to_string().contains("finite values"));
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
    fn report_rejects_anisotropic_transform_geometry() {
        let pred = Tensor::zeros((84, 1), candle::DType::F32, &Device::Cpu).unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(16, 8));
        let transform = ImageTransform {
            original_width: 16,
            original_height: 8,
            resized_width: 8,
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
    fn contract_defined_detection_reporting_is_generic_structured_and_bounded() {
        let values = vec![
            4.0_f32, 4.2, // cx
            4.0, 4.0, // cy
            2.0, 2.0, // width
            2.0, 2.0, // height
            0.9, 0.8, // class alpha
            0.1, 0.95, // class beta
        ];
        let pred = Tensor::from_vec(values, (6, 2), &Device::Cpu).unwrap();
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
        let class_names = vec!["alpha".to_string(), "beta".to_string()];

        let report =
            report_detect_with_names(&pred, image, &transform, &class_names, 2, 0.25, 0.45, 1, 0)
                .unwrap();
        assert_eq!(report.detections.len(), 1);
        assert_eq!(report.detections[0].prediction_index, 1);
        assert_eq!(report.detections[0].class_index, 1);
        assert_eq!(report.detections[0].class_name, "beta");
        assert_eq!(report.detections[0].airspace_class, None);
        assert_eq!(report.detections[0].confidence, 0.95);
        for (actual, expected) in report.detections[0]
            .bbox_xyxy
            .iter()
            .zip([3.2, 3.0, 5.2, 5.0])
        {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn generic_detection_schema_and_decoder_reject_interface_or_probability_drift() {
        let output = Tensor::zeros((1, 9, 8400), candle::DType::F32, &Device::Cpu).unwrap();
        validate_detection_output_schema(&output, 5, 8400).unwrap();
        assert!(validate_detection_output_schema(&output, 80, 8400).is_err());

        let mut values = vec![0.0_f32; 6];
        values[2] = 1.0;
        values[3] = 1.0;
        values[4] = 1.01;
        let pred = Tensor::from_vec(values, (6, 1), &Device::Cpu).unwrap();
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
        let error = report_detect_with_names(
            &pred,
            image,
            &transform,
            &["alpha".to_string(), "beta".to_string()],
            1,
            0.25,
            0.45,
            1,
            0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("model class score"));

        for invalid_count in [0, MAX_REPORT_PREDICTIONS + 1] {
            let pred = Tensor::zeros((6, 1), candle::DType::F32, &Device::Cpu).unwrap();
            let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
            let error = report_detect_with_names(
                &pred,
                image,
                &transform,
                &["alpha".to_string(), "beta".to_string()],
                invalid_count,
                0.25,
                0.45,
                1,
                0,
            )
            .unwrap_err();
            assert!(error.to_string().contains("prediction count"));
        }
    }

    #[test]
    fn contract_defined_pose_reporting_is_structured_and_validates_every_keypoint() {
        let output = Tensor::zeros((1, 11, 1), candle::DType::F32, &Device::Cpu).unwrap();
        validate_pose_output_schema(&output, 2, 1).unwrap();
        assert!(validate_pose_output_schema(&output, 17, 1).is_err());

        let values = vec![4.0_f32, 4.0, 2.0, 2.0, 0.9, 1.0, 2.0, 0.8, 7.0, 6.0, 0.4];
        let pred = Tensor::from_vec(values, (11, 1), &Device::Cpu).unwrap();
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
        let report = report_pose_with_names(
            &pred,
            image.clone(),
            &transform,
            &["person".to_owned()],
            &["left".to_owned(), "right".to_owned()],
            1,
            0.25,
            0.45,
            0.6,
            1,
            0,
        )
        .unwrap();
        assert_eq!(report.poses.len(), 1);
        assert_eq!(report.poses[0].prediction_index, 0);
        assert_eq!(report.poses[0].bbox_xyxy, [3.0, 3.0, 5.0, 5.0]);
        assert_eq!(report.poses[0].keypoints[0].name, "left");
        assert_eq!(report.poses[0].keypoints[0].xy, [1.0, 2.0]);

        let mut invalid = vec![0.0_f32; 11];
        invalid[2] = 1.0;
        invalid[3] = 1.0;
        invalid[7] = 1.01;
        let invalid = Tensor::from_vec(invalid, (11, 1), &Device::Cpu).unwrap();
        let error = report_pose_with_names(
            &invalid,
            image,
            &transform,
            &["person".to_owned()],
            &["left".to_owned(), "right".to_owned()],
            1,
            0.25,
            0.45,
            0.6,
            1,
            0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("keypoint confidence"));

        for invalid_count in [0, MAX_REPORT_PREDICTIONS + 1] {
            let pred = Tensor::zeros((11, 1), candle::DType::F32, &Device::Cpu).unwrap();
            let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
            let error = report_pose_with_names(
                &pred,
                image,
                &transform,
                &["person".to_owned()],
                &["left".to_owned(), "right".to_owned()],
                invalid_count,
                0.25,
                0.45,
                0.6,
                1,
                0,
            )
            .unwrap_err();
            assert!(error.to_string().contains("prediction count"));
        }
    }

    #[test]
    fn resize_dimensions_rejects_unbounded_source_pixels_and_canvas() {
        assert!(resize_dimensions(32_768, 32_768, 640, 32).is_err());
        assert!(resize_dimensions(640, 640, 8_192, 32).is_err());
    }
}
