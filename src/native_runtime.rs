//! Contract-bound native Candle model loading and inference.
//!
//! Every native entry point uses this module so model architecture, tensor
//! shapes, preprocessing, labels, and postprocessing cannot drift between the
//! batch CLI and the live viewer.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use image::DynamicImage;

use crate::model::{YoloV8, YoloV8Pose};
use crate::runtime_contract::{ModelContract, ModelTask, VerifiedModelPackage};
use crate::{
    device, prepare_image, report_detect_with_names, report_pose_with_names,
    validate_detection_output_schema, validate_pose_output_schema, DetectionReport, PoseReport,
};

enum RuntimeModel {
    Detect(Box<YoloV8>),
    Pose(Box<YoloV8Pose>),
}

impl RuntimeModel {
    fn forward(&self, input: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Detect(model) => model.forward(input),
            Self::Pose(model) => model.forward(input),
        }
    }
}

/// Structured result from one contract-bound native inference.
#[derive(Debug)]
pub enum NativeInferenceReport {
    Detect(DetectionReport),
    Pose(PoseReport),
}

impl NativeInferenceReport {
    /// Consume the structured result and return its annotated image.
    #[must_use]
    pub fn into_annotated_image(self) -> DynamicImage {
        match self {
            Self::Detect(report) => report.annotated_image,
            Self::Pose(report) => report.annotated_image,
        }
    }
}

/// One integrity-verified model instance and its immutable execution contract.
pub struct NativeRuntime {
    contract: ModelContract,
    contract_path: PathBuf,
    contract_sha256: String,
    artifact_path: PathBuf,
    artifact_size: usize,
    device: Device,
    device_name: &'static str,
    model: RuntimeModel,
}

impl NativeRuntime {
    /// Integrity-check a schema-2 package and construct exactly the declared model.
    pub fn load(contract_path: &Path, cpu: bool) -> Result<Self> {
        let package = VerifiedModelPackage::load(contract_path)?;
        Self::from_verified_package(package, cpu)
    }

    /// Construct a runtime from a package already verified by the caller.
    pub fn from_verified_package(package: VerifiedModelPackage, cpu: bool) -> Result<Self> {
        let (contract, contract_path, contract_sha256, artifact_path, artifact_bytes) =
            package.into_parts();
        let artifact_size = artifact_bytes.len();
        let device = device(cpu)?;
        let device_name = if device.is_cuda() {
            "cuda"
        } else if device.is_metal() {
            "metal"
        } else {
            "cpu"
        };
        let builder = VarBuilder::from_buffered_safetensors(artifact_bytes, DType::F32, &device)
            .context("failed to decode verified Candle model weights")?;
        let model = match contract.task() {
            ModelTask::Detect => RuntimeModel::Detect(Box::new(YoloV8::load(
                builder,
                contract.variant().multiples(),
                contract.class_count(),
            )?)),
            ModelTask::Pose => RuntimeModel::Pose(Box::new(YoloV8Pose::load(
                builder,
                contract.variant().multiples(),
                contract.class_count(),
                (contract.keypoint_names().len(), 3),
            )?)),
        };
        Ok(Self {
            contract,
            contract_path,
            contract_sha256,
            artifact_path,
            artifact_size,
            device,
            device_name,
            model,
        })
    }

    /// Execute preprocessing, inference, validation, postprocessing, and rendering.
    pub fn infer(&self, image: DynamicImage, legend_size: u32) -> Result<NativeInferenceReport> {
        let (input, transform) = prepare_image(
            &image,
            self.contract.input_size(),
            self.contract.input_alignment(),
            &self.device,
        )?;
        let output = self.model.forward(&input)?;
        match self.contract.task() {
            ModelTask::Detect => {
                validate_detection_output_schema(
                    &output,
                    self.contract.class_count(),
                    self.contract.expected_predictions(),
                )?;
                let mut report = report_detect_with_names(
                    &output.squeeze(0)?,
                    image,
                    &transform,
                    self.contract.class_names(),
                    self.contract.expected_predictions(),
                    self.contract.confidence_threshold(),
                    self.contract.iou_threshold(),
                    self.contract.max_detections(),
                    legend_size,
                )?;
                for detection in &mut report.detections {
                    detection.airspace_class = self
                        .contract
                        .airspace_class(detection.class_index)
                        .map(str::to_owned);
                }
                Ok(NativeInferenceReport::Detect(report))
            }
            ModelTask::Pose => {
                validate_pose_output_schema(
                    &output,
                    self.contract.keypoint_names().len(),
                    self.contract.expected_predictions(),
                )?;
                let keypoint_threshold = self
                    .contract
                    .keypoint_confidence_threshold()
                    .context("validated pose contract lost its keypoint confidence threshold")?;
                let mut report = report_pose_with_names(
                    &output.squeeze(0)?,
                    image,
                    &transform,
                    self.contract.class_names(),
                    self.contract.keypoint_names(),
                    self.contract.expected_predictions(),
                    self.contract.confidence_threshold(),
                    self.contract.iou_threshold(),
                    keypoint_threshold,
                    self.contract.max_detections(),
                    legend_size,
                )?;
                for pose in &mut report.poses {
                    pose.airspace_class = self
                        .contract
                        .airspace_class(pose.class_index)
                        .map(str::to_owned);
                }
                Ok(NativeInferenceReport::Pose(report))
            }
        }
    }

    #[must_use]
    pub fn contract(&self) -> &ModelContract {
        &self.contract
    }

    #[must_use]
    pub fn contract_path(&self) -> &Path {
        &self.contract_path
    }

    #[must_use]
    pub fn contract_sha256(&self) -> &str {
        &self.contract_sha256
    }

    #[must_use]
    pub fn artifact_path(&self) -> &Path {
        &self.artifact_path
    }

    #[must_use]
    pub fn artifact_size(&self) -> usize {
        self.artifact_size
    }

    #[must_use]
    pub fn device_name(&self) -> &'static str {
        self.device_name
    }
}
