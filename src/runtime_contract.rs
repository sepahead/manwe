//! Schema-2 model contracts consumed by the native Candle runtime.
//!
//! A contract is the only authority for model bytes, tensor shapes, class names,
//! preprocessing, decoding, and postprocessing. Parsing is bounded,
//! duplicate-free, and closed to unknown fields; loading integrity-binds the exact
//! sibling artifact before Candle sees any weight bytes. The operator remains
//! responsible for trusting the selected contract's provenance.

use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context, Result};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;

use crate::model::Multiples;
use crate::secure_io::{sha256_hex, BoundDirectory, MAX_MODEL_BYTES};
use crate::{
    is_contract_printable, COCO_KEYPOINT_NAMES, MAX_INFERENCE_DIMENSION, MAX_MODEL_OUTPUT_ELEMENTS,
    MAX_NATIVE_MODEL_CLASSES, MAX_NMS_BOXES, MAX_REPORT_PREDICTIONS,
};

const SCHEMA_VERSION: &str = "2.0";
const MAX_CONTRACT_BYTES: u64 = 1 << 20;
const MAX_JSON_NODES: usize = 100_000;
const MAX_JSON_DEPTH: usize = 32;
const MAX_CLASSES: usize = 4_096;
const MAX_TEXT_BYTES: usize = 4_096;
const MAX_CLASS_NAME_BYTES: usize = 256;

const AIRSPACE_CLASSES: [&str; 5] = ["drone", "bird", "aircraft", "helicopter", "unknown"];
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelTask {
    Detect,
    Pose,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelVariant {
    N,
    S,
    M,
    L,
    X,
}

impl ModelVariant {
    #[must_use]
    pub fn multiples(self) -> Multiples {
        match self {
            Self::N => Multiples::n(),
            Self::S => Multiples::s(),
            Self::M => Multiples::m(),
            Self::L => Multiples::l(),
            Self::X => Multiples::x(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorSpec {
    name: String,
    shape: Vec<TensorDimension>,
    dtype: String,
    layout: String,
    notes: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TensorDimension {
    Fixed(u64),
    Symbolic(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageInputSpec {
    tensor_name: String,
    width: u64,
    height: u64,
    dtype: String,
    layout: String,
    color_order: String,
    resize: String,
    interpolation: String,
    alignment: u64,
    pad_value: u64,
    scale: f64,
    mean: [f64; 3],
    std: [f64; 3],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DetectionOutputSpec {
    tensor_name: String,
    box_format: String,
    coordinate_space: String,
    class_scores: String,
    confidence_threshold: f64,
    iou_threshold: f64,
    max_detections: u64,
    keypoint_confidence_threshold: NullableThreshold,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NullableThreshold {
    Value(f64),
    Null(()),
}

impl NullableThreshold {
    fn value(&self) -> Option<f64> {
        match self {
            Self::Value(value) => Some(*value),
            Self::Null(()) => None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeSpec {
    adapter: String,
    model_variant: Option<String>,
    input: ImageInputSpec,
    output: DetectionOutputSpec,
    keypoint_names: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelContract {
    schema_version: String,
    model_name: String,
    model_version: String,
    source: String,
    rights: String,
    backend: String,
    file_path: String,
    file_sha256: String,
    num_classes: u64,
    source_classes: Vec<String>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    preprocess: String,
    postprocess: String,
    class_map: BTreeMap<String, Option<String>>,
    validation_data: String,
    benchmark_context: String,
    failure_behavior: String,
    source_sha256: String,
    export_options: String,
    signature_evidence: String,
    runtime: RuntimeSpec,
}

#[derive(Debug)]
pub struct VerifiedModelPackage {
    contract: ModelContract,
    contract_path: PathBuf,
    contract_sha256: String,
    artifact_path: PathBuf,
    artifact_bytes: Vec<u8>,
}

impl VerifiedModelPackage {
    /// Load a bounded contract and verify its exact sibling model bytes.
    pub fn load(contract_path: &Path) -> Result<Self> {
        Self::load_with_artifact_limit(contract_path, MAX_MODEL_BYTES)
    }

    /// Load a package while enforcing a caller-specific model-memory ceiling.
    ///
    /// The limit may narrow, but never widen, Manwe's global native artifact
    /// bound. Callers with an aggregate memory budget can therefore reject an
    /// oversized package before allocating and hashing its complete artifact.
    pub fn load_with_artifact_limit(contract_path: &Path, max_artifact_bytes: u64) -> Result<Self> {
        if max_artifact_bytes == 0 || max_artifact_bytes > MAX_MODEL_BYTES {
            anyhow::bail!(
                "native model artifact limit must be between 1 and {MAX_MODEL_BYTES} bytes"
            )
        }
        let parent = contract_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let name = contract_path
            .file_name()
            .context("contract path must identify one file")?;
        let parent = BoundDirectory::open(parent).context("failed to bind contract directory")?;
        let authenticated_contract_path = parent.path().join(name);
        let contract_bytes = parent
            .read_bounded_regular_file_entry(name, MAX_CONTRACT_BYTES)
            .with_context(|| {
                format!(
                    "failed to read model contract {}",
                    authenticated_contract_path.display()
                )
            })?;
        let contract_sha256 = sha256_hex(&contract_bytes);
        let contract = ModelContract::from_slice(&contract_bytes)?;
        let artifact_path = parent.path().join(contract.artifact_name());
        let artifact_bytes = parent
            .read_bounded_regular_file_entry(
                std::ffi::OsStr::new(contract.artifact_name()),
                max_artifact_bytes,
            )
            .with_context(|| {
                format!("failed to read model artifact {}", artifact_path.display())
            })?;
        let actual_sha256 = sha256_hex(&artifact_bytes);
        if actual_sha256 != contract.artifact_sha256() {
            anyhow::bail!(
                "model artifact SHA-256 mismatch for {}: expected {}, got {}",
                artifact_path.display(),
                contract.artifact_sha256(),
                actual_sha256
            )
        }
        parent.verify()?;
        Ok(Self {
            contract,
            contract_path: authenticated_contract_path,
            contract_sha256,
            artifact_path,
            artifact_bytes,
        })
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
    pub fn artifact_path(&self) -> &Path {
        &self.artifact_path
    }

    #[must_use]
    pub fn artifact_bytes(&self) -> &[u8] {
        &self.artifact_bytes
    }

    #[must_use]
    pub fn contract_sha256(&self) -> &str {
        &self.contract_sha256
    }

    #[must_use]
    pub fn into_parts(self) -> (ModelContract, PathBuf, String, PathBuf, Vec<u8>) {
        (
            self.contract,
            self.contract_path,
            self.contract_sha256,
            self.artifact_path,
            self.artifact_bytes,
        )
    }
}

impl ModelContract {
    /// Parse and validate one duplicate-free schema-2 contract without I/O.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() || bytes.len() as u64 > MAX_CONTRACT_BYTES {
            anyhow::bail!("model contract must contain 1..={MAX_CONTRACT_BYTES} bytes")
        }
        audit_json(bytes)?;
        let contract: Self =
            serde_json::from_slice(bytes).context("invalid model contract JSON")?;
        contract.validate()?;
        Ok(contract)
    }

    fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            anyhow::bail!(
                "unsupported model contract schema {:?}; expected {SCHEMA_VERSION:?}",
                self.schema_version
            )
        }
        for (name, value) in [
            ("model_name", &self.model_name),
            ("model_version", &self.model_version),
            ("source", &self.source),
            ("rights", &self.rights),
            ("preprocess", &self.preprocess),
            ("postprocess", &self.postprocess),
            ("validation_data", &self.validation_data),
            ("benchmark_context", &self.benchmark_context),
            ("failure_behavior", &self.failure_behavior),
            ("export_options", &self.export_options),
            ("signature_evidence", &self.signature_evidence),
        ] {
            validate_text(name, value, MAX_TEXT_BYTES)?;
        }
        validate_digest("file_sha256", &self.file_sha256)?;
        validate_digest("source_sha256", &self.source_sha256)?;
        if self.backend != "candle" {
            anyhow::bail!("native runtime requires backend=\"candle\"")
        }
        validate_artifact_name(&self.file_path)?;
        if !self
            .file_path
            .to_ascii_lowercase()
            .ends_with(".safetensors")
        {
            anyhow::bail!("Candle artifact filename must end in .safetensors")
        }

        let class_count = usize::try_from(self.num_classes)
            .context("num_classes cannot be represented on this platform")?;
        if !(1..=MAX_CLASSES).contains(&class_count) {
            anyhow::bail!("num_classes must be between 1 and {MAX_CLASSES}")
        }
        if self.source_classes.len() != class_count {
            anyhow::bail!("source_classes length must equal num_classes")
        }
        let mut class_names = HashSet::with_capacity(class_count);
        for (index, name) in self.source_classes.iter().enumerate() {
            validate_text(
                &format!("source_classes[{index}]"),
                name,
                MAX_CLASS_NAME_BYTES,
            )?;
            if !class_names.insert(name.as_str()) {
                anyhow::bail!("source_classes must contain unique names")
            }
        }
        validate_class_map(&self.class_map, class_count)?;

        if self.inputs.len() != 1 || self.outputs.len() != 1 {
            anyhow::bail!("native runtime requires exactly one input and one output tensor")
        }
        validate_tensor("inputs[0]", &self.inputs[0])?;
        validate_tensor("outputs[0]", &self.outputs[0])?;
        self.runtime
            .validate(class_count, &self.inputs[0], &self.outputs[0])
    }

    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    #[must_use]
    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    #[must_use]
    pub fn artifact_name(&self) -> &str {
        &self.file_path
    }

    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        &self.file_sha256
    }

    #[must_use]
    pub fn class_names(&self) -> &[String] {
        &self.source_classes
    }

    #[must_use]
    pub fn class_count(&self) -> usize {
        self.source_classes.len()
    }

    /// Return the optional Manwe airspace mapping for one source-class index.
    #[must_use]
    pub fn airspace_class(&self, source_class_index: usize) -> Option<&str> {
        self.class_map
            .get(&source_class_index.to_string())
            .and_then(|value| value.as_deref())
    }

    #[must_use]
    pub fn task(&self) -> ModelTask {
        if self.runtime.adapter.ends_with("-pose-v1") {
            ModelTask::Pose
        } else {
            ModelTask::Detect
        }
    }

    #[must_use]
    pub fn variant(&self) -> ModelVariant {
        match self.runtime.model_variant.as_deref() {
            Some("n") => ModelVariant::N,
            Some("s") => ModelVariant::S,
            Some("m") => ModelVariant::M,
            Some("l") => ModelVariant::L,
            Some("x") => ModelVariant::X,
            _ => unreachable!("validated native contract has one model variant"),
        }
    }

    #[must_use]
    pub fn input_size(&self) -> usize {
        self.runtime.input.width as usize
    }

    #[must_use]
    pub fn input_alignment(&self) -> usize {
        self.runtime.input.alignment as usize
    }

    #[must_use]
    pub fn expected_predictions(&self) -> usize {
        match self.outputs[0].shape[2] {
            TensorDimension::Fixed(value) => value as usize,
            TensorDimension::Symbolic(_) => unreachable!("native output shape is concrete"),
        }
    }

    #[must_use]
    pub fn confidence_threshold(&self) -> f32 {
        self.runtime.output.confidence_threshold as f32
    }

    #[must_use]
    pub fn iou_threshold(&self) -> f32 {
        self.runtime.output.iou_threshold as f32
    }

    #[must_use]
    pub fn max_detections(&self) -> usize {
        self.runtime.output.max_detections as usize
    }

    #[must_use]
    pub fn keypoint_names(&self) -> &[String] {
        &self.runtime.keypoint_names
    }

    #[must_use]
    pub fn keypoint_confidence_threshold(&self) -> Option<f32> {
        self.runtime
            .output
            .keypoint_confidence_threshold
            .value()
            .map(|value| value as f32)
    }
}

impl RuntimeSpec {
    fn validate(&self, class_count: usize, input: &TensorSpec, output: &TensorSpec) -> Result<()> {
        let task = match self.adapter.as_str() {
            "manwe-candle-yolov8-detect-v1" => ModelTask::Detect,
            "manwe-candle-yolov8-pose-v1" => ModelTask::Pose,
            _ => anyhow::bail!("unsupported native runtime adapter {:?}", self.adapter),
        };
        if !matches!(
            self.model_variant.as_deref(),
            Some("n" | "s" | "m" | "l" | "x")
        ) {
            anyhow::bail!("native runtime model_variant must be n, s, m, l, or x")
        }
        if class_count > MAX_NATIVE_MODEL_CLASSES {
            anyhow::bail!("native runtime supports at most {MAX_NATIVE_MODEL_CLASSES} classes")
        }
        if self.input.tensor_name != input.name || self.output.tensor_name != output.name {
            anyhow::bail!("runtime tensor names do not match the declared tensor interface")
        }
        let width = usize::try_from(self.input.width).context("input width is too large")?;
        let height = usize::try_from(self.input.height).context("input height is too large")?;
        let alignment =
            usize::try_from(self.input.alignment).context("input alignment is too large")?;
        if width != height
            || !(32..=MAX_INFERENCE_DIMENSION).contains(&width)
            || alignment != 32
            || !width.is_multiple_of(alignment)
        {
            anyhow::bail!("native input must be square, 32-aligned, and between 32 and 4096")
        }
        if self.input.dtype != "float32"
            || self.input.layout != "NCHW/RGB"
            || self.input.color_order != "RGB"
            || self.input.resize != "letterbox"
            || self.input.interpolation != "catmull-rom"
            || self.input.pad_value != 114
            || self.input.scale != 1.0 / 255.0
            || self.input.mean != [0.0; 3]
            || self.input.std != [1.0; 3]
        {
            anyhow::bail!("runtime input preprocessing does not match native prepare_image")
        }
        require_fixed_shape(input, &[1, 3, height, width], "native input")?;
        if input.dtype != "float32" || input.layout != "NCHW/RGB" {
            anyhow::bail!("native input tensor must be float32 NCHW/RGB")
        }

        if self.output.box_format != "cxcywh"
            || self.output.coordinate_space != "input-pixels"
            || self.output.class_scores != "probabilities"
        {
            anyhow::bail!("runtime output decoding does not match native YOLOv8 decoding")
        }
        validate_native_probability("confidence_threshold", self.output.confidence_threshold)?;
        validate_native_probability("iou_threshold", self.output.iou_threshold)?;
        let max_detections =
            usize::try_from(self.output.max_detections).context("max_detections is too large")?;
        if !(1..=MAX_NMS_BOXES).contains(&max_detections) {
            anyhow::bail!("max_detections must be between 1 and {MAX_NMS_BOXES}")
        }
        let keypoints = match task {
            ModelTask::Detect => {
                if !self.keypoint_names.is_empty() {
                    anyhow::bail!("detection runtime must not declare keypoint names")
                }
                if self.output.keypoint_confidence_threshold.value().is_some() {
                    anyhow::bail!(
                        "detection runtime must not declare a keypoint confidence threshold"
                    )
                }
                0
            }
            ModelTask::Pose => {
                if class_count != 1 {
                    anyhow::bail!("native YOLOv8 pose runtime requires exactly one class")
                }
                if self.keypoint_names != COCO_KEYPOINT_NAMES {
                    anyhow::bail!(
                        "native YOLOv8 pose runtime requires canonical COCO keypoint order"
                    )
                }
                let keypoint_threshold = self
                    .output
                    .keypoint_confidence_threshold
                    .value()
                    .context("pose runtime requires a keypoint confidence threshold")?;
                validate_native_probability("keypoint_confidence_threshold", keypoint_threshold)?;
                COCO_KEYPOINT_NAMES.len()
            }
        };
        let expected_features = 4 + class_count + 3 * keypoints;
        let output_shape = fixed_shape(output, "native output")?;
        if output_shape.len() != 3
            || output_shape[0] != 1
            || output_shape[1] != expected_features
            || !(1..=MAX_REPORT_PREDICTIONS).contains(&output_shape[2])
        {
            anyhow::bail!(
                "native output must have shape [1, {expected_features}, predictions] with 1..={MAX_REPORT_PREDICTIONS} predictions"
            )
        }
        let expected_predictions =
            [8_usize, 16, 32]
                .into_iter()
                .try_fold(0_usize, |total, stride| {
                    (height / stride)
                        .checked_mul(width / stride)
                        .and_then(|grid| total.checked_add(grid))
                        .context("native prediction-grid size overflowed")
                })?;
        if output_shape[2] != expected_predictions {
            anyhow::bail!(
                "native YOLOv8 output prediction count must equal its three stride-grid sizes ({expected_predictions})"
            )
        }
        let output_elements = output_shape[1]
            .checked_mul(output_shape[2])
            .context("native output element count overflowed")?;
        if output_elements > MAX_MODEL_OUTPUT_ELEMENTS {
            anyhow::bail!("native output exceeds the bounded element-count limit")
        }
        if output.dtype != "float32" || !output.layout.is_empty() {
            anyhow::bail!("native output tensor must be float32 with an empty layout")
        }
        Ok(())
    }
}

fn validate_tensor(name: &str, tensor: &TensorSpec) -> Result<()> {
    validate_identifier(&format!("{name}.name"), &tensor.name)?;
    if tensor.notes.len() > MAX_TEXT_BYTES || !is_contract_printable(&tensor.notes) {
        anyhow::bail!("{name}.notes must be a bounded printable string")
    }
    if tensor.shape.is_empty() || tensor.shape.len() > 16 {
        anyhow::bail!("{name}.shape must contain 1..=16 dimensions")
    }
    for dimension in &tensor.shape {
        match dimension {
            TensorDimension::Fixed(0) => anyhow::bail!("{name}.shape dimensions must be positive"),
            TensorDimension::Fixed(value) if *value > i32::MAX as u64 => {
                anyhow::bail!("{name}.shape dimension exceeds i32::MAX")
            }
            TensorDimension::Symbolic(value) => {
                validate_text(&format!("{name}.shape symbol"), value, 32)?;
            }
            TensorDimension::Fixed(_) => {}
        }
    }
    validate_text(&format!("{name}.dtype"), &tensor.dtype, 32)?;
    if tensor.layout.len() > 32 || !is_contract_printable(&tensor.layout) {
        anyhow::bail!("{name}.layout is invalid")
    }
    Ok(())
}

fn fixed_shape(tensor: &TensorSpec, name: &str) -> Result<Vec<usize>> {
    tensor
        .shape
        .iter()
        .map(|dimension| match dimension {
            TensorDimension::Fixed(value) => {
                usize::try_from(*value).context("tensor dimension is too large")
            }
            TensorDimension::Symbolic(value) => {
                anyhow::bail!("{name} must use concrete dimensions, got {value:?}")
            }
        })
        .collect()
}

fn require_fixed_shape(tensor: &TensorSpec, expected: &[usize], name: &str) -> Result<()> {
    if fixed_shape(tensor, name)? != expected {
        anyhow::bail!("{name} tensor shape does not match the runtime specification")
    }
    Ok(())
}

fn validate_class_map(
    class_map: &BTreeMap<String, Option<String>>,
    class_count: usize,
) -> Result<()> {
    if class_map.len() != class_count {
        anyhow::bail!("class_map must cover every source class exactly once")
    }
    for index in 0..class_count {
        let key = index.to_string();
        let value = class_map
            .get(&key)
            .with_context(|| format!("class_map is missing index {index}"))?;
        if let Some(value) = value {
            if !AIRSPACE_CLASSES.contains(&value.as_str()) {
                anyhow::bail!("class_map[{index}] is not a Manwe airspace class")
            }
        }
    }
    for key in class_map.keys() {
        let Ok(index) = key.parse::<usize>() else {
            anyhow::bail!("class_map keys must be canonical in-range decimal indices")
        };
        if index >= class_count || key != &index.to_string() {
            anyhow::bail!("class_map keys must be canonical in-range decimal indices")
        }
    }
    Ok(())
}

fn validate_text(name: &str, value: &str, max_bytes: usize) -> Result<()> {
    if value.is_empty()
        || value.trim() != value
        || value.len() > max_bytes
        || !is_contract_printable(value)
    {
        anyhow::bail!("{name} must be a bounded, nonempty, printable string")
    }
    Ok(())
}

fn validate_identifier(name: &str, value: &str) -> Result<()> {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        anyhow::bail!("{name} must be a canonical tensor identifier")
    };
    if value.len() > 128
        || !(first.is_ascii_alphabetic() || first == b'_')
        || bytes.any(|byte| {
            !(byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'/' | b'-'))
        })
    {
        anyhow::bail!("{name} must be a canonical tensor identifier")
    }
    Ok(())
}

fn validate_digest(name: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        anyhow::bail!("{name} must be 64 lowercase hexadecimal characters")
    }
    Ok(())
}

fn validate_artifact_name(value: &str) -> Result<()> {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        anyhow::bail!("file_path must be one portable ASCII artifact filename")
    };
    if value.len() > 255
        || !first.is_ascii_alphanumeric()
        || bytes.any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')))
    {
        anyhow::bail!("file_path must be one portable ASCII artifact filename")
    }
    Ok(())
}

fn validate_probability(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        anyhow::bail!("{name} must be finite and between 0 and 1")
    }
    Ok(())
}

fn validate_native_probability(name: &str, value: f64) -> Result<()> {
    validate_probability(name, value)?;
    let narrowed = value as f32;
    if (value > 0.0 && narrowed == 0.0) || (value < 1.0 && narrowed == 1.0) {
        anyhow::bail!("{name} collapses to a binary32 probability endpoint")
    }
    Ok(())
}

#[derive(Clone)]
struct JsonAudit {
    depth: usize,
    nodes: Rc<Cell<usize>>,
}

impl JsonAudit {
    fn child(&self) -> Self {
        Self {
            depth: self.depth + 1,
            nodes: Rc::clone(&self.nodes),
        }
    }

    fn count<E: de::Error>(&self) -> std::result::Result<(), E> {
        if self.depth > MAX_JSON_DEPTH {
            return Err(E::custom(format_args!(
                "model contract JSON exceeds {MAX_JSON_DEPTH} levels"
            )));
        }
        let nodes = self.nodes.get().saturating_add(1);
        if nodes > MAX_JSON_NODES {
            return Err(E::custom(format_args!(
                "model contract JSON exceeds {MAX_JSON_NODES} nodes"
            )));
        }
        self.nodes.set(nodes);
        Ok(())
    }
}

impl<'de> DeserializeSeed<'de> for JsonAudit {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        self.count::<D::Error>()?;
        deserializer.deserialize_any(JsonAuditVisitor(self))
    }
}

struct JsonAuditVisitor(JsonAudit);

impl<'de> Visitor<'de> for JsonAuditVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("bounded duplicate-free JSON")
    }

    fn visit_bool<E>(self, _value: bool) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<(), E>
    where
        E: de::Error,
    {
        if value.is_finite() {
            Ok(())
        } else {
            Err(E::custom(
                "model contract JSON contains a non-finite number",
            ))
        }
    }

    fn visit_str<E>(self, _value: &str) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_none<E>(self) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element_seed(self.0.child())?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<(), A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format_args!(
                    "model contract JSON contains duplicate field {key:?}"
                )));
            }
            map.next_value_seed(self.0.child())?;
        }
        Ok(())
    }
}

fn audit_json(bytes: &[u8]) -> Result<()> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    JsonAudit {
        depth: 1,
        nodes: Rc::new(Cell::new(0)),
    }
    .deserialize(&mut deserializer)
    .context("model contract JSON failed structural validation")?;
    deserializer
        .end()
        .context("model contract JSON has trailing content")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    const FIXTURE: &[u8] =
        include_bytes!("../python/src/manwe/schemas/model-contract-v2.candle-detect.example.json");
    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn create() -> Self {
            loop {
                let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "manwe-model-contract-test-{}-{sequence}",
                    std::process::id()
                ));
                match std::fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("failed to create test directory: {error}"),
                }
            }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn cross_language_fixture_parses_into_native_runtime_authority() {
        let contract = ModelContract::from_slice(FIXTURE).unwrap();
        assert_eq!(contract.model_name(), "manwe-airspace-yolov8");
        assert_eq!(contract.task(), ModelTask::Detect);
        assert_eq!(contract.variant(), ModelVariant::S);
        assert_eq!(contract.class_count(), 5);
        assert_eq!(contract.airspace_class(1), Some("bird"));
        assert_eq!(contract.input_size(), 640);
        assert_eq!(contract.input_alignment(), 32);
        assert_eq!(contract.expected_predictions(), 8400);
        assert_eq!(contract.max_detections(), 300);
        assert_eq!(contract.keypoint_confidence_threshold(), None);
    }

    #[test]
    fn parser_rejects_duplicates_unknown_fields_and_runtime_drift() {
        let text = std::str::from_utf8(FIXTURE).unwrap();
        let duplicate = text.replacen(
            "\"schema_version\": \"2.0\"",
            "\"schema_version\": \"2.0\", \"schema_version\": \"2.0\"",
            1,
        );
        assert!(ModelContract::from_slice(duplicate.as_bytes())
            .unwrap_err()
            .to_string()
            .contains("structural validation"));

        let unknown = text.replacen(
            "\"model_name\":",
            "\"unexpected\": true, \"model_name\":",
            1,
        );
        assert!(ModelContract::from_slice(unknown.as_bytes()).is_err());

        let drift = text.replacen(
            "\"tensor_name\": \"predictions\"",
            "\"tensor_name\": \"wrong\"",
            1,
        );
        assert!(ModelContract::from_slice(drift.as_bytes())
            .unwrap_err()
            .to_string()
            .contains("tensor names"));

        let scale_drift = text.replacen(
            "\"scale\": 0.00392156862745098",
            "\"scale\": 0.003921568627450981",
            1,
        );
        assert!(ModelContract::from_slice(scale_drift.as_bytes())
            .unwrap_err()
            .to_string()
            .contains("preprocessing"));

        let prediction_drift = text.replacen("8400]", "8401]", 1);
        assert!(ModelContract::from_slice(prediction_drift.as_bytes())
            .unwrap_err()
            .to_string()
            .contains("stride-grid sizes"));

        let contract = ModelContract::from_slice(FIXTURE).unwrap();
        assert!(contract
            .runtime
            .validate(1_001, &contract.inputs[0], &contract.outputs[0])
            .unwrap_err()
            .to_string()
            .contains("at most 1000 classes"));

        for collapsed in ["5e-324", "0.99999999"] {
            let changed = text.replacen(
                "\"confidence_threshold\": 0.25",
                &format!("\"confidence_threshold\": {collapsed}"),
                1,
            );
            assert!(ModelContract::from_slice(changed.as_bytes())
                .unwrap_err()
                .to_string()
                .contains("binary32 probability endpoint"));
        }
    }

    #[test]
    fn parser_rejects_nonportable_artifact_paths() {
        let text = std::str::from_utf8(FIXTURE).unwrap();
        for invalid in [
            "../model.safetensors",
            "/model.safetensors",
            "mødel.safetensors",
        ] {
            let changed = text.replacen("model.safetensors", invalid, 1);
            assert!(ModelContract::from_slice(changed.as_bytes()).is_err());
        }

        let uppercase_suffix = text.replacen("model.safetensors", "model.SAFETENSORS", 1);
        assert!(ModelContract::from_slice(uppercase_suffix.as_bytes()).is_ok());
    }

    #[test]
    fn parser_rejects_unicode_format_characters_in_display_text() {
        let text = std::str::from_utf8(FIXTURE).unwrap();
        let deceptive = text.replacen(
            "Manwe cross-language schema fixture",
            "Manwe cross-language\u{200b} schema fixture",
            1,
        );

        let error = ModelContract::from_slice(deceptive.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("printable"));
    }

    #[test]
    fn package_loader_binds_contract_digest_to_exact_sibling_bytes() {
        let directory = TestDirectory::create();
        let artifact = b"authenticated model bytes";
        let digest = sha256_hex(artifact);
        let contract = std::str::from_utf8(FIXTURE)
            .unwrap()
            .replacen(&"0".repeat(64), &digest, 1);
        let contract_path = directory.0.join("model.contract.json");
        let artifact_path = directory.0.join("model.safetensors");
        std::fs::write(&artifact_path, artifact).unwrap();
        std::fs::write(&contract_path, contract).unwrap();

        let package = VerifiedModelPackage::load(&contract_path).unwrap();
        assert_eq!(package.artifact_bytes(), artifact);
        assert_eq!(package.contract().artifact_sha256(), digest);

        let error = VerifiedModelPackage::load_with_artifact_limit(&contract_path, 4).unwrap_err();
        assert!(error.to_string().contains("failed to read model artifact"));

        std::fs::write(&artifact_path, b"different model bytes").unwrap();
        let error = VerifiedModelPackage::load(&contract_path).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
    }
}
