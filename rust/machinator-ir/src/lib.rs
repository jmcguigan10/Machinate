use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

const VALID_PARAM_STORE_FORMAT: &str = "safetensors";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    message: String,
}

impl ValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for ValidationError {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArchitectureSpec {
    pub model: ModelMetadata,
    pub input: InputSpec,
    pub target: TargetSpec,
    pub backbone: BackboneSpec,
    pub head: HeadSpec,
    pub procedures: ProcedureSpec,
    pub param_store: ParamStoreSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelMetadata {
    pub name: String,
    pub family: String,
    pub task: String,
    pub modality: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputSpec {
    pub kind: String,
    #[serde(default)]
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub feature_count: usize,
    #[serde(default)]
    pub token_vocab_size: Option<usize>,
    #[serde(default)]
    pub max_sequence_length: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetSpec {
    pub column: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackboneSpec {
    #[serde(default)]
    pub hidden_dims: Vec<usize>,
    #[serde(default)]
    pub model_dim: Option<usize>,
    #[serde(default)]
    pub num_heads: Option<usize>,
    #[serde(default)]
    pub num_layers: Option<usize>,
    #[serde(default)]
    pub ffn_dim: Option<usize>,
    pub activation: String,
    pub normalization: String,
    pub dropout: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HeadSpec {
    pub output_dim: usize,
    #[serde(default)]
    pub pooling: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcedureSpec {
    pub forward: ForwardProcedure,
    pub loss: LossProcedure,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForwardProcedure {
    pub input: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LossProcedure {
    pub kind: String,
    pub prediction: String,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParamStoreSpec {
    pub format: String,
    pub root_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParamStoreManifest {
    pub schema_version: u32,
    pub format: String,
    pub root_key: String,
    pub parameter_count: usize,
    pub parameters: Vec<ParameterBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParameterBinding {
    pub tensor_key: String,
    pub owner_id: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiffChange {
    pub field: String,
    pub old_value: serde_json::Value,
    pub new_value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MigrationAction {
    pub action: String,
    pub target_tensor_key: String,
    pub source_tensor_key: Option<String>,
    pub source_shape: Option<Vec<usize>>,
    pub target_shape: Vec<usize>,
    pub overlap_shape: Option<Vec<usize>>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MigrationPlan {
    pub exact_copy_count: usize,
    pub partial_copy_count: usize,
    pub reinitialize_count: usize,
    pub actions: Vec<MigrationAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelDiff {
    pub compatible: bool,
    pub old_family: String,
    pub new_family: String,
    pub parameter_delta: i64,
    pub changes: Vec<DiffChange>,
    pub migration_plan: MigrationPlan,
}

impl ArchitectureSpec {
    pub fn from_toml_str(input: &str) -> Result<Self, Box<dyn Error>> {
        let spec: ArchitectureSpec = toml::from_str(input)?;
        Ok(spec)
    }

    pub fn from_toml_file(path: &Path) -> Result<Self, Box<dyn Error>> {
        let input = fs::read_to_string(path)?;
        Self::from_toml_str(&input)
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        if !matches!(self.model.family.as_str(), "tabular_mlp" | "transformer_encoder") {
            return Err(ValidationError::new(format!(
                "unsupported model family `{}`",
                self.model.family
            )));
        }
        if !matches!(self.model.modality.as_str(), "tabular" | "text") {
            return Err(ValidationError::new(format!(
                "unsupported modality `{}`",
                self.model.modality
            )));
        }
        if !matches!(self.input.kind.as_str(), "dense_features" | "token_ids") {
            return Err(ValidationError::new(format!(
                "unsupported input kind `{}`",
                self.input.kind
            )));
        }
        if self.model.task != "binary_classification" {
            return Err(ValidationError::new(format!(
                "unsupported task `{}`",
                self.model.task
            )));
        }
        if self.target.kind != "binary" {
            return Err(ValidationError::new(format!(
                "unsupported target kind `{}`",
                self.target.kind
            )));
        }
        if self.target.column.trim().is_empty() {
            return Err(ValidationError::new("target.column is required"));
        }
        if !matches!(
            self.backbone.activation.as_str(),
            "relu" | "gelu" | "silu" | "tanh"
        ) {
            return Err(ValidationError::new(format!(
                "unsupported activation `{}`",
                self.backbone.activation
            )));
        }
        if !matches!(
            self.backbone.normalization.as_str(),
            "none" | "batchnorm" | "layernorm"
        ) {
            return Err(ValidationError::new(format!(
                "unsupported normalization `{}`",
                self.backbone.normalization
            )));
        }
        if !(0.0..1.0).contains(&self.backbone.dropout) {
            return Err(ValidationError::new(
                "backbone.dropout must be in the range [0.0, 1.0)",
            ));
        }
        if self.head.output_dim != 1 {
            return Err(ValidationError::new(
                "head.output_dim must be 1 for binary classification",
            ));
        }
        if self.procedures.forward.input != "features" {
            return Err(ValidationError::new(
                "procedures.forward.input must be `features`",
            ));
        }
        if self.procedures.forward.output != "logits" {
            return Err(ValidationError::new(
                "procedures.forward.output must be `logits`",
            ));
        }
        if self.procedures.loss.kind != "bce_with_logits" {
            return Err(ValidationError::new(format!(
                "unsupported loss `{}`",
                self.procedures.loss.kind
            )));
        }
        if self.procedures.loss.prediction != "logits" {
            return Err(ValidationError::new(
                "procedures.loss.prediction must be `logits`",
            ));
        }
        if self.procedures.loss.target != "target" {
            return Err(ValidationError::new(
                "procedures.loss.target must be `target`",
            ));
        }
        if self.param_store.format != VALID_PARAM_STORE_FORMAT {
            return Err(ValidationError::new(
                "param_store.format must be `safetensors`",
            ));
        }
        if self.param_store.root_key.trim().is_empty() {
            return Err(ValidationError::new("param_store.root_key is required"));
        }

        match self.model.family.as_str() {
            "tabular_mlp" => {
                if self.model.modality != "tabular" {
                    return Err(ValidationError::new(
                        "tabular_mlp requires modality `tabular`",
                    ));
                }
                if self.input.kind != "dense_features" {
                    return Err(ValidationError::new(
                        "tabular_mlp requires input.kind `dense_features`",
                    ));
                }
                if self.input.feature_count == 0 {
                    return Err(ValidationError::new(
                        "input.feature_count must be positive for tabular_mlp",
                    ));
                }
                if !self.input.feature_names.is_empty()
                    && self.input.feature_names.len() != self.input.feature_count
                {
                    return Err(ValidationError::new(
                        "input.feature_names must match input.feature_count",
                    ));
                }
                if self.backbone.hidden_dims.is_empty()
                    || self.backbone.hidden_dims.iter().any(|value| *value == 0)
                {
                    return Err(ValidationError::new(
                        "backbone.hidden_dims must contain one or more positive integers",
                    ));
                }
            }
            "transformer_encoder" => {
                if self.model.modality != "text" {
                    return Err(ValidationError::new(
                        "transformer_encoder requires modality `text`",
                    ));
                }
                if self.input.kind != "token_ids" {
                    return Err(ValidationError::new(
                        "transformer_encoder requires input.kind `token_ids`",
                    ));
                }
                if self.input.token_vocab_size.unwrap_or(0) == 0 {
                    return Err(ValidationError::new(
                        "input.token_vocab_size must be positive for transformer_encoder",
                    ));
                }
                if self.input.max_sequence_length.unwrap_or(0) == 0 {
                    return Err(ValidationError::new(
                        "input.max_sequence_length must be positive for transformer_encoder",
                    ));
                }
                if self.backbone.model_dim.unwrap_or(0) == 0 {
                    return Err(ValidationError::new(
                        "backbone.model_dim must be positive for transformer_encoder",
                    ));
                }
                if self.backbone.num_heads.unwrap_or(0) == 0 {
                    return Err(ValidationError::new(
                        "backbone.num_heads must be positive for transformer_encoder",
                    ));
                }
                if self.backbone.num_layers.unwrap_or(0) == 0 {
                    return Err(ValidationError::new(
                        "backbone.num_layers must be positive for transformer_encoder",
                    ));
                }
                if self.backbone.ffn_dim.unwrap_or(0) == 0 {
                    return Err(ValidationError::new(
                        "backbone.ffn_dim must be positive for transformer_encoder",
                    ));
                }
                let model_dim = self.backbone.model_dim.unwrap();
                let num_heads = self.backbone.num_heads.unwrap();
                if model_dim % num_heads != 0 {
                    return Err(ValidationError::new(
                        "backbone.model_dim must be divisible by backbone.num_heads",
                    ));
                }
                if !matches!(self.head.pooling.as_deref().unwrap_or("mean"), "mean" | "cls") {
                    return Err(ValidationError::new(format!(
                        "unsupported pooling `{}` for transformer_encoder",
                        self.head.pooling.clone().unwrap_or_else(|| "mean".to_string())
                    )));
                }
            }
            _ => {}
        }

        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        match self.model.family.as_str() {
            "tabular_mlp" => {
                let mut total = 0_usize;
                let mut input_dim = self.input.feature_count;
                for hidden_dim in &self.backbone.hidden_dims {
                    total += input_dim * hidden_dim;
                    total += hidden_dim;
                    if self.backbone.normalization != "none" {
                        total += hidden_dim * 2;
                    }
                    input_dim = *hidden_dim;
                }
                total += input_dim * self.head.output_dim;
                total += self.head.output_dim;
                total
            }
            "transformer_encoder" => {
                let token_vocab_size = self.input.token_vocab_size.unwrap_or(0);
                let max_sequence_length = self.input.max_sequence_length.unwrap_or(0);
                let model_dim = self.backbone.model_dim.unwrap_or(0);
                let ffn_dim = self.backbone.ffn_dim.unwrap_or(0);
                let num_layers = self.backbone.num_layers.unwrap_or(0);

                let mut total = token_vocab_size * model_dim;
                total += max_sequence_length * model_dim;
                for _ in 0..num_layers {
                    total += 3 * model_dim * model_dim;
                    total += 3 * model_dim;
                    total += model_dim * model_dim;
                    total += model_dim;
                    total += model_dim * ffn_dim;
                    total += ffn_dim;
                    total += ffn_dim * model_dim;
                    total += model_dim;
                    total += model_dim * 4;
                }
                total += model_dim * self.head.output_dim;
                total += self.head.output_dim;
                total
            }
            _ => 0,
        }
    }

    pub fn param_store_manifest(&self) -> ParamStoreManifest {
        let mut parameters = Vec::new();

        match self.model.family.as_str() {
            "tabular_mlp" => {
                let mut input_dim = self.input.feature_count;
                for (layer_index, hidden_dim) in self.backbone.hidden_dims.iter().enumerate() {
                    parameters.push(ParameterBinding {
                        tensor_key: format!("layers.{}.weight", layer_index),
                        owner_id: format!("backbone.layer.{}", layer_index),
                        shape: vec![*hidden_dim, input_dim],
                        dtype: "float32".to_string(),
                    });
                    parameters.push(ParameterBinding {
                        tensor_key: format!("layers.{}.bias", layer_index),
                        owner_id: format!("backbone.layer.{}", layer_index),
                        shape: vec![*hidden_dim],
                        dtype: "float32".to_string(),
                    });
                    if self.backbone.normalization != "none" {
                        parameters.push(ParameterBinding {
                            tensor_key: format!("norms.{}.weight", layer_index),
                            owner_id: format!("backbone.norm.{}", layer_index),
                            shape: vec![*hidden_dim],
                            dtype: "float32".to_string(),
                        });
                        parameters.push(ParameterBinding {
                            tensor_key: format!("norms.{}.bias", layer_index),
                            owner_id: format!("backbone.norm.{}", layer_index),
                            shape: vec![*hidden_dim],
                            dtype: "float32".to_string(),
                        });
                    }
                    input_dim = *hidden_dim;
                }
                parameters.push(ParameterBinding {
                    tensor_key: "head.weight".to_string(),
                    owner_id: "head".to_string(),
                    shape: vec![self.head.output_dim, input_dim],
                    dtype: "float32".to_string(),
                });
                parameters.push(ParameterBinding {
                    tensor_key: "head.bias".to_string(),
                    owner_id: "head".to_string(),
                    shape: vec![self.head.output_dim],
                    dtype: "float32".to_string(),
                });
            }
            "transformer_encoder" => {
                let token_vocab_size = self.input.token_vocab_size.unwrap_or(0);
                let max_sequence_length = self.input.max_sequence_length.unwrap_or(0);
                let model_dim = self.backbone.model_dim.unwrap_or(0);
                let ffn_dim = self.backbone.ffn_dim.unwrap_or(0);
                let num_layers = self.backbone.num_layers.unwrap_or(0);

                parameters.push(ParameterBinding {
                    tensor_key: "token_embedding.weight".to_string(),
                    owner_id: "token_embedding".to_string(),
                    shape: vec![token_vocab_size, model_dim],
                    dtype: "float32".to_string(),
                });
                parameters.push(ParameterBinding {
                    tensor_key: "position_embedding.weight".to_string(),
                    owner_id: "position_embedding".to_string(),
                    shape: vec![max_sequence_length, model_dim],
                    dtype: "float32".to_string(),
                });
                for layer_index in 0..num_layers {
                    let prefix = format!("encoder.layers.{}", layer_index);
                    parameters.extend([
                        ParameterBinding {
                            tensor_key: format!("{}.self_attn.in_proj_weight", prefix),
                            owner_id: format!("encoder.layer.{}.self_attn", layer_index),
                            shape: vec![model_dim * 3, model_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.self_attn.in_proj_bias", prefix),
                            owner_id: format!("encoder.layer.{}.self_attn", layer_index),
                            shape: vec![model_dim * 3],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.self_attn.out_proj.weight", prefix),
                            owner_id: format!("encoder.layer.{}.self_attn.out_proj", layer_index),
                            shape: vec![model_dim, model_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.self_attn.out_proj.bias", prefix),
                            owner_id: format!("encoder.layer.{}.self_attn.out_proj", layer_index),
                            shape: vec![model_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.linear1.weight", prefix),
                            owner_id: format!("encoder.layer.{}.linear1", layer_index),
                            shape: vec![ffn_dim, model_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.linear1.bias", prefix),
                            owner_id: format!("encoder.layer.{}.linear1", layer_index),
                            shape: vec![ffn_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.linear2.weight", prefix),
                            owner_id: format!("encoder.layer.{}.linear2", layer_index),
                            shape: vec![model_dim, ffn_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.linear2.bias", prefix),
                            owner_id: format!("encoder.layer.{}.linear2", layer_index),
                            shape: vec![model_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.norm1.weight", prefix),
                            owner_id: format!("encoder.layer.{}.norm1", layer_index),
                            shape: vec![model_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.norm1.bias", prefix),
                            owner_id: format!("encoder.layer.{}.norm1", layer_index),
                            shape: vec![model_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.norm2.weight", prefix),
                            owner_id: format!("encoder.layer.{}.norm2", layer_index),
                            shape: vec![model_dim],
                            dtype: "float32".to_string(),
                        },
                        ParameterBinding {
                            tensor_key: format!("{}.norm2.bias", prefix),
                            owner_id: format!("encoder.layer.{}.norm2", layer_index),
                            shape: vec![model_dim],
                            dtype: "float32".to_string(),
                        },
                    ]);
                }
                parameters.push(ParameterBinding {
                    tensor_key: "head.weight".to_string(),
                    owner_id: "head".to_string(),
                    shape: vec![self.head.output_dim, model_dim],
                    dtype: "float32".to_string(),
                });
                parameters.push(ParameterBinding {
                    tensor_key: "head.bias".to_string(),
                    owner_id: "head".to_string(),
                    shape: vec![self.head.output_dim],
                    dtype: "float32".to_string(),
                });
            }
            _ => {}
        }

        ParamStoreManifest {
            schema_version: 1,
            format: self.param_store.format.clone(),
            root_key: self.param_store.root_key.clone(),
            parameter_count: self.parameter_count(),
            parameters,
        }
    }

    pub fn diff(&self, other: &Self) -> ModelDiff {
        let mut changes = Vec::new();
        track_change(
            &mut changes,
            "family",
            json!(self.model.family),
            json!(other.model.family),
        );
        track_change(
            &mut changes,
            "task",
            json!(self.model.task),
            json!(other.model.task),
        );
        track_change(
            &mut changes,
            "modality",
            json!(self.model.modality),
            json!(other.model.modality),
        );
        track_change(
            &mut changes,
            "input_kind",
            json!(self.input.kind),
            json!(other.input.kind),
        );
        track_change(
            &mut changes,
            "feature_count",
            json!(self.input.feature_count),
            json!(other.input.feature_count),
        );
        track_change(
            &mut changes,
            "feature_names",
            json!(self.input.feature_names),
            json!(other.input.feature_names),
        );
        track_change(
            &mut changes,
            "token_vocab_size",
            json!(self.input.token_vocab_size),
            json!(other.input.token_vocab_size),
        );
        track_change(
            &mut changes,
            "max_sequence_length",
            json!(self.input.max_sequence_length),
            json!(other.input.max_sequence_length),
        );
        track_change(
            &mut changes,
            "target_column",
            json!(self.target.column),
            json!(other.target.column),
        );
        track_change(
            &mut changes,
            "hidden_dims",
            json!(self.backbone.hidden_dims),
            json!(other.backbone.hidden_dims),
        );
        track_change(
            &mut changes,
            "model_dim",
            json!(self.backbone.model_dim),
            json!(other.backbone.model_dim),
        );
        track_change(
            &mut changes,
            "num_heads",
            json!(self.backbone.num_heads),
            json!(other.backbone.num_heads),
        );
        track_change(
            &mut changes,
            "num_layers",
            json!(self.backbone.num_layers),
            json!(other.backbone.num_layers),
        );
        track_change(
            &mut changes,
            "ffn_dim",
            json!(self.backbone.ffn_dim),
            json!(other.backbone.ffn_dim),
        );
        track_change(
            &mut changes,
            "activation",
            json!(self.backbone.activation),
            json!(other.backbone.activation),
        );
        track_change(
            &mut changes,
            "normalization",
            json!(self.backbone.normalization),
            json!(other.backbone.normalization),
        );
        track_change(
            &mut changes,
            "dropout",
            json!(self.backbone.dropout),
            json!(other.backbone.dropout),
        );
        track_change(
            &mut changes,
            "pooling",
            json!(self.head.pooling),
            json!(other.head.pooling),
        );

        ModelDiff {
            compatible: self.model.task == other.model.task
                && self.model.modality == other.model.modality
                && self.target.kind == other.target.kind,
            old_family: self.model.family.clone(),
            new_family: other.model.family.clone(),
            parameter_delta: other.parameter_count() as i64 - self.parameter_count() as i64,
            changes,
            migration_plan: build_migration_plan(self, other),
        }
    }
}

fn track_change(
    changes: &mut Vec<DiffChange>,
    field: &str,
    old_value: serde_json::Value,
    new_value: serde_json::Value,
) {
    if old_value != new_value {
        changes.push(DiffChange {
            field: field.to_string(),
            old_value,
            new_value,
        });
    }
}

pub fn build_migration_plan(old_spec: &ArchitectureSpec, new_spec: &ArchitectureSpec) -> MigrationPlan {
    let old_manifest = old_spec.param_store_manifest();
    let new_manifest = new_spec.param_store_manifest();
    let old_map = old_manifest
        .parameters
        .iter()
        .map(|item| (item.tensor_key.as_str(), item))
        .collect::<std::collections::HashMap<_, _>>();

    let mut exact_copy_count = 0_usize;
    let mut partial_copy_count = 0_usize;
    let mut reinitialize_count = 0_usize;
    let mut actions = Vec::new();

    for new_param in &new_manifest.parameters {
        let Some(old_param) = old_map.get(new_param.tensor_key.as_str()) else {
            reinitialize_count += 1;
            actions.push(MigrationAction {
                action: "reinitialize".to_string(),
                target_tensor_key: new_param.tensor_key.clone(),
                source_tensor_key: None,
                source_shape: None,
                target_shape: new_param.shape.clone(),
                overlap_shape: None,
                reason: "tensor key is new in the updated spec".to_string(),
            });
            continue;
        };

        if old_param.shape == new_param.shape {
            exact_copy_count += 1;
            actions.push(MigrationAction {
                action: "exact_copy".to_string(),
                target_tensor_key: new_param.tensor_key.clone(),
                source_tensor_key: Some(old_param.tensor_key.clone()),
                source_shape: Some(old_param.shape.clone()),
                target_shape: new_param.shape.clone(),
                overlap_shape: Some(new_param.shape.clone()),
                reason: "matching tensor key and shape".to_string(),
            });
            continue;
        }

        if old_param.shape.len() == new_param.shape.len() {
            let overlap_shape = old_param
                .shape
                .iter()
                .zip(new_param.shape.iter())
                .map(|(old_dim, new_dim)| usize::min(*old_dim, *new_dim))
                .collect::<Vec<_>>();
            if overlap_shape.iter().all(|value| *value > 0) {
                partial_copy_count += 1;
                actions.push(MigrationAction {
                    action: "partial_copy".to_string(),
                    target_tensor_key: new_param.tensor_key.clone(),
                    source_tensor_key: Some(old_param.tensor_key.clone()),
                    source_shape: Some(old_param.shape.clone()),
                    target_shape: new_param.shape.clone(),
                    overlap_shape: Some(overlap_shape),
                    reason: "tensor key matches but shape changed; preserve the overlapping slice"
                        .to_string(),
                });
                continue;
            }
        }

        reinitialize_count += 1;
        actions.push(MigrationAction {
            action: "reinitialize".to_string(),
            target_tensor_key: new_param.tensor_key.clone(),
            source_tensor_key: Some(old_param.tensor_key.clone()),
            source_shape: Some(old_param.shape.clone()),
            target_shape: new_param.shape.clone(),
            overlap_shape: None,
            reason: "tensor rank changed or there is no meaningful overlap to preserve".to_string(),
        });
    }

    MigrationPlan {
        exact_copy_count,
        partial_copy_count,
        reinitialize_count,
        actions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABULAR_SPEC: &str = r#"
[model]
name = "demo"
family = "tabular_mlp"
task = "binary_classification"
modality = "tabular"

[input]
kind = "dense_features"
feature_names = ["a", "b"]
feature_count = 2

[target]
column = "label"
kind = "binary"

[backbone]
hidden_dims = [128, 64]
activation = "relu"
normalization = "layernorm"
dropout = 0.1

[head]
output_dim = 1

[procedures.forward]
input = "features"
output = "logits"

[procedures.loss]
kind = "bce_with_logits"
prediction = "logits"
target = "target"

[param_store]
format = "safetensors"
root_key = "demo_model"
"#;

    const TRANSFORMER_SPEC: &str = r#"
[model]
name = "demo-text"
family = "transformer_encoder"
task = "binary_classification"
modality = "text"

[input]
kind = "token_ids"
token_vocab_size = 32000
max_sequence_length = 256

[target]
column = "label"
kind = "binary"

[backbone]
model_dim = 128
num_heads = 4
num_layers = 2
ffn_dim = 256
activation = "gelu"
normalization = "layernorm"
dropout = 0.1

[head]
output_dim = 1
pooling = "mean"

[procedures.forward]
input = "features"
output = "logits"

[procedures.loss]
kind = "bce_with_logits"
prediction = "logits"
target = "target"

[param_store]
format = "safetensors"
root_key = "demo_text_model"
"#;

    #[test]
    fn valid_tabular_spec_parses_and_validates() {
        let spec = ArchitectureSpec::from_toml_str(TABULAR_SPEC).unwrap();
        spec.validate().unwrap();
        assert_eq!(spec.parameter_count(), 9089);
        let manifest = spec.param_store_manifest();
        assert_eq!(manifest.parameters[0].tensor_key, "layers.0.weight");
        assert_eq!(manifest.parameters.last().unwrap().tensor_key, "head.bias");
    }

    #[test]
    fn valid_transformer_spec_parses_and_validates() {
        let spec = ArchitectureSpec::from_toml_str(TRANSFORMER_SPEC).unwrap();
        spec.validate().unwrap();
        assert!(spec.parameter_count() > 0);
        let manifest = spec.param_store_manifest();
        assert_eq!(manifest.parameters[0].tensor_key, "token_embedding.weight");
    }

    #[test]
    fn invalid_head_is_rejected() {
        let invalid = TABULAR_SPEC.replace("output_dim = 1", "output_dim = 2");
        let spec = ArchitectureSpec::from_toml_str(&invalid).unwrap();
        let error = spec.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "head.output_dim must be 1 for binary classification"
        );
    }

    #[test]
    fn migration_plan_detects_partial_copy() {
        let old_spec = ArchitectureSpec::from_toml_str(TABULAR_SPEC).unwrap();
        let new_spec = ArchitectureSpec::from_toml_str(
            &TABULAR_SPEC.replace("[128, 64]", "[256, 64]"),
        )
        .unwrap();
        let plan = build_migration_plan(&old_spec, &new_spec);
        assert!(plan.partial_copy_count > 0);
        assert!(plan
            .actions
            .iter()
            .any(|item| item.target_tensor_key == "layers.0.weight" && item.action == "partial_copy"));
    }
}
