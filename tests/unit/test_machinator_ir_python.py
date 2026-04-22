from __future__ import annotations

import unittest

from machinator.machinator_ir_python import ArchitectureSpec, ValidationError, build_migration_plan


TABULAR_SPEC = """
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
"""


TRANSFORMER_SPEC = """
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
"""


VISION_CNN_SPEC = """
[model]
name = "demo-vision"
family = "vision_cnn"
task = "binary_classification"
modality = "vision"

[input]
kind = "image_tensor"
image_channels = 1
image_height = 28
image_width = 28

[target]
column = "label"
kind = "binary"

[backbone]
channels = [32, 64, 128]
activation = "relu"
normalization = "batchnorm"
dropout = 0.1

[head]
output_dim = 1
pooling = "avg"

[procedures.forward]
input = "features"
output = "logits"

[procedures.loss]
kind = "bce_with_logits"
prediction = "logits"
target = "target"

[param_store]
format = "safetensors"
root_key = "demo_vision_model"
"""


VISION_RESNET_SPEC = """
[model]
name = "demo-resnet"
family = "vision_resnet"
task = "binary_classification"
modality = "vision"

[input]
kind = "image_tensor"
image_channels = 1
image_height = 28
image_width = 28

[target]
column = "label"
kind = "binary"

[backbone]
channels = [32, 64, 128]
num_layers = 2
activation = "relu"
normalization = "batchnorm"
dropout = 0.1

[head]
output_dim = 1
pooling = "avg"

[procedures.forward]
input = "features"
output = "logits"

[procedures.loss]
kind = "bce_with_logits"
prediction = "logits"
target = "target"

[param_store]
format = "safetensors"
root_key = "demo_resnet_model"
"""


class MachinatorIrPythonTests(unittest.TestCase):
    def test_valid_tabular_spec_parses_and_validates(self) -> None:
        spec = ArchitectureSpec.from_toml_str(TABULAR_SPEC)
        spec.validate()
        self.assertEqual(spec.parameter_count(), 9089)
        manifest = spec.param_store_manifest()
        self.assertEqual(manifest.parameters[0].tensor_key, "layers.0.weight")
        self.assertEqual(manifest.parameters[-1].tensor_key, "head.bias")

    def test_valid_transformer_spec_parses_and_validates(self) -> None:
        spec = ArchitectureSpec.from_toml_str(TRANSFORMER_SPEC)
        spec.validate()
        self.assertGreater(spec.parameter_count(), 0)
        manifest = spec.param_store_manifest()
        self.assertEqual(manifest.parameters[0].tensor_key, "token_embedding.weight")

    def test_valid_vision_specs_parse_and_validate(self) -> None:
        cnn_spec = ArchitectureSpec.from_toml_str(VISION_CNN_SPEC)
        cnn_spec.validate()
        self.assertGreater(cnn_spec.parameter_count(), 0)
        self.assertEqual(cnn_spec.param_store_manifest().parameters[0].tensor_key, "convs.0.weight")

        resnet_spec = ArchitectureSpec.from_toml_str(VISION_RESNET_SPEC)
        resnet_spec.validate()
        self.assertGreater(resnet_spec.parameter_count(), cnn_spec.parameter_count())
        self.assertEqual(resnet_spec.param_store_manifest().parameters[0].tensor_key, "stem.weight")

    def test_invalid_head_is_rejected(self) -> None:
        invalid_spec = ArchitectureSpec.from_toml_str(TABULAR_SPEC.replace("output_dim = 1", "output_dim = 2"))
        with self.assertRaisesRegex(
            ValidationError,
            "head.output_dim must be 1 for binary classification",
        ):
            invalid_spec.validate()

    def test_migration_plan_detects_partial_copy(self) -> None:
        old_spec = ArchitectureSpec.from_toml_str(TABULAR_SPEC)
        new_spec = ArchitectureSpec.from_toml_str(TABULAR_SPEC.replace("[128, 64]", "[256, 64]"))
        plan = build_migration_plan(old_spec, new_spec)
        self.assertGreater(plan.partial_copy_count, 0)
        self.assertTrue(
            any(
                item.target_tensor_key == "layers.0.weight" and item.action == "partial_copy"
                for item in plan.actions
            )
        )


if __name__ == "__main__":
    unittest.main()
