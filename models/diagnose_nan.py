"""
Diagnostic script to identify NaN issues in the VGGNet model
"""

import numpy as np
from keras import models
import tensorflow as tf


def check_model_weights(model_path):
    """Check if model weights contain NaN values."""
    print("=" * 60)
    print("CHECKING MODEL WEIGHTS FOR NaN VALUES")
    print("=" * 60)

    model = models.load_model(model_path)

    nan_layers = []
    total_weights = 0
    total_nan = 0

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for i, w in enumerate(weights):
                num_weights = w.size
                num_nan = np.isnan(w).sum()
                total_weights += num_weights
                total_nan += num_nan

                if num_nan > 0:
                    nan_layers.append({
                        'layer': layer.name,
                        'weight_index': i,
                        'shape': w.shape,
                        'nan_count': num_nan,
                        'nan_percentage': 100 * num_nan / num_weights
                    })
                    print(
                        f"❌ {layer.name} weight[{i}]: {num_nan}/{num_weights} NaN values ({100 * num_nan / num_weights:.1f}%)")
                else:
                    print(f"✓ {layer.name} weight[{i}]: OK ({num_weights} weights)")

    print("-" * 60)
    if nan_layers:
        print(f"PROBLEM FOUND: {total_nan}/{total_weights} weights are NaN!")
        print("The model was saved with corrupted weights.")
        print("This typically means training diverged (loss went to NaN).")
    else:
        print(f"All {total_weights} weights are valid (no NaN).")

    return model, len(nan_layers) == 0


def check_inference_pipeline(model, sample_image=None):
    """Check if inference produces NaN."""
    print("\n" + "=" * 60)
    print("CHECKING INFERENCE PIPELINE")
    print("=" * 60)

    # Create a simple test input (random noise, normalized)
    if sample_image is None:
        test_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
    else:
        test_input = sample_image

    print(f"Test input shape: {test_input.shape}")
    print(f"Test input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"Test input has NaN: {np.isnan(test_input).any()}")

    # Run prediction
    output = model.predict(test_input, verbose=0)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output has NaN: {np.isnan(output).any()}")
    print(f"Output sum: {output.sum()}")  # Should be ~1.0 for softmax

    if np.isnan(output).any():
        print("\n❌ Model produces NaN outputs!")
        print("Checking intermediate layers...")
        check_intermediate_layers(model, test_input)
    else:
        print("\n✓ Model produces valid outputs")
        print(f"Sample predictions: {output[0][:5]}...")

    return output


def check_intermediate_layers(model, test_input):
    """Check which layer first produces NaN."""
    print("\n" + "-" * 60)
    print("LAYER-BY-LAYER ANALYSIS")
    print("-" * 60)

    # Create a model that outputs all intermediate layers
    layer_outputs = [layer.output for layer in model.layers]
    debug_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

    outputs = debug_model.predict(test_input, verbose=0)

    first_nan_layer = None
    for layer, output in zip(model.layers, outputs):
        has_nan = np.isnan(output).any()
        has_inf = np.isinf(output).any()

        status = "✓"
        if has_nan:
            status = "❌ NaN"
            if first_nan_layer is None:
                first_nan_layer = layer.name
        if has_inf:
            status = "⚠️ Inf"

        out_range = f"[{np.nanmin(output):.4f}, {np.nanmax(output):.4f}]"
        print(f"{status} {layer.name}: shape={output.shape}, range={out_range}")

    if first_nan_layer:
        print(f"\n⚠️ First NaN appears at layer: {first_nan_layer}")


def check_training_data_preprocessing():
    """Check if the preprocessing function exists and works."""
    print("\n" + "=" * 60)
    print("CHECKING PREPROCESSING IMPORT")
    print("=" * 60)

    try:
        from preprocessing.input_data_preprocessing import preprocess_and_fix_orientation
        print("✓ preprocess_and_fix_orientation imported successfully")

        # Test with dummy data
        dummy_image = tf.random.uniform((28, 28, 1), 0, 255, dtype=tf.float32)
        dummy_label = tf.constant(0)

        result = preprocess_and_fix_orientation(dummy_image, dummy_label)
        print(f"✓ Function works. Output image shape: {result[0].shape}")
        print(f"  Output range: [{float(tf.reduce_min(result[0])):.4f}, {float(tf.reduce_max(result[0])):.4f}]")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error testing preprocessing: {e}")


def main():
    model_path = 'saved/vggnet_model_preprocessed.keras'

    print("\n" + "=" * 60)
    print("VGGNet MODEL NaN DIAGNOSTIC")
    print("=" * 60)
    print(f"Model path: {model_path}\n")

    # Step 1: Check model weights
    model, weights_ok = check_model_weights(model_path)

    # Step 2: Check inference
    check_inference_pipeline(model)

    # Step 3: Check preprocessing
    check_training_data_preprocessing()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if not weights_ok:
        print("""
The model weights contain NaN values. This means training diverged.

SOLUTIONS:
1. Lower the learning rate (try 0.0001 instead of 0.001)
2. Add gradient clipping:
   optimizer = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
3. Check if preprocessing produces valid values
4. Add early stopping to catch divergence:
   callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
5. Monitor training loss - if it goes to NaN, stop and adjust
        """)
    else:
        print("""
Model weights are OK but predictions are NaN. Check:
1. Input preprocessing - make sure values are normalized correctly
2. Batch normalization layers may have issues with single samples
        """)


if __name__ == "__main__":
    main()