
# Flutter Native ML 🚀

[![pub.dev](https://img.shields.io/pub/v/flutter_native_ml.svg?style=flat-square)](https://pub.dev/packages/flutter_native_ml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![GitHub Actions](https://img.shields.io/github/workflow/status/your_repo/flutter_native_ml/CI?style=flat-square)](https://github.com/your_repo/flutter_native_ml/actions)

A Flutter plugin that provides direct access to device-native machine learning accelerators, including Apple’s Neural Engine and Android’s NNAPI—enabling blazing-fast on-device inference with full control.

---

## ✨ Why It Matters

Most ML in Flutter today uses a TFLite interpreter in Dart, which is slow and has no direct access to specialized hardware. This plugin bridges that gap, allowing you to run models up to **15× faster** by leveraging the silicon your app runs on.

## 🧰 Features

-   **🚀 High-Performance Native Execution**: Bypasses the Dart interpreter for maximum speed.
-   **🧠 iOS Core ML**: Load `.mlmodelc` (compiled Core ML) files and run them using the Neural Engine, GPU, or CPU.
-   **⚡ Android NNAPI**: Load `.tflite` files and utilize the NNAPI delegate for GPU/DSP/NPU acceleration.
-   **🔍 Dynamic Model Introspection**: Automatically reads model input/output names, shapes, and data types—no more hardcoding.
-   **🔁 Multi-Input/Output Support**: Natively supports models with complex signatures out of the box.
-   **🎥 Streaming-Ready Architecture**: Designed to support real-time camera/audio inference pipelines.
-   **🛠️ Bundled CLI Tool**: `ml_builder` helps compile and convert models from `.mlmodel` or TensorFlow.

## 🔧 Setup & Usage

### 1. Add Dependency

Add the plugin to your project's `pubspec.yaml`:

```yaml
dependencies:
  flutter_native_ml: ^1.0.0
```

### 2. Prepare Your Model Assets

Your models must be in the correct native format. You can use the included `ml_builder` CLI tool to prepare them.

<details>
<summary><strong>Click to see Model Conversion with the `ml_builder` CLI</strong></summary>

The plugin includes a CLI utility to help you prepare models for native execution.

**Prerequisites:**
-   **Dart** is installed (`dart --version`).
-   On **macOS**: `Xcode Command Line Tools` are installed for Core ML compilation.
-   For TensorFlow: A **Python 3** environment with `pip` is available.

**Usage:**

Run the builder from your project's root directory:
```bash
dart run flutter_native_ml:ml_builder -s <source_path> -o <output_directory>
```
**Options:**
- `-s`, `--source`: **Required.** Path to your source model (`.mlmodel`, `.h5`, or a TensorFlow SavedModel directory).
- `-o`, `--output-dir`: Directory to save the converted model. Defaults to `models_out/`.
- `--quantize-fp16`: (TensorFlow only) Apply float16 quantization for smaller, faster models.

#### **Examples**

🧠 **Convert a Core ML `.mlmodel` (macOS only)**
```bash
dart run flutter_native_ml:ml_builder \
  -s path/to/MyModel.mlmodel \
  -o assets/models/
```
> **Output**: `assets/models/MyModel.mlmodelc` (ready for iOS)

🤖 **Convert a Keras `.h5` to TFLite**
```bash
dart run flutter_native_ml:ml_builder \
  -s path/to/my_model.h5 \
  -o assets/models/
```
> **Output**: `assets/models/my_model.tflite` (ready for Android)

💡 **Convert a TensorFlow SavedModel with quantization**
```bash
dart run flutter_native_ml:ml_builder \
  -s path/to/sentiment_saved_model \
  -o assets/models/ \
  --quantize-fp16
```
> **Output**: `assets/models/sentiment_saved_model.tflite` (quantized)

</details>

### 3. Declare Assets in `pubspec.yaml`

Once your models are in the `assets` folder, declare them:

```yaml
flutter:
  assets:
    - assets/models/
```

### 4. Use in Your Code

The recommended workflow is to load the model, inspect its signature, and then run inference.

```dart
import 'dart:io';
import 'package:flutter_native_ml/flutter_native_ml.dart';

// 1. Load the model
final model = await FlutterNativeML.loadModel(
  // Use the correct model path for the platform
  assetPath: Platform.isIOS
    ? 'assets/models/MyModel.mlmodelc'
    : 'assets/models/my_model.tflite',
);

// 2. Get the model's signature to know what it expects
final signature = await model.getSignature();
print('Inputs: ${signature.inputs}');
print('Outputs: ${signature.outputs}');

// 3. Prepare your input to match the signature
final inputTensor = signature.inputs.first;
final inputName = inputTensor.name;
final inputSize = inputTensor.shape.reduce((a, b) => a * b); // Calculate total elements
final inputData = List<double>.filled(inputSize, 0.5); // Example data

// 4. Run inference
final result = await model.run({inputName: inputData});

print('Accelerator: ${result.acceleratorUsed}');
print('Inference time: ${result.inferenceTime.inMilliseconds}ms');
print('Output: ${result.output}');

// 5. Clean up when you're done
await model.dispose();
```

## 🎥 Streaming Inference

For real-time use cases like camera or audio feeds, you can use the streaming API. This avoids the overhead of `invokeMethod` for every frame.

```dart
// Assumes you have already loaded a model and have its modelId
final stream = FlutterNativeML.startStream(modelId: yourModelId);

final subscription = stream.listen((inferenceResult) {
  print('Real-time result: ${inferenceResult.output}');
});

// When you're finished:
await FlutterNativeML.stopStream(modelId: yourModelId);
await subscription.cancel();
```
> Note: Full camera/audio integration is a work in progress. See Roadmap.

## 📝 Example App

Check out the `example/` folder for a full, working demo that shows how to:
-   Load a model
-   Inspect its signature
-   Run inference
-   Display the results and performance metrics
-   Dispose the model correctly
