# Flutter Native ML 🚀

[![pub.dev](https://img.shields.io/pub/v/flutter_native_ml.svg?style=flat-square)](https://pub.dev/packages/flutter_native_ml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

A Flutter plugin that gives your app direct access to the device's native machine
learning runtimes: **Core ML** on iOS (Apple Neural Engine, GPU, CPU) and
**LiteRT / TensorFlow Lite** on Android (GPU delegate, NNAPI, XNNPACK CPU).

---

## 📖 Table of Contents

- [Why It Matters](#-why-it-matters)
- [Features](#-features)
- [Requirements](#-requirements)
- [Setup and Usage](#-setup-and-usage)
  - [1. Add the dependency](#1-add-the-dependency)
  - [2. Prepare your model](#2-prepare-your-model)
  - [3. Declare the assets](#3-declare-the-assets)
  - [4. Load, inspect, run, dispose](#4-load-inspect-run-dispose)
  - [Choosing compute units](#choosing-compute-units)
  - [Input formats](#input-formats)
  - [Android build notes](#android-build-notes)
  - [Loading a model from the file system](#loading-a-model-from-the-file-system)
- [Zero-Copy Camera Input](#-zero-copy-camera-input)
- [Streaming Inference](#-streaming-inference)
- [Example App](#-example-app)
- [Testing](#-testing)
- [Author](#-author)

## ✨ Why It Matters

Running a TFLite interpreter from Dart keeps the hot path in the Dart VM and
cannot reach specialised silicon. This plugin loads the model in the platform's
own runtime, executes on the fastest available compute unit and only ships the
tensors across the platform channel, as typed lists.

## 🧰 Features

- **🚀 Native execution** on a dedicated worker thread per model; results are
  posted back to Flutter on the platform thread.
- **🧠 iOS Core ML**: load `.mlmodel` / `.mlpackage` assets (compiled on device
  and cached) or precompiled `.mlmodelc` bundles. Multi-array, image, string,
  int64, double, dictionary and sequence features are supported.
- **⚡ Android LiteRT**: load `.tflite` files with the GPU delegate, NNAPI or the
  multi-threaded XNNPACK CPU backend, with automatic fallback. All tensor
  types (float32, int8/uint8/int16/int32/int64, bool, string) are supported.
- **🔍 Model introspection**: names, shapes, data types, quantization
  parameters, optional inputs, SignatureDef aliases and metadata.
- **🔁 Multi-input / multi-output** models, dynamic shapes and typed outputs
  (`Float32List`, `Int32List`, ...).
- **📷 Zero-copy camera input**: the plugin runs the camera natively, resizes
  every frame straight into the model's input on the model thread and streams
  only the results to Dart, with a live preview texture.
- **🎥 Streaming**: push frames into a bounded native queue and receive results
  on a `Stream`; stale frames are dropped automatically.
- **🩺 Capabilities**: ask the device which accelerators it has before choosing
  a compute unit.
- **🛠️ Bundled CLI**: `dart run flutter_native_ml:ml_builder` converts Keras /
  SavedModel to `.tflite` and compiles Core ML models.

## 📋 Requirements

| Platform | Minimum                                       |
|----------|-----------------------------------------------|
| Flutter  | 3.24 (Dart 3.5)                               |
| Android  | API 23, AGP 8.6 – 9.x (built-in Kotlin supported), compileSdk 35+, Java 17 |
| iOS      | 13.0, CocoaPods or Swift Package Manager      |

## 🔧 Setup and Usage

### 1. Add the dependency

```yaml
dependencies:
  flutter_native_ml: ^1.1.0
```

### 2. Prepare your model

Models must be in the platform's native format:

| Platform | Format                                   |
|----------|------------------------------------------|
| Android  | `.tflite`                                |
| iOS      | `.mlmodel` or `.mlpackage` (recommended), or a compiled `.mlmodelc` |

> **Tip:** ship the single-file `.mlmodel` on iOS. Compiled `.mlmodelc` models
> are *directories*, which Flutter does not bundle as one asset. The plugin
> compiles `.mlmodel` files on first load and caches the result.

<details>
<summary><strong>Converting models with the <code>ml_builder</code> CLI</strong></summary>

**Prerequisites**

- Core ML compilation: macOS with Xcode command line tools.
- TensorFlow conversion: Python 3 with `tensorflow` installed (or pass
  `--install-tensorflow`).

**Usage** (from your app's root):

```bash
dart run flutter_native_ml:ml_builder -s <source> -o <output-dir> [--quantize-fp16]
```

| Option              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `-s`, `--source`    | `.mlmodel`, `.mlpackage`, `.h5`, `.keras` or a SavedModel directory.        |
| `-o`, `--output-dir`| Output directory (default `models_out/`).                                   |
| `--quantize-fp16`   | Float16 quantization for smaller, faster TFLite models.                     |
| `--python`          | Python interpreter to use (default `python3`).                              |
| `--install-tensorflow` | Install TensorFlow with pip when it is missing.                          |

Examples:

```bash
# Keras -> TFLite
dart run flutter_native_ml:ml_builder -s models/sentiment.h5 -o assets/models/

# SavedModel -> quantized TFLite
dart run flutter_native_ml:ml_builder -s models/sentiment_saved_model -o assets/models/ --quantize-fp16

# Core ML -> compiled .mlmodelc (macOS only)
dart run flutter_native_ml:ml_builder -s models/Sentiment.mlmodel -o build/models/
```

</details>

### 3. Declare the assets

```yaml
flutter:
  assets:
    - assets/models/
```

### 4. Load, inspect, run, dispose

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_native_ml/flutter_native_ml.dart';

// 1. Load the model (the signature is fetched at the same time).
final model = await FlutterNativeML.loadModel(
  assetPath: Platform.isIOS
      ? 'assets/models/Sentiment.mlmodel'
      : 'assets/models/sentiment.tflite',
  computeUnits: ComputeUnit.all, // GPU/ANE when available, CPU otherwise
);
print('Running on ${model.acceleratorUsed}');

// 2. Inspect the signature.
final signature = model.signature ?? await model.getSignature();
for (final input in signature.inputs) {
  print('$input'); // Tensor(name: input, shape: [1, 128], type: float32)
}

// 3. Build the input. Typed lists are the fastest way to send data.
final input = signature.inputs.first;
final data = Float32List(input.elementCount);

// 4. Run inference.
final result = await model.run({input.name: data});
print('Took ${result.inferenceTime.inMicroseconds} µs on ${result.acceleratorUsed}');
print('Best class: ${result.argmax(signature.outputs.first.name)}');
print(result.output); // {probs: Float32List(...)}

// 5. Release native resources.
await model.dispose();
```

Errors from the native side surface as `NativeMLException` with a stable `code`
(`MODEL_NOT_FOUND`, `SHAPE_MISMATCH`, `MISSING_INPUT`, `INFERENCE_FAILED`, ...).

### Choosing compute units

| `ComputeUnit`          | Android (LiteRT)                             | iOS (Core ML)               |
|------------------------|----------------------------------------------|-----------------------------|
| `all` (default)        | GPU delegate when supported, otherwise CPU   | CPU + GPU + Neural Engine   |
| `cpuOnly`              | XNNPACK, multi-threaded                      | CPU                         |
| `cpuAndGpu`            | GPU delegate, falls back to CPU              | CPU + GPU                   |
| `cpuAndNeuralEngine`   | NNAPI (API 27+), falls back to CPU           | CPU + Neural Engine         |

`loadModel` also accepts `numThreads` (Android CPU) and `allowFp16` (reduced
precision on GPU accelerators). Use `FlutterNativeML.getDeviceCapabilities()`
to see what the current device supports.

### Input formats

| Value                                        | Use for                                              |
|----------------------------------------------|------------------------------------------------------|
| `Float32List`, `Int32List`, `Uint8List`, ... | Numeric tensors (fastest)                            |
| `List<num>`, `List<bool>`, `List<String>`    | Numeric, boolean and string tensors                  |
| `TensorData(data, shape: [...])`             | Inputs with dynamic dimensions                       |
| `ImageInput(bytes, width:, height:, format:)`| Core ML image inputs (raw RGBA/BGRA/RGB/grayscale or PNG/JPEG via `ImageInput.encoded`) |
| `String`, `int`, `double`, `Map`             | Core ML scalar and dictionary features               |

On Android inputs can also be addressed by their SignatureDef alias
(e.g. `input_1` instead of `serving_default_input_1:0`).

### Android build notes

The plugin follows Flutter's built-in Kotlin guidance: it does not apply the
`org.jetbrains.kotlin.android` plugin when the Android Gradle Plugin provides
Kotlin support itself, or when Flutter's tooling has already applied it, and
it applies the plugin on its own only for older toolchains. Both of these
`gradle.properties` setups are supported:

```properties
# Flutter 3.47+ (AGP 9 built-in Kotlin, the AGP 9 default)
android.builtInKotlin=true
android.newDsl=false

# Flutter 3.35 – 3.46, or any app that has not migrated yet
android.builtInKotlin=false
android.newDsl=false
```

### Loading a model from the file system

```dart
final model = await FlutterNativeML.loadModel(filePath: '/path/to/downloaded/model.tflite');
```

## 📷 Zero-Copy Camera Input

For live camera use cases let the plugin own the camera. Frames never enter
Dart: CameraX (Android) / AVFoundation (iOS) hand each frame to the model's
worker thread, which resizes it into the input tensor and runs inference. Only
the results and a preview texture reach Flutter.

```dart
// Ask for permission once (iOS needs NSCameraUsageDescription in Info.plist).
if (!await FlutterNativeML.requestCameraPermission()) return;

final camera = await model.startCamera(
  lens: CameraLens.back,
  resolution: CameraResolution.medium,
  preprocessing: CameraPreprocessing.zeroToOne, // mean / std / resize mode
  maxFps: 15,                                   // optional throttle
);

// Live results, one per processed frame (stale frames are skipped).
final sub = camera.results.listen((result) {
  final best = result.argmax('probs');
  print('frame ${result.frameId} (${result.frame?.width}x${result.frame?.height}): '
        'class $best in ${result.inferenceTime.inMilliseconds} ms');
});

// Show the preview anywhere in your widget tree.
NativeCameraPreview(session: camera, fit: BoxFit.cover);

// Pause in the background, resume when the app returns, stop when done.
await camera.pause();
await camera.resume();
await camera.stop();
```

The model needs one image-like input: a Core ML image input, or a `uint8` /
`int8` / `float32` tensor shaped `[1, height, width, channels]` (LiteRT) or
`[1, channels, height, width]` (Core ML) with 1, 3 or 4 channels. Use
`inputName` for models with several inputs. `CameraPreprocessing` controls
the resize mode (`cover`, `fill`, `contain`) and per-channel mean / std
normalisation for float inputs (`zeroToOne`, `minusOneToOne`, `imagenet`
presets); integer inputs receive raw 0-255 pixels.

> Android apps that never use the camera can remove the permission the plugin
> declares with `<uses-permission android:name="android.permission.CAMERA" tools:node="remove" />`.

## 🎥 Streaming Inference

For camera or audio pipelines, keep a native queue busy instead of awaiting each
call:

```dart
final results = model.startStream(maxQueueSize: 2);
final subscription = results.listen((result) {
  print('frame ${result.frameId}: ${result.doubles('probs')} '
        '(${result.droppedFrames} dropped so far)');
});

// Feed frames as they arrive; the oldest queued frame is dropped when the
// queue is full so the stream never falls behind.
await model.pushStreamInput({'input': frameBytes});

// Later:
await model.stopStream();
await subscription.cancel();
```

## 📝 Example App

The `example/` app lets you load a model from the bundled assets or from a file
path, pick a compute unit, inspect the signature, run inference, exercise the
streaming API and run the zero-copy camera pipeline with a live preview. Drop a `model.tflite` / `model.mlmodel` into
`example/assets/models/` to try it.

## 🧪 Testing

```bash
flutter test                                   # Dart unit tests
cd example/android && ./gradlew :flutter_native_ml:testDebugUnitTest   # Kotlin unit tests
```

## 👤 Author

**Miracle Okolo** · [LinkedIn](https://www.linkedin.com/in/miracle-okolo-bb2133183/) · [GitHub](https://github.com/miracle101000)
