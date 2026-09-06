## 1.1.0

### Fixed
* **Android did not compile.** The Kotlin plugin imported `com.google.ai.edge.litert.*`
  classes that no Maven artifact provides. The plugin now depends on LiteRT
  (`com.google.ai.edge.litert:litert` / `litert-gpu`) and uses its
  `org.tensorflow.lite` Interpreter API.
* Android returned the tensor type under `dtype` while Dart expected `dataType`,
  which crashed `getSignature()`.
* Android ran model loading and inference on the UI thread.
* Inference with the GPU delegate now always happens on the thread that created
  the delegate (a LiteRT requirement); every model has its own worker thread.
* iOS mutated shared state from a concurrent queue and returned some results
  off the main thread.
* iOS ignored the model's declared input shape and always built 1-D arrays.
* iOS/Android accepted only `List<double>` inputs; integer lists, typed lists and
  mixed `num` lists now work.
* `startStream` / `stopStream` were not implemented on Android and only emitted
  timestamps on iOS.
* `dart run flutter_native_ml:ml_builder` did not work (the CLI lived in a
  separate package). It now ships as `bin/ml_builder.dart`.
* Manifest `package` attribute removed (deprecated with AGP 8).

### Added
* `ComputeUnit.cpuAndGpu` and `ComputeUnit.cpuAndNeuralEngine` (NNAPI on Android),
  with automatic CPU fallback. The compute unit actually used is reported in
  `NativeMLModel.acceleratorUsed` and `InferenceResult.acceleratorUsed`.
* `loadModel(filePath: ...)` for models stored on the device, `numThreads` and
  `allowFp16` options.
* iOS: `.mlmodel` / `.mlpackage` assets are compiled on device and cached, so
  a single-file asset is enough (compiled `.mlmodelc` bundles are directories
  that Flutter does not bundle as one asset).
* iOS: image inputs/outputs (`ImageInput`, raw or PNG/JPEG), string, int64,
  double, dictionary and sequence features; float16 arrays.
* Android: all LiteRT tensor types (float32, int8/uint8/int16/int32/int64,
  bool, string), quantization parameters in the signature, SignatureDef
  aliases, dynamic shapes via `TensorData(data, shape: [...])`, native
  inference timing.
* Streaming API: `model.startStream()`, `model.pushStreamInput()`,
  `model.stopStream()` with a bounded native queue that drops stale frames.
* `FlutterNativeML.getDeviceCapabilities()`, `getPlatformVersion()`,
  `disposeAll()`.
* `NativeMLException` with stable error codes instead of raw `PlatformException`s.
* Typed-list outputs (`Float32List`, `Int32List`, ...) plus
  `InferenceResult.doubles()`, `ints()`, `argmax()` and `outputShapes`.
* Swift Package Manager support alongside CocoaPods; privacy manifest bundled.
* Kotlin DSL build script compatible with Flutter's built-in Kotlin support.

### Changed
* Minimum iOS version is 13.0; Android plugin targets Java 17 (as Flutter requires).
* `ModelSignature`/`TensorInfo` gained `type`, `elementCount`, `hasDynamicShape`,
  `isOptional`, quantization fields and `metadata`.
* The static `FlutterNativeML.startStream`/`stopStream` helpers are deprecated in
  favour of the `NativeMLModel` methods.

## 1.0.1

* Fixed bugs on android.

## 1.0.0

* Initial release.
