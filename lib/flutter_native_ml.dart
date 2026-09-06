/// Direct access to on-device ML accelerators: Core ML (Apple Neural Engine,
/// GPU) on iOS and LiteRT / TensorFlow Lite (GPU delegate, NNAPI, XNNPACK) on
/// Android.
///
/// ```dart
/// final model = await FlutterNativeML.loadModel(
///   assetPath: Platform.isIOS
///       ? 'assets/models/MyModel.mlmodel'
///       : 'assets/models/my_model.tflite',
/// );
/// final input = model.signature!.inputs.first;
/// final result = await model.run({
///   input.name: Float32List(input.elementCount),
/// });
/// print(result.output);
/// await model.dispose();
/// ```
library;

import 'dart:async';

import 'package:flutter/services.dart';
import 'package:flutter_native_ml/src/camera.dart';
import 'package:flutter_native_ml/src/exceptions.dart';
import 'package:flutter_native_ml/src/models.dart';
import 'package:flutter_native_ml/src/native_ml_model.dart';

export 'package:flutter_native_ml/src/camera.dart'
    show CameraLens, CameraResolution, ResizeMode, CameraPermissionStatus, CameraPreprocessing, NativeCameraSession;
export 'package:flutter_native_ml/src/camera_preview.dart';
export 'package:flutter_native_ml/src/exceptions.dart';
export 'package:flutter_native_ml/src/models.dart';
export 'package:flutter_native_ml/src/native_ml_model.dart' show NativeMLModel;

/// Entry point of the plugin.
class FlutterNativeML {
  FlutterNativeML._();

  static const MethodChannel _methodChannel = MethodChannel('flutter_native_ml');

  /// Models created by the deprecated static stream helpers, keyed by id.
  static final Map<String, NativeMLModel> _legacyStreamModels = {};

  /// Loads a model and returns a handle to it.
  ///
  /// Exactly one of [assetPath] (a Flutter asset declared in `pubspec.yaml`)
  /// or [filePath] (an absolute path on the device, e.g. a downloaded model)
  /// must be given.
  ///
  /// Supported formats:
  /// * Android: `.tflite` (LiteRT / TensorFlow Lite flatbuffers).
  /// * iOS: `.mlmodel` / `.mlpackage` (compiled on device and cached) or an
  ///   already compiled `.mlmodelc` bundle.
  ///
  /// [computeUnits] selects the hardware (see [ComputeUnit]); the plugin falls
  /// back to the CPU when the requested accelerator is unavailable and reports
  /// what was actually used in [NativeMLModel.acceleratorUsed].
  /// [numThreads] limits CPU threads on Android. [allowFp16] permits reduced
  /// precision on GPU accelerators for extra speed.
  static Future<NativeMLModel> loadModel({
    String? assetPath,
    String? filePath,
    ComputeUnit computeUnits = ComputeUnit.all,
    int? numThreads,
    bool allowFp16 = false,
  }) async {
    final hasAsset = assetPath != null && assetPath.isNotEmpty;
    final hasFile = filePath != null && filePath.isNotEmpty;
    if (!hasAsset && !hasFile) {
      throw ArgumentError('Provide either assetPath or filePath.');
    }
    if (hasAsset && hasFile) {
      throw ArgumentError('Provide only one of assetPath or filePath.');
    }
    if (numThreads != null && numThreads < 1) {
      throw ArgumentError.value(numThreads, 'numThreads', 'must be at least 1');
    }

    final result = await invokeNative<dynamic>(_methodChannel, 'loadModel', {
      if (hasAsset) 'assetPath': assetPath,
      if (hasFile) 'filePath': filePath,
      'computeUnits': computeUnits.name,
      if (numThreads != null) 'numThreads': numThreads,
      'allowFp16': allowFp16,
    });

    if (result is String && result.isNotEmpty) {
      // Older native implementations returned just the id.
      return NativeMLModel.fromId(result, _methodChannel);
    }
    if (result is Map) {
      final id = result['modelId']?.toString();
      if (id == null || id.isEmpty) {
        throw const NativeMLException('LOAD_FAILED', 'The native side returned no model id');
      }
      final signature = result['signature'];
      return NativeMLModel.fromId(
        id,
        _methodChannel,
        acceleratorUsed: result['acceleratorUsed']?.toString() ?? 'unknown',
        signature: signature is Map ? ModelSignature.fromMap(signature) : null,
      );
    }
    throw const NativeMLException(
      'LOAD_FAILED',
      'Failed to load model: unexpected response from the native side',
    );
  }

  /// The OS version string, e.g. `Android 14` or `iOS 17.4`.
  static Future<String> getPlatformVersion() async {
    final version = await invokeNative<String>(_methodChannel, 'getPlatformVersion');
    return version ?? 'unknown';
  }

  /// Reports which accelerators and compute units this device supports.
  static Future<DeviceCapabilities> getDeviceCapabilities() async {
    final map = await invokeNative<Map<dynamic, dynamic>>(_methodChannel, 'getDeviceCapabilities');
    return DeviceCapabilities.fromMap(map ?? const {});
  }

  /// Whether the app may use the camera (see [NativeMLModel.startCamera]).
  static Future<CameraPermissionStatus> checkCameraPermission() async {
    final status = await invokeNative<String>(_methodChannel, 'cameraCheckPermission');
    return CameraPermissionStatus.fromName(status);
  }

  /// Asks the user for camera permission. Returns true when granted.
  ///
  /// On iOS `NSCameraUsageDescription` must be present in `Info.plist`,
  /// otherwise a `MISSING_USAGE_DESCRIPTION` error is thrown instead of
  /// crashing the app.
  static Future<bool> requestCameraPermission() async {
    final granted = await invokeNative<bool>(_methodChannel, 'cameraRequestPermission');
    return granted ?? false;
  }

  /// Releases every model loaded by this plugin. Useful after a hot restart
  /// when Dart-side handles have been lost.
  static Future<void> disposeAll() async {
    final legacy = _legacyStreamModels.values.toList();
    _legacyStreamModels.clear();
    for (final model in legacy) {
      await model.stopStream();
    }
    await invokeNative<dynamic>(_methodChannel, 'disposeAll');
  }

  /// Starts a stream for the model with [modelId].
  ///
  /// Prefer [NativeMLModel.startStream] together with
  /// [NativeMLModel.pushStreamInput], which also lets you feed frames.
  @Deprecated('Use NativeMLModel.startStream() instead')
  static Stream<InferenceResult> startStream({
    required String modelId,
    int maxQueueSize = 2,
  }) {
    final model = _legacyStreamModels.putIfAbsent(
      modelId,
      () => NativeMLModel.fromId(modelId, _methodChannel),
    );
    return model.startStream(maxQueueSize: maxQueueSize);
  }

  /// Stops a stream started with the deprecated [startStream].
  @Deprecated('Use NativeMLModel.stopStream() instead')
  static Future<void> stopStream({required String modelId}) async {
    final model = _legacyStreamModels.remove(modelId);
    if (model != null) {
      await model.stopStream();
      return;
    }
    await invokeNative<dynamic>(_methodChannel, 'stopStream', {'modelId': modelId});
  }
}
