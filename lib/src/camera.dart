import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_native_ml/src/exceptions.dart';
import 'package:flutter_native_ml/src/models.dart';
import 'package:flutter_native_ml/src/native_ml_model.dart';

/// Name prefix of the per-session [EventChannel] that carries camera results.
const String cameraChannelPrefix = 'flutter_native_ml_camera/';

/// Which camera to open.
enum CameraLens { back, front }

/// Capture resolution presets (the preview stream; the model input is always
/// resized to what the model expects).
enum CameraResolution {
  /// About 640x480.
  low,

  /// About 1280x720 (default).
  medium,

  /// About 1920x1080.
  high,
}

/// How camera frames are fitted into the model's input size.
enum ResizeMode {
  /// Stretch to the input size, ignoring aspect ratio.
  fill,

  /// Scale to cover the input size and crop the centre (default).
  cover,

  /// Scale to fit inside the input size and pad with black.
  contain,
}

/// Result of a camera permission check.
enum CameraPermissionStatus {
  granted,
  denied,

  /// iOS only: access is restricted by device policy (parental controls, MDM).
  restricted,

  /// The user has not been asked yet.
  notDetermined,
  unknown;

  static CameraPermissionStatus fromName(String? name) {
    for (final status in CameraPermissionStatus.values) {
      if (status.name == name) return status;
    }
    return CameraPermissionStatus.unknown;
  }
}

/// How camera pixels are converted into model input values.
///
/// Pixels are converted per channel as `(value - mean[c]) / std[c]` for
/// floating-point inputs. Integer (`uint8`) inputs receive the raw 0-255 pixel
/// values and ignore [mean] / [std].
class CameraPreprocessing {
  /// Per-channel mean subtracted from each pixel (RGB order). Defaults to 0.
  final List<double>? mean;

  /// Per-channel divisor applied after subtracting [mean]. Defaults to 255 for
  /// floating-point inputs (producing values in `0..1`).
  final List<double>? std;

  /// How the frame is fitted to the model input size.
  final ResizeMode resizeMode;

  const CameraPreprocessing({
    this.mean,
    this.std,
    this.resizeMode = ResizeMode.cover,
  });

  /// Values in `0..1` (the default for float inputs).
  static const zeroToOne = CameraPreprocessing(mean: [0, 0, 0], std: [255, 255, 255]);

  /// Values in `-1..1` (MobileNet-style models).
  static const minusOneToOne = CameraPreprocessing(mean: [127.5, 127.5, 127.5], std: [127.5, 127.5, 127.5]);

  /// ImageNet mean / standard deviation in 0-255 pixel units.
  static const imagenet = CameraPreprocessing(
    mean: [123.675, 116.28, 103.53],
    std: [58.395, 57.12, 57.375],
  );

  Map<String, dynamic> toMap() => {
        if (mean != null) 'mean': mean,
        if (std != null) 'std': std,
        'resizeMode': resizeMode.name,
      };
}

/// A running camera pipeline that feeds frames to a model natively.
///
/// Frames never enter Dart: the platform camera writes into the model's input
/// buffer on the model's own thread and only [results] cross the platform
/// channel. Frames that arrive while inference is busy are skipped, so the
/// stream never falls behind ([InferenceResult.droppedFrames] counts them).
///
/// Show the live preview with [NativeCameraPreview] (or a `Texture` widget
/// using [textureId]).
class NativeCameraSession {
  /// Identifier of the session on the native side.
  final String id;

  /// The model receiving the frames.
  final NativeMLModel model;

  /// Flutter texture id of the preview, or null when the preview is disabled.
  final int? textureId;

  /// Size of the preview texture in its native orientation.
  final int previewWidth;
  final int previewHeight;

  /// Clockwise rotation to apply to the preview texture so it appears upright
  /// in a portrait app (0 on iOS, usually 90 or 270 on Android).
  final int previewRotationDegrees;

  /// Orientation of the camera sensor relative to the device (Android).
  final int sensorOrientation;

  final CameraLens lens;

  /// Name of the model input that receives the frames.
  final String inputName;

  /// Size and channel count the frames are converted to.
  final int inputWidth;
  final int inputHeight;
  final int inputChannels;

  final MethodChannel _channel;
  final StreamController<InferenceResult> _controller = StreamController.broadcast();
  StreamSubscription<dynamic>? _subscription;
  bool _stopped = false;
  bool _paused = false;

  NativeCameraSession._({
    required this.id,
    required this.model,
    required this.textureId,
    required this.previewWidth,
    required this.previewHeight,
    required this.previewRotationDegrees,
    required this.sensorOrientation,
    required this.lens,
    required this.inputName,
    required this.inputWidth,
    required this.inputHeight,
    required this.inputChannels,
    required MethodChannel channel,
  }) : _channel = channel;

  /// Creates a session that is not connected to the platform, for widget tests.
  @visibleForTesting
  NativeCameraSession.forTesting({
    required this.id,
    required this.model,
    this.textureId,
    this.previewWidth = 0,
    this.previewHeight = 0,
    this.previewRotationDegrees = 0,
    this.sensorOrientation = 0,
    this.lens = CameraLens.back,
    this.inputName = '',
    this.inputWidth = 0,
    this.inputHeight = 0,
    this.inputChannels = 0,
  }) : _channel = const MethodChannel('flutter_native_ml');

  /// Inference results, one per processed frame. Broadcast stream.
  Stream<InferenceResult> get results => _controller.stream;

  /// True until [stop] is called (or the model is disposed).
  bool get isRunning => !_stopped;

  /// True while paused with [pause].
  bool get isPaused => _paused;

  /// Aspect ratio (width / height) of the upright preview.
  double get previewAspectRatio {
    if (previewWidth == 0 || previewHeight == 0) return 1;
    final rotated = (previewRotationDegrees ~/ 90).isOdd;
    return rotated ? previewHeight / previewWidth : previewWidth / previewHeight;
  }

  /// Opens the camera for [model]. Used by [NativeMLModel.startCamera].
  static Future<NativeCameraSession> open(
    MethodChannel channel,
    NativeMLModel model, {
    required CameraLens lens,
    required CameraResolution resolution,
    required String? inputName,
    required CameraPreprocessing preprocessing,
    required double? maxFps,
    required bool preview,
  }) async {
    final response = await invokeNative<Map<dynamic, dynamic>>(channel, 'cameraStart', {
      'modelId': model.id,
      'lens': lens.name,
      'resolution': resolution.name,
      if (inputName != null) 'inputName': inputName,
      'preprocessing': preprocessing.toMap(),
      if (maxFps != null) 'maxFps': maxFps,
      'preview': preview,
    });
    if (response == null || response['sessionId'] == null) {
      throw const NativeMLException('CAMERA_START_FAILED', 'The native side returned no camera session');
    }
    final session = NativeCameraSession._(
      id: response['sessionId'].toString(),
      model: model,
      textureId: (response['textureId'] as num?)?.toInt(),
      previewWidth: (response['previewWidth'] as num?)?.toInt() ?? 0,
      previewHeight: (response['previewHeight'] as num?)?.toInt() ?? 0,
      previewRotationDegrees: (response['previewRotationDegrees'] as num?)?.toInt() ?? 0,
      sensorOrientation: (response['sensorOrientation'] as num?)?.toInt() ?? 0,
      lens: lens,
      inputName: response['inputName']?.toString() ?? inputName ?? '',
      inputWidth: (response['inputWidth'] as num?)?.toInt() ?? 0,
      inputHeight: (response['inputHeight'] as num?)?.toInt() ?? 0,
      inputChannels: (response['inputChannels'] as num?)?.toInt() ?? 0,
      channel: channel,
    );
    session._listen();
    return session;
  }

  void _listen() {
    final events = EventChannel('$cameraChannelPrefix$id').receiveBroadcastStream(id);
    _subscription = events.listen(
      (dynamic event) {
        if (event is Map) {
          _controller.add(InferenceResult.fromMap(event));
        } else {
          _controller.addError(FormatException('Unexpected camera event: ${event.runtimeType}'));
        }
      },
      onError: (Object error, StackTrace stackTrace) {
        _controller.addError(
          error is PlatformException ? NativeMLException.fromPlatform(error) : error,
          stackTrace,
        );
      },
      onDone: () => _teardown(notifyNative: false),
      cancelOnError: false,
    );
  }

  /// Stops delivering frames without releasing the camera.
  Future<void> pause() async {
    _ensureRunning();
    await invokeNative<dynamic>(_channel, 'cameraPause', {'sessionId': id});
    _paused = true;
  }

  /// Resumes after [pause]. Also re-attaches the preview surface, so call it
  /// when the app returns to the foreground.
  Future<void> resume() async {
    _ensureRunning();
    await invokeNative<dynamic>(_channel, 'cameraResume', {'sessionId': id});
    _paused = false;
  }

  /// Releases the camera and closes [results]. Idempotent.
  Future<void> stop() => _teardown(notifyNative: true);

  Future<void> _teardown({required bool notifyNative}) async {
    if (_stopped) return;
    _stopped = true;
    final subscription = _subscription;
    _subscription = null;
    await subscription?.cancel();
    if (notifyNative) {
      try {
        await invokeNative<dynamic>(_channel, 'cameraStop', {'sessionId': id});
      } on NativeMLException catch (e) {
        if (e.code != 'SESSION_NOT_FOUND' && e.code != 'MODEL_NOT_FOUND') rethrow;
      }
    }
    await _controller.close();
  }

  void _ensureRunning() {
    if (_stopped) throw StateError('Camera session $id has been stopped.');
  }

  @override
  String toString() => 'NativeCameraSession(id: $id, lens: ${lens.name}, '
      'preview: ${previewWidth}x$previewHeight@$previewRotationDegrees°, '
      'input: $inputName ${inputWidth}x$inputHeight×$inputChannels)';
}
