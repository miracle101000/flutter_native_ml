import 'dart:async';

import 'package:flutter/services.dart';
import 'package:flutter_native_ml/src/exceptions.dart';
import 'package:flutter_native_ml/src/models.dart';

/// Name prefix of the per-model [EventChannel] that carries streaming results.
const String streamChannelPrefix = 'flutter_native_ml_stream/';

/// Invokes [method] on [channel], converting platform errors into
/// [NativeMLException]s.
Future<T?> invokeNative<T>(
  MethodChannel channel,
  String method, [
  Map<String, Object?>? arguments,
]) async {
  try {
    return await channel.invokeMethod<T>(method, arguments);
  } on PlatformException catch (e) {
    throw NativeMLException.fromPlatform(e);
  } on MissingPluginException catch (e) {
    throw NativeMLException(
      'MISSING_PLUGIN',
      e.message ?? 'flutter_native_ml is not available on this platform',
    );
  }
}

/// Converts user-facing input values into the wire format understood by the
/// native side. [TensorData] and [ImageInput] become maps; everything else
/// (lists, typed lists, numbers, strings, maps) is passed through unchanged.
Map<String, Object?> encodeInputs(Map<String, Object?> input) {
  final encoded = <String, Object?>{};
  input.forEach((name, value) {
    if (value == null) {
      throw ArgumentError.value(null, name, 'Input values must not be null');
    }
    if (value is TensorData) {
      encoded[name] = value.toMap();
    } else if (value is ImageInput) {
      encoded[name] = value.toMap();
    } else {
      encoded[name] = value;
    }
  });
  return encoded;
}

/// A model that has been loaded into the native runtime
/// (LiteRT on Android, Core ML on iOS).
///
/// Obtain an instance with [FlutterNativeML.loadModel] and release it with
/// [dispose] when you are done.
class NativeMLModel {
  /// Identifier of the model on the native side.
  final String id;

  /// Description of the compute unit the model was loaded on, as reported by
  /// the platform (for example `GPU`, `NNAPI`, `CPU (XNNPACK, 4 threads)` or
  /// `CPU+GPU+ANE`).
  final String acceleratorUsed;

  final MethodChannel _channel;
  ModelSignature? _signature;
  bool _isDisposed = false;

  StreamController<InferenceResult>? _streamController;
  Stream<InferenceResult>? _stream;
  StreamSubscription<dynamic>? _eventSubscription;
  Future<void>? _streamReady;

  NativeMLModel.fromId(
    this.id,
    this._channel, {
    ModelSignature? signature,
    this.acceleratorUsed = 'unknown',
  }) : _signature = signature;

  /// Whether [dispose] has been called.
  bool get isDisposed => _isDisposed;

  /// The cached signature, if it has been fetched (models loaded with
  /// [FlutterNativeML.loadModel] always have it).
  ModelSignature? get signature => _signature;

  /// Whether [startStream] is active.
  bool get isStreaming => _streamController != null;

  /// Returns the model's inputs and outputs.
  ///
  /// The signature is cached; pass `refresh: true` to query the platform again.
  Future<ModelSignature> getSignature({bool refresh = false}) async {
    _ensureNotDisposed();
    final cached = _signature;
    if (!refresh && cached != null) return cached;
    final map = await invokeNative<Map<dynamic, dynamic>>(
      _channel,
      'getSignature',
      {'modelId': id},
    );
    final parsed = ModelSignature.fromMap(map ?? const {});
    _signature = parsed;
    return parsed;
  }

  /// Runs a single inference.
  ///
  /// [input] maps input names (see [ModelSignature.inputs]) to values:
  ///
  /// * flat `List<num>` / `List<bool>` / `List<String>`, or a typed list such
  ///   as `Float32List`, `Int32List`, `Uint8List` (fastest);
  /// * [TensorData] to also provide an explicit shape (dynamic inputs);
  /// * [ImageInput] for Core ML image inputs;
  /// * `String`, `int`, `double` or `Map` for scalar / dictionary features.
  ///
  /// On Android inputs may also be addressed by their SignatureDef alias
  /// (for example `input_1` instead of `serving_default_input_1:0`);
  /// [signatureKey] narrows the lookup to one signature.
  Future<InferenceResult> run(
    Map<String, Object?> input, {
    String? signatureKey,
  }) async {
    _ensureNotDisposed();
    if (input.isEmpty) {
      throw ArgumentError.value(input, 'input', 'must contain at least one input');
    }
    final result = await invokeNative<Map<dynamic, dynamic>>(_channel, 'run', {
      'modelId': id,
      'input': encodeInputs(input),
      if (signatureKey != null) 'signatureKey': signatureKey,
    });
    if (result == null) {
      throw const NativeMLException('NULL_RESULT', 'The native side returned no result');
    }
    return InferenceResult.fromMap(result);
  }

  /// Starts a streaming inference session and returns the stream of results.
  ///
  /// Push frames with [pushStreamInput]; each produces one [InferenceResult]
  /// on the returned stream. Frames are processed sequentially on a dedicated
  /// native thread. When frames arrive faster than the model can process them,
  /// the oldest queued frames are dropped once [maxQueueSize] is exceeded
  /// ([InferenceResult.droppedFrames] reports how many).
  ///
  /// The returned stream is a broadcast stream. Call [stopStream] to end the
  /// session; calling [startStream] again while active returns the same stream.
  Stream<InferenceResult> startStream({int maxQueueSize = 2}) {
    _ensureNotDisposed();
    final existing = _stream;
    if (existing != null) return existing;
    if (maxQueueSize < 1) {
      throw ArgumentError.value(maxQueueSize, 'maxQueueSize', 'must be at least 1');
    }
    final controller = StreamController<InferenceResult>.broadcast();
    final stream = controller.stream;
    _streamController = controller;
    _stream = stream;
    _streamReady = _openStream(controller, maxQueueSize);
    return stream;
  }

  Future<void> _openStream(
    StreamController<InferenceResult> controller,
    int maxQueueSize,
  ) async {
    try {
      await invokeNative<dynamic>(_channel, 'startStream', {
        'modelId': id,
        'maxQueueSize': maxQueueSize,
      });
    } catch (error, stackTrace) {
      controller.addError(error, stackTrace);
      await _closeStream(notifyNative: false);
      return;
    }
    if (_streamController != controller) return; // stopped meanwhile

    final events = EventChannel('$streamChannelPrefix$id').receiveBroadcastStream(id);
    _eventSubscription = events.listen(
      (dynamic event) {
        if (event is Map) {
          controller.add(InferenceResult.fromMap(event));
        } else {
          controller.addError(
            FormatException('Unexpected event from native stream: ${event.runtimeType}'),
          );
        }
      },
      onError: (Object error, StackTrace stackTrace) {
        controller.addError(
          error is PlatformException ? NativeMLException.fromPlatform(error) : error,
          stackTrace,
        );
      },
      onDone: () {
        if (_streamController == controller) {
          _closeStream(notifyNative: false);
        }
      },
      cancelOnError: false,
    );
  }

  /// Queues one frame of input for the active stream (see [startStream]).
  ///
  /// Returns immediately once the frame has been queued; the result arrives on
  /// the stream. Throws a [StateError] if no stream is active.
  Future<StreamPushResult> pushStreamInput(Map<String, Object?> input) async {
    _ensureNotDisposed();
    final ready = _streamReady;
    if (_streamController == null || ready == null) {
      throw StateError('Call startStream() before pushStreamInput().');
    }
    await ready;
    if (_streamController == null) {
      throw StateError('The stream has been stopped.');
    }
    final result = await invokeNative<Map<dynamic, dynamic>>(_channel, 'streamInput', {
      'modelId': id,
      'input': encodeInputs(input),
    });
    return StreamPushResult.fromMap(result);
  }

  /// Stops the streaming session started with [startStream]. Safe to call
  /// when no stream is active.
  Future<void> stopStream() async {
    if (_streamController == null) return;
    final ready = _streamReady;
    if (ready != null) {
      await ready;
    }
    await _closeStream(notifyNative: true);
  }

  Future<void> _closeStream({required bool notifyNative}) async {
    final controller = _streamController;
    _streamController = null;
    _stream = null;
    _streamReady = null;
    final subscription = _eventSubscription;
    _eventSubscription = null;
    await subscription?.cancel();
    if (notifyNative) {
      try {
        await invokeNative<dynamic>(_channel, 'stopStream', {'modelId': id});
      } on NativeMLException {
        // The model may already be gone on the native side; nothing to stop.
      }
    }
    await controller?.close();
  }

  /// Releases the native model. Idempotent.
  Future<void> dispose() async {
    if (_isDisposed) return;
    _isDisposed = true;
    if (_streamController != null) {
      await _closeStream(notifyNative: false);
    }
    try {
      await invokeNative<dynamic>(_channel, 'dispose', {'modelId': id});
    } on NativeMLException catch (e) {
      if (e.code != 'MODEL_NOT_FOUND' && e.code != 'MODEL_DISPOSED') rethrow;
    }
  }

  void _ensureNotDisposed() {
    if (_isDisposed) {
      throw StateError('NativeMLModel $id has been disposed.');
    }
  }

  @override
  String toString() => 'NativeMLModel(id: $id, accelerator: $acceleratorUsed'
      '${_isDisposed ? ', disposed' : ''})';
}
