import 'dart:async';

import 'package:flutter/services.dart';
import 'package:flutter_native_ml/src/models.dart';
import 'package:flutter_native_ml/src/native_ml_model.dart';

export 'package:flutter_native_ml/src/models.dart';
export 'package:flutter_native_ml/src/native_ml_model.dart';

class FlutterNativeML {
  static const MethodChannel _methodChannel =
      MethodChannel('flutter_native_ml');

  /// Load a model by asset path — returns a NativeMLModel instance.
  static Future<NativeMLModel> loadModel({
    required String assetPath,
    ComputeUnit computeUnits = ComputeUnit.all,
  }) async {
    final modelId = await _methodChannel.invokeMethod<String>('loadModel', {
      'assetPath': assetPath,
      'computeUnits': computeUnits.name,
    });

    if (modelId == null) {
      throw Exception(
          'Failed to load model. Received null ID from native side.');
    }

    return NativeMLModel.fromId(modelId, _methodChannel);
  }

  /// Starts a continuous inference stream for a given modelId.
  /// Emits [InferenceResult]s as they arrive.
  static Stream<InferenceResult> startStream({required String modelId}) {
    final eventChannel =
        EventChannel('flutter_native_ml_stream/$modelId');

    // Tell native side to start producing events for this modelId
    _methodChannel.invokeMethod('startStream', {'modelId': modelId});

    return eventChannel
        .receiveBroadcastStream(modelId)
        .asyncMap((event) async {
      if (event is Map) {
        return InferenceResult.fromMap(Map<String, dynamic>.from(event));
      } else {
        throw const FormatException('Unexpected event type from native stream');
      }
    });
  }

  /// Stop the continuous inference stream for this modelId.
  static Future<void> stopStream({required String modelId}) async {
    await _methodChannel.invokeMethod('stopStream', {'modelId': modelId});
  }
}
