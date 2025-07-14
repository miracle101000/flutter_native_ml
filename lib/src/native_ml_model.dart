import 'package:flutter/services.dart';
import 'package:flutter_native_ml/src/models.dart';

class NativeMLModel {
  final String _modelId;
  final MethodChannel _channel;
  bool _isDisposed = false;
  ModelSignature? _signature;

  NativeMLModel.fromId(this._modelId, this._channel);

  Future<ModelSignature> getSignature() async {
    if (_isDisposed) throw StateError('Cannot get signature of a disposed model.');
    if (_signature != null) return _signature!;

    final signatureMap = await _channel.invokeMethod('getSignature', {'modelId': _modelId});
    _signature = ModelSignature.fromMap(signatureMap);
    return _signature!;
  }

  Future<InferenceResult> run(Map<String, dynamic> input) async {
    if (_isDisposed) throw StateError('Cannot run inference on a disposed model.');

    final result = await _channel.invokeMethod('run', {
      'modelId': _modelId,
      'input': input,
    });
    return InferenceResult.fromMap(result);
  }

  Future<void> dispose() async {
    if (!_isDisposed) {
      await _channel.invokeMethod('dispose', {'modelId': _modelId});
      _isDisposed = true;
    }
  }
}