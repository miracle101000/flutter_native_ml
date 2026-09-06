import 'package:flutter/services.dart';

/// An error reported by the native (Android / iOS) side of the plugin.
///
/// [code] is a stable, machine-readable identifier such as `MODEL_NOT_FOUND`,
/// `SHAPE_MISMATCH`, `MISSING_INPUT`, `INFERENCE_FAILED` or `LOAD_FAILED`.
class NativeMLException implements Exception {
  /// Machine-readable error code.
  final String code;

  /// Human-readable description of what went wrong.
  final String? message;

  /// Optional platform-specific details (for example a native stack trace).
  final Object? details;

  const NativeMLException(this.code, this.message, {this.details});

  /// Wraps a [PlatformException] thrown by a platform channel call.
  factory NativeMLException.fromPlatform(PlatformException e) =>
      NativeMLException(e.code, e.message, details: e.details);

  @override
  String toString() => 'NativeMLException($code): ${message ?? 'no message'}';
}
