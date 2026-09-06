import 'package:flutter/widgets.dart';
import 'package:flutter_native_ml/src/camera.dart';

/// Shows the live preview of a [NativeCameraSession], rotated upright.
///
/// The widget sizes itself to the preview's aspect ratio and scales it into
/// the available space using [fit].
class NativeCameraPreview extends StatelessWidget {
  final NativeCameraSession session;
  final BoxFit fit;

  const NativeCameraPreview({super.key, required this.session, this.fit = BoxFit.cover});

  @override
  Widget build(BuildContext context) {
    final textureId = session.textureId;
    if (textureId == null || session.previewWidth == 0 || session.previewHeight == 0) {
      return const SizedBox.expand();
    }
    final quarterTurns = (session.previewRotationDegrees ~/ 90) % 4;
    final rotated = quarterTurns.isOdd;
    final width = (rotated ? session.previewHeight : session.previewWidth).toDouble();
    final height = (rotated ? session.previewWidth : session.previewHeight).toDouble();
    return ClipRect(
      child: FittedBox(
        fit: fit,
        clipBehavior: Clip.hardEdge,
        child: SizedBox(
          width: width,
          height: height,
          child: RotatedBox(
            quarterTurns: quarterTurns,
            child: Texture(textureId: textureId),
          ),
        ),
      ),
    );
  }
}
