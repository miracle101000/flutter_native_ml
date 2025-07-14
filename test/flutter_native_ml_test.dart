import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_native_ml/flutter_native_ml.dart';


// class MockFlutterNativeMlPlatform
//     with MockPlatformInterfaceMixin
//     implements FlutterNativeMlPlatform {

//   @override
//   Future<String?> getPlatformVersion() => Future.value('42');
// }

// void main() {
//   final FlutterNativeMlPlatform initialPlatform = FlutterNativeMlPlatform.instance;

//   test('$MethodChannelFlutterNativeMl is the default instance', () {
//     expect(initialPlatform, isInstanceOf<MethodChannelFlutterNativeMl>());
//   });

//   test('getPlatformVersion', () async {
//     FlutterNativeMl flutterNativeMlPlugin = FlutterNativeMl();
//     MockFlutterNativeMlPlatform fakePlatform = MockFlutterNativeMlPlatform();
//     FlutterNativeMlPlatform.instance = fakePlatform;

//     expect(await flutterNativeMlPlugin.getPlatformVersion(), '42');
//   });
// }
