#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_native_ml.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'flutter_native_ml'
  s.version          = '1.1.0'
  s.summary          = 'Direct access to Core ML (Neural Engine / GPU) for on-device inference in Flutter.'
  s.description      = <<-DESC
A Flutter plugin that gives direct access to device-native machine learning
accelerators: Core ML on iOS (Apple Neural Engine, GPU, CPU) and LiteRT on
Android (GPU delegate, NNAPI, XNNPACK).
                       DESC
  s.homepage         = 'https://github.com/miracle101000/flutter_native_ml'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Miracle Okolo' => 'okolomiracle101000@gmail.com' }
  s.source           = { :path => '.' }
  s.source_files     = 'flutter_native_ml/Sources/flutter_native_ml/**/*.swift'
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'
  s.frameworks = 'CoreML', 'CoreVideo', 'CoreGraphics', 'QuartzCore', 'UIKit'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  s.resource_bundles = {'flutter_native_ml_privacy' => ['flutter_native_ml/Sources/flutter_native_ml/PrivacyInfo.xcprivacy']}
end
