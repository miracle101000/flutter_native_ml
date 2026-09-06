import AVFoundation
import CoreImage
import CoreML
import Flutter
import QuartzCore

/// Options parsed from the `cameraStart` call.
struct CameraConfig {
    let lens: String
    let resolution: String
    let mean: [Float]?
    let std: [Float]?
    let resizeMode: String
    let maxFps: Double?
    let preview: Bool
}

/// Layout of the model input that receives camera frames.
struct CameraInputSpec {
    enum Kind { case image, multiArray }
    enum Layout { case hwc, chw }

    let name: String
    let kind: Kind
    let width: Int
    let height: Int
    let channels: Int
    let layout: Layout
    let dataType: MLMultiArrayDataType
    let pixelFormat: OSType
    let shape: [Int]
}

/**
 An AVFoundation pipeline that feeds camera frames straight into a Core ML model.

 Frames are delivered as `CVPixelBuffer`s. Image inputs receive a GPU-resized pixel buffer
 directly; multi-array inputs are filled from the resized pixels on the model's queue. Only the
 results cross the platform channel. Frames that arrive while inference is busy are skipped.
 */
final class CameraSession: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, FlutterTexture {
    private static let channelPrefix = "flutter_native_ml_camera/"

    let id: String
    let model: LoadedModel
    let input: CameraInputSpec
    private let config: CameraConfig
    private let registrar: FlutterPluginRegistrar
    private let convertOutput: (MLFeatureProvider) -> ([String: Any], [String: Any])
    private let flutterError: (Error) -> FlutterError

    private let session = AVCaptureSession()
    private let captureQueue: DispatchQueue
    private let sessionQueue = DispatchQueue(label: "flutter_native_ml.camera.session")
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

    private var eventChannel: FlutterEventChannel?
    private var streamHandler: ModelStreamHandler?
    var sink: FlutterEventSink? // main thread only
    private(set) var textureId: Int64 = -1
    private(set) var previewWidth = 0
    private(set) var previewHeight = 0

    private let previewLock = NSLock()
    private var latestPreviewBuffer: CVPixelBuffer?

    // Capture-queue state.
    private var busy = false
    private var paused = false
    private var stopped = false
    private var frameCounter: Int64 = 0
    private var skippedFrames: Int64 = 0
    private var lastFrameTime: Double = 0

    // Model-queue state (reused between frames).
    private var inputPixelBuffer: CVPixelBuffer?
    private var inputArray: MLMultiArray?

    init(
        id: String,
        model: LoadedModel,
        input: CameraInputSpec,
        config: CameraConfig,
        registrar: FlutterPluginRegistrar,
        convertOutput: @escaping (MLFeatureProvider) -> ([String: Any], [String: Any]),
        flutterError: @escaping (Error) -> FlutterError
    ) {
        self.id = id
        self.model = model
        self.input = input
        self.config = config
        self.registrar = registrar
        self.convertOutput = convertOutput
        self.flutterError = flutterError
        self.captureQueue = DispatchQueue(label: "flutter_native_ml.camera.\(id)", qos: .userInitiated)
        super.init()
    }

    // MARK: - Lifecycle (main thread)

    func start() throws {
        guard Bundle.main.object(forInfoDictionaryKey: "NSCameraUsageDescription") != nil else {
            throw NativeMLError("MISSING_USAGE_DESCRIPTION", "Add NSCameraUsageDescription to your app's Info.plist before using the camera")
        }
        let position: AVCaptureDevice.Position = config.lens == "front" ? .front : .back
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            throw NativeMLError("CAMERA_UNAVAILABLE", "No \(config.lens) camera is available on this device")
        }

        session.beginConfiguration()
        session.sessionPreset = preset()
        let deviceInput: AVCaptureDeviceInput
        do {
            deviceInput = try AVCaptureDeviceInput(device: device)
        } catch {
            session.commitConfiguration()
            throw NativeMLError("CAMERA_UNAVAILABLE", "Could not open the camera: \(error.localizedDescription)")
        }
        guard session.canAddInput(deviceInput) else {
            session.commitConfiguration()
            throw NativeMLError("CAMERA_UNAVAILABLE", "The camera input could not be added to the capture session")
        }
        session.addInput(deviceInput)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: captureQueue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            throw NativeMLError("CAMERA_UNAVAILABLE", "The video output could not be added to the capture session")
        }
        session.addOutput(output)
        if let connection = output.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
            } else if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            if position == .front, connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
        }
        session.commitConfiguration()

        // Frames are rotated to portrait, so swap the sensor dimensions.
        let dimensions = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        previewWidth = Int(min(dimensions.width, dimensions.height))
        previewHeight = Int(max(dimensions.width, dimensions.height))

        if config.preview {
            textureId = registrar.textures().register(self)
        }

        let channel = FlutterEventChannel(name: CameraSession.channelPrefix + id, binaryMessenger: registrar.messenger())
        let handler = ModelStreamHandler(
            onListen: { [weak self] sink in self?.sink = sink },
            onCancel: { [weak self] in self?.sink = nil }
        )
        channel.setStreamHandler(handler)
        eventChannel = channel
        streamHandler = handler

        sessionQueue.async { [session] in session.startRunning() }
    }

    func pause() {
        captureQueue.sync { self.paused = true }
        sessionQueue.async { [session] in
            if session.isRunning { session.stopRunning() }
        }
    }

    func resume() {
        captureQueue.sync { self.paused = false }
        sessionQueue.async { [session] in
            if !session.isRunning { session.startRunning() }
        }
    }

    /// Main thread. Stops capture and releases the preview texture and event channel.
    func stop() {
        captureQueue.sync { self.stopped = true }
        let session = self.session
        sessionQueue.async {
            if session.isRunning { session.stopRunning() }
            session.beginConfiguration()
            session.inputs.forEach { session.removeInput($0) }
            session.outputs.forEach { session.removeOutput($0) }
            session.commitConfiguration()
        }
        if let sink = sink {
            self.sink = nil
            sink(FlutterEndOfEventStream)
        }
        eventChannel?.setStreamHandler(nil)
        eventChannel = nil
        streamHandler = nil
        if textureId >= 0 {
            registrar.textures().unregisterTexture(textureId)
            textureId = -1
        }
        previewLock.lock()
        latestPreviewBuffer = nil
        previewLock.unlock()
    }

    private func preset() -> AVCaptureSession.Preset {
        switch config.resolution {
        case "low":
            return .vga640x480
        case "high":
            return .hd1920x1080
        default:
            return .hd1280x720
        }
    }

    // MARK: - FlutterTexture

    func copyPixelBuffer() -> Unmanaged<CVPixelBuffer>? {
        previewLock.lock()
        defer { previewLock.unlock() }
        guard let buffer = latestPreviewBuffer else { return nil }
        return Unmanaged.passRetained(buffer)
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate (capture queue)

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !stopped, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        if config.preview, textureId >= 0 {
            previewLock.lock()
            latestPreviewBuffer = pixelBuffer
            previewLock.unlock()
            registrar.textures().textureFrameAvailable(textureId)
        }

        if paused || busy {
            skippedFrames += 1
            return
        }
        let now = CACurrentMediaTime()
        if let maxFps = config.maxFps, maxFps > 0, now - lastFrameTime < 1.0 / maxFps {
            skippedFrames += 1
            return
        }
        lastFrameTime = now
        busy = true
        let frameId = frameCounter
        frameCounter += 1
        let dropped = skippedFrames
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let timestampMicros = timestamp.isValid ? Int64(CMTimeGetSeconds(timestamp) * 1_000_000) : Int64(now * 1_000_000)

        model.queue.async { [weak self] in
            guard let self = self else { return }
            defer { self.captureQueue.async { self.busy = false } }
            do {
                var payload = try self.infer(pixelBuffer)
                payload["frameId"] = frameId
                payload["droppedFrames"] = dropped
                payload["frame"] = [
                    "width": CVPixelBufferGetWidth(pixelBuffer),
                    "height": CVPixelBufferGetHeight(pixelBuffer),
                    "rotationDegrees": 0,
                    "timestampMicros": timestampMicros,
                ]
                self.emit { sink in sink(payload) }
            } catch {
                let flutterError = self.flutterError(error)
                self.emit { sink in
                    sink(FlutterError(code: flutterError.code, message: flutterError.message, details: ["frameId": frameId]))
                }
            }
        }
    }

    private func emit(_ action: @escaping (FlutterEventSink) -> Void) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self, let sink = self.sink else { return }
            // Results that complete after pause()/stop() are dropped.
            let suppressed = self.captureQueue.sync { self.paused || self.stopped }
            if suppressed { return }
            action(sink)
        }
    }

    // MARK: - Inference (model queue)

    private func infer(_ source: CVPixelBuffer) throws -> [String: Any] {
        let resized = try resize(source)
        let feature: MLFeatureValue
        switch input.kind {
        case .image:
            feature = MLFeatureValue(pixelBuffer: resized)
        case .multiArray:
            feature = MLFeatureValue(multiArray: try fillMultiArray(from: resized))
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [input.name: feature])
        let start = CACurrentMediaTime()
        let outputProvider: MLFeatureProvider
        do {
            outputProvider = try model.model.prediction(from: provider)
        } catch {
            throw NativeMLError("INFERENCE_FAILED", error.localizedDescription)
        }
        let micros = (CACurrentMediaTime() - start) * 1_000_000
        let (output, shapes) = convertOutput(outputProvider)
        return [
            "output": output,
            "outputShapes": shapes,
            "inferenceTime": micros,
            "acceleratorUsed": model.acceleratorLabel,
        ]
    }

    /// Scales/crops the frame into a reusable pixel buffer of the model's input size using Core Image (GPU).
    private func resize(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = input.width
        let height = input.height
        let format = input.kind == .image ? input.pixelFormat : kCVPixelFormatType_32BGRA
        if inputPixelBuffer == nil {
            var buffer: CVPixelBuffer?
            let attributes: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferMetalCompatibilityKey: true,
            ]
            let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attributes as CFDictionary, &buffer)
            guard status == kCVReturnSuccess, let created = buffer else {
                throw NativeMLError("INTERNAL", "Could not allocate the \(width)x\(height) model input buffer (status \(status))")
            }
            inputPixelBuffer = created
        }
        let target = inputPixelBuffer!

        let sourceWidth = CGFloat(CVPixelBufferGetWidth(source))
        let sourceHeight = CGFloat(CVPixelBufferGetHeight(source))
        let targetWidth = CGFloat(width)
        let targetHeight = CGFloat(height)
        var image = CIImage(cvPixelBuffer: source)
        let transform: CGAffineTransform
        switch config.resizeMode {
        case "fill":
            transform = CGAffineTransform(scaleX: targetWidth / sourceWidth, y: targetHeight / sourceHeight)
        case "contain":
            let scale = min(targetWidth / sourceWidth, targetHeight / sourceHeight)
            transform = CGAffineTransform(translationX: (targetWidth - sourceWidth * scale) / 2, y: (targetHeight - sourceHeight * scale) / 2)
                .scaledBy(x: scale, y: scale)
        default: // cover
            let scale = max(targetWidth / sourceWidth, targetHeight / sourceHeight)
            transform = CGAffineTransform(translationX: (targetWidth - sourceWidth * scale) / 2, y: (targetHeight - sourceHeight * scale) / 2)
                .scaledBy(x: scale, y: scale)
        }
        image = image.transformed(by: transform)
        let bounds = CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight)
        let composed = image.composited(over: CIImage(color: .black).cropped(to: bounds))
        ciContext.render(composed, to: target, bounds: bounds, colorSpace: colorSpace)
        return target
    }

    /// Converts the resized BGRA pixels into the model's multi-array with mean / std normalisation.
    private func fillMultiArray(from buffer: CVPixelBuffer) throws -> MLMultiArray {
        if inputArray == nil {
            do {
                inputArray = try MLMultiArray(shape: input.shape.map { NSNumber(value: $0) }, dataType: input.dataType)
            } catch {
                throw NativeMLError("INVALID_SHAPE", "Could not allocate the camera input array \(input.shape): \(error.localizedDescription)")
            }
        }
        let array = inputArray!
        let width = input.width
        let height = input.height
        let channels = input.channels
        let plane = height * width
        let mean = config.mean ?? [0, 0, 0, 0]
        let std = config.std ?? [255, 255, 255, 255]
        var invStd = [Float](repeating: 1, count: 4)
        var meanC = [Float](repeating: 0, count: 4)
        for c in 0..<4 {
            let s = c < std.count ? std[c] : (std.last ?? 255)
            invStd[c] = s == 0 ? 1 : 1 / s
            meanC[c] = c < mean.count ? mean[c] : (mean.last ?? 0)
        }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw NativeMLError("INTERNAL", "The camera input buffer has no storage")
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        let dataType = input.dataType
        let layout = input.layout

        try withWritableBytes(array) { destination in
            @inline(__always) func store(_ index: Int, _ value: Float) {
                switch dataType {
                case .float32:
                    destination.storeBytes(of: value, toByteOffset: index * 4, as: Float.self)
                case .double:
                    destination.storeBytes(of: Double(value), toByteOffset: index * 8, as: Double.self)
                case .int32:
                    destination.storeBytes(of: Int32(value.rounded()), toByteOffset: index * 4, as: Int32.self)
                case .float16:
                    destination.storeBytes(of: CameraSession.floatToHalf(value), toByteOffset: index * 2, as: UInt16.self)
                @unknown default:
                    break
                }
            }
            for y in 0..<height {
                let row = pixels + y * bytesPerRow
                for x in 0..<width {
                    let p = row + x * 4 // BGRA
                    let b = Float(p[0]), g = Float(p[1]), r = Float(p[2]), a = Float(p[3])
                    let values: [Float]
                    switch channels {
                    case 1:
                        values = [(0.299 * r + 0.587 * g + 0.114 * b - meanC[0]) * invStd[0]]
                    case 3:
                        values = [(r - meanC[0]) * invStd[0], (g - meanC[1]) * invStd[1], (b - meanC[2]) * invStd[2]]
                    default:
                        values = [(r - meanC[0]) * invStd[0], (g - meanC[1]) * invStd[1], (b - meanC[2]) * invStd[2], (a - meanC[3]) * invStd[3]]
                    }
                    for c in 0..<values.count {
                        let index = layout == .chw ? c * plane + y * width + x : (y * width + x) * channels + c
                        store(index, values[c])
                    }
                }
            }
        }
        return array
    }

    private func withWritableBytes<R>(_ array: MLMultiArray, _ body: (UnsafeMutableRawPointer) throws -> R) throws -> R {
        if #available(iOS 15.4, *) {
            return try array.withUnsafeMutableBytes { buffer, _ in
                guard let base = buffer.baseAddress else {
                    throw NativeMLError("INTERNAL", "Multi-array has no storage")
                }
                return try body(base)
            }
        } else {
            return try body(array.dataPointer)
        }
    }

    private static func floatToHalf(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exponent = Int32((bits >> 23) & 0xFF) - 127 + 15
        var mantissa = bits & 0x7F_FFFF
        if (bits & 0x7F80_0000) == 0x7F80_0000 {
            return sign | 0x7C00 | (mantissa != 0 ? 0x200 : 0)
        }
        if exponent >= 31 {
            return sign | 0x7C00
        }
        if exponent <= 0 {
            if exponent < -10 {
                return sign
            }
            mantissa |= 0x80_0000
            let shift = UInt32(14 - exponent)
            var half = UInt16(mantissa >> shift)
            let remainder = mantissa & ((1 << shift) - 1)
            let halfway = UInt32(1) << (shift - 1)
            if remainder > halfway || (remainder == halfway && (half & 1) == 1) {
                half += 1
            }
            return sign | half
        }
        var half = sign | UInt16(exponent << 10) | UInt16(mantissa >> 13)
        let remainder = mantissa & 0x1FFF
        if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) == 1) {
            half += 1
        }
        return half
    }
}
