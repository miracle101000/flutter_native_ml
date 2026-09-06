package com.example.flutter_native_ml

import android.content.Context
import android.os.Handler
import android.util.Log
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.view.TextureRegistry
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean

/** Options parsed from the `cameraStart` call. */
internal class CameraConfig(
    val lens: String,
    val resolution: String,
    val mean: FloatArray?,
    val std: FloatArray?,
    val resizeMode: String,
    val maxFps: Double?,
    val preview: Boolean
)

/** Layout of the model input the camera frames are written into. */
internal class CameraInputSpec(
    val name: String,
    val index: Int,
    val width: Int,
    val height: Int,
    val channels: Int,
    val dataType: DataType,
    val numBytes: Int
)

/**
 * A CameraX pipeline that converts every frame directly into a LiteRT input buffer and runs the
 * model, without the frame ever crossing the platform channel.
 *
 * Frame analysis runs on the model's own executor, so frames are processed sequentially and, with
 * CameraX's keep-only-latest backpressure, stale frames are dropped automatically.
 */
internal class CameraSession(
    val id: String,
    val modelId: String,
    private val context: Context,
    private val messenger: BinaryMessenger,
    private val textureRegistry: TextureRegistry,
    private val mainHandler: Handler,
    private val modelExecutor: ExecutorService,
    private val input: CameraInputSpec,
    private val config: CameraConfig,
    /** Runs inference with the filled input buffer. Called on the model executor. */
    private val infer: (ByteBuffer) -> MutableMap<String, Any?>
) {
    private companion object {
        const val TAG = "FlutterNativeML.Camera"
        const val CHANNEL_PREFIX = "flutter_native_ml_camera/"
        const val PADDING = Int.MIN_VALUE
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var lifecycleOwner: LifecycleOwner? = null
    private var preview: Preview? = null
    private var analysis: ImageAnalysis? = null
    private var surfaceProducer: TextureRegistry.SurfaceProducer? = null
    private var eventChannel: EventChannel? = null

    @Volatile
    var sink: EventChannel.EventSink? = null

    @Volatile
    private var paused = false

    @Volatile
    private var stopped = false
    private val started = AtomicBoolean(false)

    var previewWidth = 0
        private set
    var previewHeight = 0
        private set
    var sensorOrientation = 0
        private set
    val textureId: Long?
        get() = surfaceProducer?.id()

    // Frame conversion state (model thread only).
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(input.numBytes).order(ByteOrder.nativeOrder())
    private var rowOffsets = IntArray(0)
    private var colOffsets = IntArray(0)
    private var lutKey = ""
    private var uprightWidth = 0
    private var uprightHeight = 0
    private var frameCounter = 0L
    private var skippedFrames = 0L
    private var lastFrameNanos = 0L
    private val minFrameIntervalNanos: Long =
        config.maxFps?.takeIf { it > 0 }?.let { (1_000_000_000.0 / it).toLong() } ?: 0L

    /**
     * Opens the camera. Must be called on the main thread. [onReady] is invoked on the main thread
     * once the preview size is known (or immediately when the preview is disabled); [onError]
     * when the camera could not be started.
     */
    fun start(lifecycleOwner: LifecycleOwner, onReady: () -> Unit, onError: (Throwable) -> Unit) {
        this.lifecycleOwner = lifecycleOwner
        eventChannel = EventChannel(messenger, CHANNEL_PREFIX + id).also {
            it.setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
                    sink = events
                }

                override fun onCancel(arguments: Any?) {
                    sink = null
                }
            })
        }

        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            try {
                if (stopped) return@addListener
                val provider = providerFuture.get()
                cameraProvider = provider
                bindUseCases(provider, lifecycleOwner, onReady)
            } catch (t: Throwable) {
                onError(t)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bindUseCases(provider: ProcessCameraProvider, owner: LifecycleOwner, onReady: (() -> Unit)?) {
        val selector = if (config.lens == "front") {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
        if (!provider.hasCamera(selector)) {
            throw NativeMlException("CAMERA_UNAVAILABLE", "No ${config.lens} camera is available on this device")
        }

        val useCases = mutableListOf<UseCase>()
        val analysisUseCase = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(maxOf(input.width, 640), maxOf(input.height, 480)),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()
            )
            .build()
        analysisUseCase.setAnalyzer(modelExecutor) { image -> onFrame(image) }
        analysis = analysisUseCase
        useCases += analysisUseCase

        var readyCalled = onReady == null
        val ready: () -> Unit = {
            if (!readyCalled) {
                readyCalled = true
                onReady?.invoke()
            }
        }

        if (config.preview) {
            val producer = surfaceProducer ?: textureRegistry.createSurfaceProducer().also { surfaceProducer = it }
            val previewUseCase = Preview.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(previewTargetSize(), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                        )
                        .build()
                )
                .build()
            previewUseCase.setSurfaceProvider(ContextCompat.getMainExecutor(context)) { request ->
                val resolution = request.resolution
                previewWidth = resolution.width
                previewHeight = resolution.height
                producer.setSize(resolution.width, resolution.height)
                request.provideSurface(producer.surface, ContextCompat.getMainExecutor(context)) { }
                ready()
            }
            preview = previewUseCase
            useCases += previewUseCase
            // Never leave the caller waiting if CameraX does not request a surface.
            mainHandler.postDelayed({ ready() }, 3_000)
        }

        provider.unbindAll()
        val boundCamera = provider.bindToLifecycle(owner, selector, *useCases.toTypedArray())
        camera = boundCamera
        sensorOrientation = try {
            boundCamera.cameraInfo.sensorRotationDegrees
        } catch (t: Throwable) {
            0
        }
        if (!config.preview) ready()
    }

    private fun previewTargetSize(): Size = when (config.resolution) {
        "low" -> Size(640, 480)
        "high" -> Size(1920, 1080)
        else -> Size(1280, 720)
    }

    /** Whether the Flutter surface applies the camera rotation itself (newer Flutter versions). */
    fun previewRotationDegrees(): Int {
        val producer = surfaceProducer ?: return 0
        val handled = try {
            producer.javaClass.getMethod("handlesCropAndRotation").invoke(producer) as? Boolean ?: false
        } catch (t: Throwable) {
            false
        }
        return if (handled) 0 else sensorOrientation
    }

    /** Main thread. */
    fun pause() {
        paused = true
    }

    /** Main thread. Re-binds the use cases so a recreated preview surface is picked up. */
    fun resume() {
        paused = false
        val provider = cameraProvider ?: return
        val owner = lifecycleOwner ?: return
        try {
            provider.unbindAll()
            bindUseCases(provider, owner, null)
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to re-bind camera use cases on resume", t)
        }
    }

    /** Main thread. */
    fun stop() {
        if (stopped) return
        stopped = true
        paused = true
        try {
            analysis?.clearAnalyzer()
            cameraProvider?.unbindAll()
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to unbind camera use cases", t)
        }
        analysis = null
        preview = null
        camera = null
        sink?.let { s ->
            sink = null
            try {
                s.endOfStream()
            } catch (t: Throwable) {
            }
        }
        eventChannel?.setStreamHandler(null)
        eventChannel = null
        try {
            surfaceProducer?.release()
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to release preview surface", t)
        }
        surfaceProducer = null
    }

    // ---------------------------------------------------------------------------------------------
    // Frame processing (model executor thread)
    // ---------------------------------------------------------------------------------------------

    private fun onFrame(image: ImageProxy) {
        try {
            if (stopped || paused) {
                return
            }
            val now = System.nanoTime()
            if (minFrameIntervalNanos > 0 && now - lastFrameNanos < minFrameIntervalNanos) {
                skippedFrames++
                return
            }
            lastFrameNanos = now
            val rotation = image.imageInfo.rotationDegrees
            fillInput(image, rotation)
            val payload = infer(inputBuffer)
            payload["frameId"] = frameCounter++
            payload["droppedFrames"] = skippedFrames
            payload["frame"] = mapOf(
                "width" to uprightWidth,
                "height" to uprightHeight,
                "rotationDegrees" to rotation,
                "timestampMicros" to image.imageInfo.timestamp / 1_000
            )
            emit { it.success(payload) }
        } catch (e: NativeMlException) {
            emit { it.error(e.code, e.message, e.details) }
        } catch (t: Throwable) {
            Log.e(TAG, "Camera frame inference failed", t)
            emit { it.error("INFERENCE_FAILED", t.message ?: t.toString(), null) }
        } finally {
            image.close()
        }
    }

    private fun emit(action: (EventChannel.EventSink) -> Unit) {
        mainHandler.post {
            // Results that complete after pause()/stop() are dropped, so callers never see
            // stragglers once those calls have returned.
            if (paused || stopped) return@post
            val s = sink ?: return@post
            try {
                action(s)
            } catch (t: Throwable) {
                Log.w(TAG, "Failed to deliver camera event", t)
            }
        }
    }

    /** Samples the RGBA frame (with rotation and resize mode) into [inputBuffer]. */
    private fun fillInput(image: ImageProxy, rotation: Int) {
        val plane = image.planes[0]
        val source = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val srcWidth = image.width
        val srcHeight = image.height
        rebuildLookupTables(srcWidth, srcHeight, rowStride, pixelStride, rotation)

        val width = input.width
        val height = input.height
        val channels = input.channels
        val out = inputBuffer
        out.clear()

        val mean = config.mean ?: FloatArray(4)
        val std = config.std ?: FloatArray(4) { 255f }
        val invStd = FloatArray(4) { i -> 1f / (std.getOrElse(i) { std.last() }.takeIf { it != 0f } ?: 1f) }
        val meanC = FloatArray(4) { i -> mean.getOrElse(i) { mean.last() } }

        when (input.dataType) {
            DataType.UINT8, DataType.INT8 -> {
                val signedOffset = if (input.dataType == DataType.INT8) -128 else 0
                var index = 0
                for (y in 0 until height) {
                    val rowOffset = rowOffsets[y]
                    for (x in 0 until width) {
                        val colOffset = colOffsets[x]
                        if (rowOffset == PADDING || colOffset == PADDING) {
                            for (c in 0 until channels) out.put(index++, signedOffset.toByte())
                            continue
                        }
                        val offset = rowOffset + colOffset
                        val r = source.get(offset).toInt() and 0xFF
                        val g = source.get(offset + 1).toInt() and 0xFF
                        val b = source.get(offset + 2).toInt() and 0xFF
                        when (channels) {
                            1 -> out.put(index++, (luminance(r, g, b) + signedOffset).toByte())
                            3 -> {
                                out.put(index++, (r + signedOffset).toByte())
                                out.put(index++, (g + signedOffset).toByte())
                                out.put(index++, (b + signedOffset).toByte())
                            }
                            else -> {
                                out.put(index++, (r + signedOffset).toByte())
                                out.put(index++, (g + signedOffset).toByte())
                                out.put(index++, (b + signedOffset).toByte())
                                out.put(index++, ((source.get(offset + 3).toInt() and 0xFF) + signedOffset).toByte())
                            }
                        }
                    }
                }
            }
            DataType.FLOAT32 -> {
                val floats = out.asFloatBuffer()
                var index = 0
                for (y in 0 until height) {
                    val rowOffset = rowOffsets[y]
                    for (x in 0 until width) {
                        val colOffset = colOffsets[x]
                        if (rowOffset == PADDING || colOffset == PADDING) {
                            for (c in 0 until channels) floats.put(index++, (0f - meanC[c]) * invStd[c])
                            continue
                        }
                        val offset = rowOffset + colOffset
                        val r = (source.get(offset).toInt() and 0xFF).toFloat()
                        val g = (source.get(offset + 1).toInt() and 0xFF).toFloat()
                        val b = (source.get(offset + 2).toInt() and 0xFF).toFloat()
                        when (channels) {
                            1 -> floats.put(index++, (luminance(r.toInt(), g.toInt(), b.toInt()) - meanC[0]) * invStd[0])
                            3 -> {
                                floats.put(index++, (r - meanC[0]) * invStd[0])
                                floats.put(index++, (g - meanC[1]) * invStd[1])
                                floats.put(index++, (b - meanC[2]) * invStd[2])
                            }
                            else -> {
                                floats.put(index++, (r - meanC[0]) * invStd[0])
                                floats.put(index++, (g - meanC[1]) * invStd[1])
                                floats.put(index++, (b - meanC[2]) * invStd[2])
                                val a = (source.get(offset + 3).toInt() and 0xFF).toFloat()
                                floats.put(index++, (a - meanC[3]) * invStd[3])
                            }
                        }
                    }
                }
            }
            else -> throw NativeMlException(
                "UNSUPPORTED_INPUT",
                "Camera input '${input.name}' has type ${input.dataType}; only uint8, int8 and float32 image tensors are supported"
            )
        }
        out.rewind()
    }

    private fun luminance(r: Int, g: Int, b: Int): Int = (r * 299 + g * 587 + b * 114) / 1000

    /**
     * Builds per-row / per-column byte offsets so each target pixel maps to one source pixel,
     * accounting for the sensor rotation and the configured resize mode.
     */
    private fun rebuildLookupTables(srcWidth: Int, srcHeight: Int, rowStride: Int, pixelStride: Int, rotation: Int) {
        val key = "$srcWidth,$srcHeight,$rowStride,$pixelStride,$rotation"
        if (key == lutKey) return
        lutKey = key
        val swap = rotation == 90 || rotation == 270
        uprightWidth = if (swap) srcHeight else srcWidth
        uprightHeight = if (swap) srcWidth else srcHeight

        val targetWidth = input.width
        val targetHeight = input.height
        // Map target pixel -> upright pixel (or PADDING) along each axis.
        val uprightX = IntArray(targetWidth)
        val uprightY = IntArray(targetHeight)
        when (config.resizeMode) {
            "fill" -> {
                for (x in 0 until targetWidth) uprightX[x] = ((x + 0.5) * uprightWidth / targetWidth).toInt().coerceIn(0, uprightWidth - 1)
                for (y in 0 until targetHeight) uprightY[y] = ((y + 0.5) * uprightHeight / targetHeight).toInt().coerceIn(0, uprightHeight - 1)
            }
            "contain" -> {
                val scale = minOf(targetWidth.toDouble() / uprightWidth, targetHeight.toDouble() / uprightHeight)
                val drawnWidth = uprightWidth * scale
                val drawnHeight = uprightHeight * scale
                val offsetX = (targetWidth - drawnWidth) / 2
                val offsetY = (targetHeight - drawnHeight) / 2
                for (x in 0 until targetWidth) {
                    val ux = (x + 0.5 - offsetX) / scale
                    uprightX[x] = if (ux < 0 || ux >= uprightWidth) PADDING else ux.toInt()
                }
                for (y in 0 until targetHeight) {
                    val uy = (y + 0.5 - offsetY) / scale
                    uprightY[y] = if (uy < 0 || uy >= uprightHeight) PADDING else uy.toInt()
                }
            }
            else -> { // cover (centre crop)
                val scale = maxOf(targetWidth.toDouble() / uprightWidth, targetHeight.toDouble() / uprightHeight)
                val cropX = (uprightWidth - targetWidth / scale) / 2
                val cropY = (uprightHeight - targetHeight / scale) / 2
                for (x in 0 until targetWidth) uprightX[x] = (cropX + (x + 0.5) / scale).toInt().coerceIn(0, uprightWidth - 1)
                for (y in 0 until targetHeight) uprightY[y] = (cropY + (y + 0.5) / scale).toInt().coerceIn(0, uprightHeight - 1)
            }
        }

        // Upright pixel -> source byte offset, split into a row part and a column part so the
        // inner loop is a single addition. Rotation is clockwise, as reported by CameraX.
        colOffsets = IntArray(targetWidth) { x ->
            val ux = uprightX[x]
            if (ux == PADDING) PADDING else when (rotation) {
                90 -> (srcHeight - 1 - ux) * rowStride
                180 -> (srcWidth - 1 - ux) * pixelStride
                270 -> ux * rowStride
                else -> ux * pixelStride
            }
        }
        rowOffsets = IntArray(targetHeight) { y ->
            val uy = uprightY[y]
            if (uy == PADDING) PADDING else when (rotation) {
                90 -> uy * pixelStride
                180 -> (srcHeight - 1 - uy) * rowStride
                270 -> (srcWidth - 1 - uy) * pixelStride
                else -> uy * rowStride
            }
        }
    }
}
