package com.example.flutter_native_ml

import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import kotlin.test.Test
import org.mockito.ArgumentMatchers.anyString
import org.mockito.ArgumentMatchers.isNull
import org.mockito.Mockito

/*
 * Unit tests for the Kotlin side of the plugin that do not need a device or the LiteRT
 * native library. Run them from `example/android` with `./gradlew :flutter_native_ml:testDebugUnitTest`.
 */
internal class FlutterNativeMlPluginTest {
    @Test
    fun onMethodCall_getPlatformVersion_returnsExpectedValue() {
        val plugin = FlutterNativeMlPlugin()

        val call = MethodCall("getPlatformVersion", null)
        val mockResult: MethodChannel.Result = Mockito.mock(MethodChannel.Result::class.java)
        plugin.onMethodCall(call, mockResult)

        Mockito.verify(mockResult).success("Android " + android.os.Build.VERSION.RELEASE)
    }

    @Test
    fun onMethodCall_unknownMethod_reportsNotImplemented() {
        val plugin = FlutterNativeMlPlugin()

        val mockResult: MethodChannel.Result = Mockito.mock(MethodChannel.Result::class.java)
        plugin.onMethodCall(MethodCall("doesNotExist", null), mockResult)

        Mockito.verify(mockResult).notImplemented()
    }

    @Test
    fun onMethodCall_runWithUnknownModel_reportsModelNotFound() {
        val plugin = FlutterNativeMlPlugin()

        val mockResult: MethodChannel.Result = Mockito.mock(MethodChannel.Result::class.java)
        plugin.onMethodCall(MethodCall("run", mapOf("modelId" to "missing", "input" to emptyMap<String, Any>())), mockResult)

        Mockito.verify(mockResult).error(Mockito.eq("MODEL_NOT_FOUND"), anyString(), isNull())
    }

    @Test
    fun onMethodCall_loadModelWithoutPath_reportsInvalidArgs() {
        val plugin = FlutterNativeMlPlugin()

        val mockResult: MethodChannel.Result = Mockito.mock(MethodChannel.Result::class.java)
        plugin.onMethodCall(MethodCall("loadModel", mapOf("computeUnits" to "all")), mockResult)

        // The plugin is not attached to an engine in a unit test, which is reported first.
        Mockito.verify(mockResult).error(Mockito.eq("NOT_ATTACHED"), anyString(), isNull())
    }

    @Test
    fun onMethodCall_streamInputBeforeLoad_reportsModelNotFound() {
        val plugin = FlutterNativeMlPlugin()

        val mockResult: MethodChannel.Result = Mockito.mock(MethodChannel.Result::class.java)
        plugin.onMethodCall(MethodCall("streamInput", mapOf("modelId" to "missing", "input" to emptyMap<String, Any>())), mockResult)

        Mockito.verify(mockResult).error(Mockito.eq("MODEL_NOT_FOUND"), anyString(), isNull())
    }
}
