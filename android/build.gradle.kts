group = "com.example.flutter_native_ml"
version = "1.2.2"

buildscript {
    val kotlinVersion = "2.1.0"
    repositories {
        google()
        mavenCentral()
    }

    dependencies {
        classpath("com.android.tools.build:gradle:8.7.3")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlinVersion")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    id("com.android.library")
}

// Kotlin support, in order of preference:
//  1. AGP 9+ built-in Kotlin (`android.builtInKotlin`, on by default since AGP 9):
//     AGP registers the `kotlin` extension itself and applying the Kotlin Gradle
//     plugin on top fails, so nothing to do.
//  2. Flutter 3.35+ with built-in Kotlin disabled: Flutter's tooling applies the
//     Kotlin Gradle plugin to plugin projects before this script runs.
//  3. Older Flutter versions: apply the Kotlin Gradle plugin ourselves.
val builtInKotlin = extensions.findByName("kotlin") != null ||
    ((findProperty("android.builtInKotlin") as? String)?.toBoolean()
        ?: (androidComponents.pluginVersion.major >= 9))
if (!builtInKotlin &&
    !pluginManager.hasPlugin("org.jetbrains.kotlin.android") &&
    !pluginManager.hasPlugin("kotlin-android")
) {
    pluginManager.apply("org.jetbrains.kotlin.android")
}

// LiteRT (the successor of TensorFlow Lite) still ships the classic
// `org.tensorflow.lite.*` Interpreter API. Apps can override the version with
// `flutter_native_ml.litertVersion=<version>` in their gradle.properties.
val litertVersion = (findProperty("flutter_native_ml.litertVersion") as? String) ?: "1.4.2"

// CameraX powers the zero-copy camera input. 1.5.x matches Flutter's minimum
// toolchain (AGP 8.6, compileSdk 35, minSdk 23).
val cameraxVersion = (findProperty("flutter_native_ml.cameraxVersion") as? String) ?: "1.5.3"

android {
    namespace = "com.example.flutter_native_ml"

    compileSdk = 35

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/kotlin")
        }
        getByName("test") {
            java.srcDirs("src/test/kotlin")
        }
    }

    defaultConfig {
        // CameraX 1.5 requires API 23 (LiteRT alone would work from 21).
        minSdk = 23
    }

    testOptions {
        unitTests {
            isReturnDefaultValues = true
            all {
                it.useJUnitPlatform()

                it.outputs.upToDateWhen { false }

                it.testLogging {
                    events("passed", "skipped", "failed", "standardOut", "standardError")
                    showStandardStreams = true
                }
            }
        }
    }
}

// Keep the Kotlin JVM target aligned with `compileOptions` above, regardless
// of which JDK runs Gradle. Uses the task API (rather than the `kotlin {}`
// extension, whose type-safe accessor only exists when the Kotlin plugin was
// applied before this script) so it works for all three cases above.
tasks.withType(org.jetbrains.kotlin.gradle.tasks.KotlinCompile::class.java).configureEach {
    compilerOptions.jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:$litertVersion")
    implementation("com.google.ai.edge.litert:litert-api:$litertVersion")
    implementation("com.google.ai.edge.litert:litert-gpu:$litertVersion")
    implementation("com.google.ai.edge.litert:litert-gpu-api:$litertVersion")

    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")

    // Pinned and framework-specific on purpose: the version-less `kotlin-test`
    // shorthand relies on the Kotlin Gradle plugin's test-framework
    // auto-selection, which AGP's built-in Kotlin does not perform.
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit5:2.1.0")
    testImplementation("org.mockito:mockito-core:5.0.0")
}
