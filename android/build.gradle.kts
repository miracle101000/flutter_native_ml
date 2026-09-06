group = "com.example.flutter_native_ml"
version = "1.1.0"

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

// Flutter's Gradle plugin (3.35+) applies the Kotlin Gradle plugin to plugin
// projects automatically (see "built-in Kotlin" in the Flutter docs). Older
// Flutter versions do not, so apply it ourselves when it is missing and the
// host project has not opted into AGP's built-in Kotlin support.
val usesBuiltInKotlin = (findProperty("android.builtInKotlin") as? String)?.toBoolean() ?: false
if (!usesBuiltInKotlin &&
    !pluginManager.hasPlugin("org.jetbrains.kotlin.android") &&
    !pluginManager.hasPlugin("kotlin-android")
) {
    pluginManager.apply("org.jetbrains.kotlin.android")
}

// LiteRT (the successor of TensorFlow Lite) still ships the classic
// `org.tensorflow.lite.*` Interpreter API. Apps can override the version with
// `flutter_native_ml.litertVersion=<version>` in their gradle.properties.
val litertVersion = (findProperty("flutter_native_ml.litertVersion") as? String) ?: "1.4.2"

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
        // LiteRT requires API 21+. Flutter itself requires a higher minSdk.
        minSdk = 21
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
// of which JDK runs Gradle. Uses the task API so it works whether Kotlin was
// applied by this script, by Flutter, or through AGP's built-in support.
tasks.withType(org.jetbrains.kotlin.gradle.tasks.KotlinCompile::class.java).configureEach {
    compilerOptions.jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:$litertVersion")
    implementation("com.google.ai.edge.litert:litert-api:$litertVersion")
    implementation("com.google.ai.edge.litert:litert-gpu:$litertVersion")
    implementation("com.google.ai.edge.litert:litert-gpu-api:$litertVersion")

    testImplementation("org.jetbrains.kotlin:kotlin-test")
    testImplementation("org.mockito:mockito-core:5.0.0")
}
