package com.example.flutter_native_ml;

import androidx.annotation.NonNull;
import androidx.lifecycle.Lifecycle;
import androidx.lifecycle.LifecycleOwner;

/**
 * Adapts a {@link Lifecycle} (for example the one provided by Flutter's
 * {@code FlutterLifecycleAdapter}) into a {@link LifecycleOwner} that CameraX can bind to.
 *
 * <p>Written in Java on purpose: {@code LifecycleOwner.getLifecycle()} is a Kotlin property in
 * androidx.lifecycle 2.6+ but a plain method in older versions, so a Java implementation compiles
 * against either.
 */
final class ProxyLifecycleOwner implements LifecycleOwner {
    private final Lifecycle lifecycle;

    ProxyLifecycleOwner(@NonNull Lifecycle lifecycle) {
        this.lifecycle = lifecycle;
    }

    @NonNull
    @Override
    public Lifecycle getLifecycle() {
        return lifecycle;
    }
}
