# Example models

Drop your models in this folder; it is declared as an asset directory in
`pubspec.yaml`.

The example app looks for:

* `assets/models/model.tflite` on Android
* `assets/models/model.mlmodel` on iOS (compiled on device by the plugin)

Generate them with the bundled CLI from the example directory, for example:

```bash
dart run flutter_native_ml:ml_builder -s path/to/my_keras_model.h5 -o assets/models/
```

You can also pick any model file on the device at runtime via the
"Load from file path" field in the app.
