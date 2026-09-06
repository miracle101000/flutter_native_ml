import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_native_ml_example/main.dart';

void main() {
  testWidgets('shows the load button', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());
    await tester.pump();

    expect(find.text('1. Load model'), findsOneWidget);
    expect(find.text('Flutter Native ML'), findsOneWidget);
  });
}
