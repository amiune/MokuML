# Packaging MokuPredictor

The packaging flow creates a normal macOS `.app` bundle and wraps it in a `.dmg`.

The app bundle includes:

- The compiled SwiftUI app executable.
- The Python worker code in `Contents/Resources/worker`.
- A relocatable Python 3.11 environment in `Contents/Resources/python`.
- A bundled Python runtime in `Contents/Resources/python-runtime`.
- Postgres, SQL validation, FastAPI, pandas, scikit-learn, and TabPFN dependencies.

## Local DMG

```bash
cd /Users/hernanamiune/Documents/MokuML
scripts/package_app.sh
```

Output:

```text
dist/MokuPredictor.app
dist/MokuPredictor-0.1.0.dmg
```

The default build uses ad-hoc signing. It is good for local testing, but not for broad distribution.

## Fast UI-Only Package

For packaging layout tests without the large Python dependency bundle:

```bash
MOKU_SKIP_PYTHON_BUNDLE=1 scripts/package_app.sh
```

## Production Signing And Notarization

Create a Developer ID signed app:

```bash
MOKU_SIGN_IDENTITY="Developer ID Application: Your Company (TEAMID)" \
scripts/package_app.sh
```

Create and notarize a DMG using an `xcrun notarytool` keychain profile:

```bash
MOKU_SIGN_IDENTITY="Developer ID Application: Your Company (TEAMID)" \
MOKU_NOTARY_PROFILE="notarytool-profile-name" \
scripts/package_app.sh
```

Create the notary profile once with:

```bash
xcrun notarytool store-credentials notarytool-profile-name \
  --apple-id "you@example.com" \
  --team-id "TEAMID" \
  --password "app-specific-password"
```

## Verification

```bash
codesign --verify --deep --strict dist/MokuPredictor.app
hdiutil verify dist/MokuPredictor-0.1.0.dmg
scripts/verify_packaged_worker.sh
```

`verify_packaged_worker.sh` checks that the bundled worker can import the core dependencies from inside the packaged app.

## Size

The current full package is large because it includes Python, scientific packages, Torch/MLX-related dependencies, and TabPFN:

```text
MokuPredictor.app: about 1.2 GB
MokuPredictor.dmg: about 519 MB
```

Future optimization can split TabPFN/model assets into a first-run download or an optional “full ML runtime” package.

