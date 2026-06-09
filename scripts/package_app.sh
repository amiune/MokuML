#!/usr/bin/env bash
set -euo pipefail

APP_NAME="MokuPredictor"
BUNDLE_ID="com.mokuml.MokuPredictor"
VERSION="${MOKU_VERSION:-0.1.0}"
CONFIGURATION="${MOKU_CONFIGURATION:-release}"
SIGN_IDENTITY="${MOKU_SIGN_IDENTITY:--}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT/dist"
BUILD_DIR="$DIST_DIR/build"
STAGING_DIR="$DIST_DIR/dmg-staging"
APP_PATH="$DIST_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_PATH/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
DMG_PATH="$DIST_DIR/$APP_NAME-$VERSION.dmg"

log() {
  printf '\n==> %s\n' "$1"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

require_command swift
require_command uv
require_command hdiutil
require_command codesign
require_command plutil
require_command rsync

log "Cleaning package output"
rm -rf "$APP_PATH" "$STAGING_DIR" "$BUILD_DIR" "$DMG_PATH"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR/home" "$BUILD_DIR/xdg-cache" "$BUILD_DIR/clang-module-cache" "$BUILD_DIR/uv-cache"

export HOME="$BUILD_DIR/home"
export XDG_CACHE_HOME="$BUILD_DIR/xdg-cache"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/clang-module-cache"
export UV_CACHE_DIR="$BUILD_DIR/uv-cache"

log "Building Swift app ($CONFIGURATION)"
swift build -c "$CONFIGURATION" --product "$APP_NAME"
BIN_DIR="$(swift build -c "$CONFIGURATION" --show-bin-path)"
cp "$BIN_DIR/$APP_NAME" "$MACOS_DIR/$APP_NAME"
chmod +x "$MACOS_DIR/$APP_NAME"

log "Writing app metadata"
cat > "$CONTENTS_DIR/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>
  <string>$BUNDLE_ID</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>$VERSION</string>
  <key>CFBundleVersion</key>
  <string>$VERSION</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>LSApplicationCategoryType</key>
  <string>public.app-category.business</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSSupportsAutomaticGraphicsSwitching</key>
  <true/>
</dict>
</plist>
PLIST
printf 'APPL????' > "$CONTENTS_DIR/PkgInfo"
plutil -lint "$CONTENTS_DIR/Info.plist"

log "Copying local worker"
rsync -a \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  "$ROOT/worker/" "$RESOURCES_DIR/worker/"

if [[ "${MOKU_SKIP_PYTHON_BUNDLE:-0}" == "1" ]]; then
  log "Skipping bundled Python because MOKU_SKIP_PYTHON_BUNDLE=1"
else
  log "Creating bundled Python environment"
  uv venv \
    --relocatable \
    --python 3.11 \
    --python-preference managed \
    --link-mode copy \
    "$RESOURCES_DIR/python"

  PYTHON_LINK_TARGET="$(readlink "$RESOURCES_DIR/python/bin/python")"
  if [[ "$PYTHON_LINK_TARGET" = /* ]]; then
    PYTHON_RUNTIME_SOURCE="$(cd "$(dirname "$PYTHON_LINK_TARGET")/.." && pwd)"
    log "Copying managed Python runtime into app bundle"
    rsync -a "$PYTHON_RUNTIME_SOURCE/" "$RESOURCES_DIR/python-runtime/"
    rm -f "$RESOURCES_DIR/python/bin/python" "$RESOURCES_DIR/python/bin/python3" "$RESOURCES_DIR/python/bin/python3.11"
    ln -s "../../python-runtime/bin/python3.11" "$RESOURCES_DIR/python/bin/python"
    ln -s "python" "$RESOURCES_DIR/python/bin/python3"
    ln -s "python" "$RESOURCES_DIR/python/bin/python3.11"
  fi

  log "Installing worker dependencies into bundled Python"
  uv pip install \
    --python "$RESOURCES_DIR/python/bin/python" \
    --link-mode copy \
    --compile-bytecode \
    "$RESOURCES_DIR/worker"

  log "Verifying bundled worker imports"
  PYTHONPATH="$RESOURCES_DIR/worker/src" "$RESOURCES_DIR/python/bin/python" -m worker.main --help >/dev/null
fi

if [[ "$SIGN_IDENTITY" == "-" ]]; then
  log "Ad-hoc signing app for local distribution"
  codesign --force --deep --sign "$SIGN_IDENTITY" "$APP_PATH"
else
  log "Signing app with Developer ID identity: $SIGN_IDENTITY"
  codesign --force --deep --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_PATH"
fi
codesign --verify --deep --strict "$APP_PATH"

log "Creating DMG"
mkdir -p "$STAGING_DIR"
cp -R "$APP_PATH" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

if [[ -n "${MOKU_NOTARY_PROFILE:-}" ]]; then
  require_command xcrun
  log "Submitting DMG for notarization with keychain profile: $MOKU_NOTARY_PROFILE"
  xcrun notarytool submit "$DMG_PATH" --keychain-profile "$MOKU_NOTARY_PROFILE" --wait
  log "Stapling notarization ticket"
  xcrun stapler staple "$DMG_PATH"
fi

log "Package complete"
printf '%s\n' "$DMG_PATH"
