[app]

# (str) Title of your application
title = FaceApp Attendance

# (str) Package name
package.name = com.yourcompany.faceapp

# (str) Package domain (needed for android/ios packaging)
package.domain = yourcompany.com

# (str) Application versioning (method 1)
version = 0.1

# (list) Source files to include (let empty to include all the files
# in the current directory)
source.include_exts = py,png,jpg,mp3,wav,kv,xml,json

# (list) List of inclusions using pattern matching
# This option allows to select which files to include in the apk
# by matching them against a list of patterns. Default to ['**']
# (everything)
# source.include_patterns = assets/*,images/*.png

# (list) List of exclusions using pattern matching
# This option allows to select which files to exclude in the apk
# by matching them against a list of patterns. Default to []
# source.exclude_patterns = .git/*,.buildozer/*

# (list) Application requirements
# comma separated list of packages
# These will be installed by pip in the target machine
requirements = python3,kivy,opencv,numpy,requests,setuptools,pyjnius,android,certifi

# (str) Kivy version to use
kivy.version = 2.3.0

# (str) Android API level to use
android.api = 33

# (str) Minimum Android API level required
android.minapi = 21

# (str) Android target SDK version
android.targetsdk = 33

# (list) Android architecture to build for
# Available architectures: armeabi-v7a, arm64-v8a, x86, x86_64
android.archs = arm64-v8a, armeabi-v7a

# (list) Android permissions
android.permissions = INTERNET, CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (str) Icon file to use. This should be a .png file.
# icon.filename = %(source.dir)s/icon.png
# If you have a specific icon, uncomment the line above and provide the path.
# Otherwise, Buildozer will use a default Kivy icon.

# (str) A string that specifies the orientation of the screen.
# Can be one of 'landscape', 'portrait', 'sensor' (default), 'all'.
orientation = portrait

# (list) Python modules to exclude from the build (for size optimization)
# python.exclude_modules = _ssl, _hashlib, _json, _socket

# (list) Libraries to include (relative to the project root)
# android.add_libs = lib/mylib.so

# (int) The version number of your application.
# This is an integer value that is incremented with each release.
# It is used internally by Android to determine if a new version is available.
android.version_code = 1

# (str) The name of the main file of your application.
main.py = main.py
# If your main file is named 'face_recognition.py', change the line above to:
# main.py = face_recognition.py

# (list) Extra setup.py arguments to pass to the build process
# android.extra_setup_args = --enable-gstreamer

# (list) Extra Buildozer arguments to pass to the build process
# buildozer.extra_args = --clean

# (str) The URL of the Android SDK.
# android.sdk = https://dl.google.com/android/repository/commandlinetools-linux-6609375_latest.zip

# (str) The URL of the Android NDK.
# android.ndk = https://dl.google.com/android/repository/android-ndk-r25b-linux.zip

# (str) The URL of the Python for Android toolchain.
# android.p4a_branch = master

# (str) The Java version to use for Android builds.
# android.java_version = 17

# (bool) Enable/disable multi-dex support (for apps with many methods)
# android.enable_multidex = 0

# (list) Additional Java source files to include (relative to the project root)
# android.add_src = java_src

# (list) Additional Java libraries to include (relative to the project root)
# android.add_libs_armeabi-v7a = libs/armeabi-v7a/libfoo.so
# android.add_libs_arm64-v8a = libs/arm64-v8a/libfoo.so
