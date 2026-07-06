plugins {
    // Χρησιμοποιούμε τα id κατευθείαν, παρακάμπτοντας τα "libs"
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.speculativediffusionv1"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.speculativediffusionv1"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters.add("arm64-v8a")
            abiFilters.add("x86_64")
        }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    // --- ΒΑΣΙΚΑ ANDROID DEPENDENCIES (Χωρίς libs) --- //
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.activity:activity-ktx:1.9.0")

    // --- ΤΑ ΔΙΚΑ ΜΑΣ DEPENDENCIES --- //

    // 1. ONNX Runtime (Η μηχανή για το Draft UNet)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    // 2. Kotlin Coroutines (Για δίκτυο και AI στο background)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // 3. Paho MQTT (Ο Client για τον Server)
    implementation("org.eclipse.paho:org.eclipse.paho.client.mqttv3:1.2.5")

    // --- TESTING --- //
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}