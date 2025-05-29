import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    // Temporarily disable Kover plugin to fix wasmJs build issue
    // alias(libs.plugins.kover)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.vanniktech.mavenPublish)
}


kotlin {
    targets.configureEach {
        compilations.configureEach {
            compileTaskProvider.get().compilerOptions {
                freeCompilerArgs.add("-Xexpect-actual-classes")
            }
        }
    }


    androidTarget {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    listOf(
        iosX64(),
        iosArm64(),
        iosSimulatorArm64()
    ).forEach { iosTarget ->
        iosTarget.binaries.framework {
            baseName = "safetensor"
            isStatic = true
        }
    }

    macosX64 ()
    linuxX64 ()


    jvm()

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        moduleName = "safetensor"
        browser()
        binaries.executable()

        // Configure Wasm target to handle dependencies properly
        // This helps resolve issues with duplicate 'unique_name=kotlin'
        nodejs()

        // Configure browser distribution
        browser {
            commonWebpackConfig {
                // Configure webpack to handle kotlinx-io properly
                cssSupport {
                    enabled.set(true)
                }
                // Use proper configuration for dependency resolution
                export = true
            }
        }
    }

    sourceSets {
        val jvmMain by getting
        val wasmJsMain by getting

        androidMain.dependencies {
            // Android-specific dependencies if needed
        }
        commonMain.dependencies {
            // Common dependencies
            implementation(project(":skainet-graph"))
            implementation(libs.kotlinx.serialization.json)
            // Use a newer version of kotlinx-io that's compatible with Kotlin 2.1.21
            implementation(libs.kotlinx.io.core)
        }
        commonTest {
            resources.srcDirs("src/commonMain/resources")
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
        jvmMain.dependencies {
            // Desktop-specific dependencies if needed
        }
        wasmJsMain.dependencies {
            // Wasm-specific dependencies
            // Explicitly specify the version for Wasm to avoid conflicts
            implementation(libs.kotlinx.io.core.wasm.js)
        }

        // Make sure resources are available for all test targets
        val iosX64Test by getting {
            resources.srcDirs("src/commonMain/resources")
        }
        val iosArm64Test by getting {
            resources.srcDirs("src/commonMain/resources")
        }
        val iosSimulatorArm64Test by getting {
            resources.srcDirs("src/commonMain/resources")
        }
        val wasmJsTest by getting {
            resources.srcDirs("src/commonMain/resources")
        }
        val jvmTest by getting {
            resources.srcDirs("src/commonMain/resources")
        }
        val androidUnitTest by getting {
            resources.srcDirs("src/commonMain/resources")
        }
    }
}

android {
    namespace = "sk.ai.net.safetensor"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

// Set duplicatesStrategy for all resource processing tasks
tasks.withType<ProcessResources> {
    duplicatesStrategy = DuplicatesStrategy.INCLUDE
}

publishing {
    repositories {
        maven {
            name = "githubPackages"
            url = uri("https://maven.pkg.github.com/sk-ai-net/skainet")
            credentials {
                credentials(PasswordCredentials::class)
            }
        }
    }
}

mavenPublishing {

    coordinates(group.toString(), "safetensors", version.toString())

    pom {
        description.set("skainet")
        name.set(project.name)
        url.set("https://github.com/sk-ai-net/skainet/")
        licenses {
            license {
                name.set("MIT")
                distribution.set("repo")
            }
        }
        scm {
            url.set("https://github.com/sk-ai-net/skainet/")
            connection.set("scm:git:git@github.com:sk-ai-net/skainet.git")
            developerConnection.set("scm:git:ssh://git@github.com:sk-ai-net/skainet.git")
        }
        developers {
            developer {
                id.set("sk-ai-net")
                name.set("sk-ai-net")
            }
        }
    }
}
