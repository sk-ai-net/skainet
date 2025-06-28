@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
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

    jvm()
    androidTarget {
        publishLibraryVariants("release")
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_1_8)
        }
    }
    iosX64()
    iosArm64()
    iosSimulatorArm64()
    wasmJs().nodejs()
    macosX64 ()
    linuxX64 ()

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(project(":core"))
                implementation(libs.kotlinx.io.core)

                implementation(libs.kotlinx.serialization.json)
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlinx.coroutines)
            }
        }

    }
}

android {
    namespace = "sk.ai.net.io"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
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

    coordinates(group.toString(), "io", version.toString())

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