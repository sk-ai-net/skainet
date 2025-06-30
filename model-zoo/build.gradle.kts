import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    id("com.android.library")
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
        compilations.all {

        }
    }

    wasmJs {
        browser()
    }

    iosX64()
    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(project(":core"))
                implementation(libs.kotlinx.io.core)
                implementation(libs.kotlinx.serialization.json)
                implementation(libs.ktor.client.core)
                implementation(libs.kotlinx.coroutines)
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }

        val jvmMain by getting {
            dependencies {
                implementation(libs.ktor.client.cio)
                implementation(libs.ktor.client.plugins)
                implementation(libs.ktor.client.logging)
                implementation(libs.ktor.client.content.negotiation)
                implementation(libs.kotlinx.coroutines.core.jvm)
                implementation(libs.logback.classic) // For logging
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlinx.coroutines)
            }
        }

        val androidMain by getting {
            dependencies {
                implementation("io.ktor:ktor-client-android:3.2.0")
            }
        }

        val wasmJsMain by getting {
            dependencies {
                implementation("io.ktor:ktor-client-js:3.2.0")
            }
        }

        val iosMain by creating {
            dependencies {
                implementation("io.ktor:ktor-client-darwin:3.2.0")
            }
        }
    }
}

android {
    namespace = "sk.ai.net.model.zoo"
    compileSdk = 34
    defaultConfig {
        minSdk = 24
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

    coordinates(group.toString(), "model-zoo", version.toString())

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
