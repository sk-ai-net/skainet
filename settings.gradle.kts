pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "skainet"
include(":skainet-graph")
//include(":core")
include(":io")
include(":gguf")
include(":safetensors")
