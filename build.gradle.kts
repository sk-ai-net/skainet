import org.jetbrains.dokka.gradle.DokkaMultiModuleTask

plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.binaryCompatibility) apply false
    alias(libs.plugins.dokka) apply false
    alias(libs.plugins.modulegraph.souza) apply true
}

apply(plugin = "org.jetbrains.dokka")

allprojects {
    group = "sk.ai.net"
    version = "0.0.6-SNAPSHOT"
}

moduleGraphConfig {
    readmePath.set("./Modules.md")
    heading = "### Module Graph"
}

tasks.register<org.jetbrains.dokka.gradle.DokkaMultiModuleTask>("dokkaHtmlMultiModule") {
    outputDirectory.set(buildDir.resolve("dokka"))
}
