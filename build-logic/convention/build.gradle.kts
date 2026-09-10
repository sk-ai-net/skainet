plugins {
    `kotlin-dsl`
    kotlin("jvm") version "2.4.20"
    kotlin("plugin.serialization") version "2.4.10"
}

repositories {
    google()
    gradlePluginPortal()
    mavenCentral()
}

group = "sk.ainet.buildlogic"

dependencies {
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.optimumcode.json.schema.validator)
    implementation(libs.asciidoctorj.core)
    implementation("org.jetbrains.dokka:dokka-gradle-plugin:2.2.0")
    implementation(gradleApi())

    // compileOnly, never implementation: build-logic is an included build, so putting these
    // on the plugin's runtime classpath would load a second copy of KGP/AGP in a different
    // classloader and turn every `getByType(SomeKgpType::class)` into a ClassCastException.
    // compileOnly gives us the types while deferring to the classes the consuming build loads.
    compileOnly(libs.kotlin.gradlePlugin)
    compileOnly(libs.android.gradlePlugin)
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(21)
}

gradlePlugin {
    plugins {
        register("SKaiNetDocumentation") {
            id = "sk.ainet.documentation"
            implementationClass = "DocumentationPlugin"
        }
        register("SKaiNetBomCoverage") {
            id = "sk.ainet.transformers.bom-coverage"
            implementationClass = "sk.ainet.buildlogic.bom.BomCoveragePlugin"
        }
        register("SKaiNetMultiplatform") {
            id = "sk.ainet.multiplatform"
            implementationClass = "sk.ainet.buildlogic.kmp.SkainetMultiplatformPlugin"
        }
        register("SKaiNetNpmPins") {
            id = "sk.ainet.npm-pins"
            implementationClass = "sk.ainet.buildlogic.npm.NpmPinsPlugin"
        }
        register("SKaiNetMavenPins") {
            id = "sk.ainet.maven-pins"
            implementationClass = "sk.ainet.buildlogic.maven.MavenPinsPlugin"
        }
    }
}
