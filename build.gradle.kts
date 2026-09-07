import sk.ainet.buildlogic.npm.NpmPinTarget

plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.androidMultiplatformLibrary) apply false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.vanniktech.mavenPublish) apply false
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator) apply false
    alias(libs.plugins.ksp) apply false
    alias(libs.plugins.asciidoctorJvm) apply false
    alias(libs.plugins.dokka)
    alias(libs.plugins.skainet.docs)
    alias(libs.plugins.skainet.npmPins)
    alias(libs.plugins.skainet.mavenPins)
    id("org.jetbrains.kotlinx.benchmark") version "0.4.19" apply false
}

allprojects {
    group = "sk.ainet.core"
    version = providers.gradleProperty("VERSION_NAME").getOrElse("unspecified")
}

// Root-project SKaiNET conventions. npm pins are forced onto the kotlin-js-store
// lockfiles via Yarn resolutions; see docs "Pinning npm Packages".
//
// `ws` is the only pinned package present in both dependency graphs, so it is the only
// unscoped pin. Everything else lives solely in the Kotlin/JS graph and is scoped to it:
// Yarn writes a lockfile entry for every resolutions key whether or not that graph
// requests the package, so an unscoped pin would add the package — and, for webpack, its
// ~76 transitive dependencies — to kotlin-js-store/wasm/yarn.lock for nothing.
skainet {
    npmPins {
        pin("ws", libs.versions.npm.ws)
        pin("js-yaml", libs.versions.npm.js.yaml, NpmPinTarget.JS)
        pin("socket.io-parser", libs.versions.npm.socketio.parser, NpmPinTarget.JS)
        pin("fast-uri", libs.versions.npm.fast.uri, NpmPinTarget.JS)
        pin("serialize-javascript", libs.versions.npm.serialize.javascript, NpmPinTarget.JS)
        pin("qs", libs.versions.npm.qs, NpmPinTarget.JS)
        pin("brace-expansion", libs.versions.npm.brace.expansion, NpmPinTarget.JS)
        pin("diff", libs.versions.npm.diff, NpmPinTarget.JS)
        pin("webpack", libs.versions.npm.webpack, NpmPinTarget.JS)
    }
}

// Require JDK 21+ but allow any newer version (produces Java 21 bytecode via --release / jvmTarget)
subprojects {
    require(JavaVersion.current() >= JavaVersion.VERSION_21) {
        "This project requires JDK 21+, but found ${JavaVersion.current()}"
    }

    // Kotlin Multiplatform projects – set jvmTarget on every JVM-like target
    plugins.withId("org.jetbrains.kotlin.multiplatform") {
        extensions.findByType(org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension::class.java)?.apply {
            targets.withType(org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget::class.java) {
                compilerOptions {
                    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
                }
            }
        }
    }
    // Kotlin/JVM projects
    plugins.withId("org.jetbrains.kotlin.jvm") {
        extensions.findByType(org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension::class.java)?.apply {
            compilerOptions {
                jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
            }
        }
    }

    // Java sources – produce Java 21 bytecode regardless of the JDK used to compile.
    // Skip Android projects: AGP manages source/target compatibility itself and rejects --release.
    afterEvaluate {
        if (!plugins.hasPlugin("com.android.library") && !plugins.hasPlugin("com.android.application") && !plugins.hasPlugin("com.android.kotlin.multiplatform.library")) {
            tasks.withType<JavaCompile>().configureEach {
                options.release.set(21)
            }
        }
    }

    tasks.withType<Test>().configureEach {
        maxHeapSize = "8192m"
    }

    apply(plugin = "org.jetbrains.kotlinx.kover")
}

kover {
    reports {
        total {
            html {
                onCheck = true
            }
            xml {
                onCheck = true
            }
        }
    }
}

dependencies {
    subprojects.forEach {
        kover(it)
    }
}

// Custom task to generate operator documentation
tasks.register("generateOperatorDocs") {
    group = "documentation"
    description = "Generate operator documentation from KSP-generated JSON files"
    
    // Configure inputs for incremental builds
    inputs.files("skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json")
    // Configure outputs for incremental builds
    outputs.dir("docs/modules/operators/_generated_")
    outputs.cacheIf { true }
    
    // Depend on KSP processing
    dependsOn(":skainet-lang:skainet-lang-core:kspCommonMainKotlinMetadata")
    
    // Run built-in documentation generation task (provided by sk.ainet.documentation plugin)
    dependsOn("generateDocs")
    
    doLast {
        println("Operator documentation generation completed")
    }
}

// Documentation plugin configuration — emits operator doc fragments
// into the Antora ROOT module so the published site can surface them
// under Reference > Operator coverage.
documentation {
    inputFile.set(file("skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json"))
    outputDirectory.set(file("docs/modules/ROOT/pages/reference/operators/generated"))
    includeBackendStatus.set(true)
    generateIndex.set(true)
    // Kernel × platform matrix — registry-introspected JSON (emitted by KernelSupportMatrixTest)
    // rendered to an Antora page, mirroring the operators.json → ops-status-matrix.adoc pipeline.
    kernelInputFile.set(file("skainet-backends/skainet-backend-native-cpu/build/generated/kernel-support/kernel-support.json"))
    kernelOutputFile.set(file("docs/modules/ROOT/pages/reference/kernel-support-matrix.adoc"))
}

tasks.named("generateDocs") {
    dependsOn(":skainet-lang:skainet-lang-core:kspCommonMainKotlinMetadata")
}

tasks.named("generateKernelMatrix") {
    // The generator test introspects the registered providers + emits kernel-support.json.
    dependsOn(":skainet-backends:skainet-backend-native-cpu:jvmTest")
}

// Dokka aggregation – unified API reference across all library modules
dokka {
    moduleName.set("SKaiNET")
    dokkaPublications.html {
        includes.from("README.md")
    }
}

dependencies {
    // skainet-lang
    dokka(project(":skainet-lang:skainet-lang-core"))
    dokka(project(":skainet-lang:skainet-lang-models"))
    dokka(project(":skainet-lang:skainet-lang-ksp-annotations"))
    dokka(project(":skainet-lang:skainet-lang-dag"))

    // skainet-compile
    dokka(project(":skainet-compile:skainet-compile-core"))
    dokka(project(":skainet-compile:skainet-compile-dag"))
    dokka(project(":skainet-compile:skainet-compile-json"))
    dokka(project(":skainet-compile:skainet-compile-hlo"))
    dokka(project(":skainet-compile:skainet-compile-c"))

    // skainet-backends
    dokka(project(":skainet-backends:skainet-backend-cpu"))

    // skainet-data
    dokka(project(":skainet-data:skainet-data-api"))
    dokka(project(":skainet-data:skainet-data-source"))
    dokka(project(":skainet-data:skainet-data-transform"))
    dokka(project(":skainet-data:skainet-data-simple"))
    dokka(project(":skainet-data:skainet-data-media"))

    // skainet-io
    dokka(project(":skainet-io:skainet-io-core"))
    dokka(project(":skainet-io:skainet-io-gguf"))
    dokka(project(":skainet-io:skainet-io-image"))
    dokka(project(":skainet-io:skainet-io-onnx"))
    dokka(project(":skainet-io:skainet-io-safetensors"))

    // Other
    dokka(project(":skainet-pipeline"))
    dokka(project(":skainet-models:skainet-model-yolo"))
}

// Copy the Dokka-generated HTML aggregate into the Antora site
// output as a sibling `/api/` path. Invoked by .github/workflows/docs.yml
// AFTER Antora has populated `docs/build/site/`; intentionally NOT
// wired into the `build` lifecycle so that running `./gradlew build`
// locally never silently creates a half-populated site directory.
tasks.register<Copy>("bundleDokkaIntoSite") {
    group = "documentation"
    description = "Copy build/dokka/html into docs/build/site/api for GitHub Pages publish"
    dependsOn("dokkaGenerate")
    from(layout.buildDirectory.dir("dokka/html"))
    into(layout.projectDirectory.dir("docs/build/site/api"))
}
