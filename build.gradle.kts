//import org.jetbrains.dokka.gradle.DokkaMultiModuleTask

plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.binaryCompatibility) apply false
    //alias(libs.plugins.dokka) apply false
    alias(libs.plugins.modulegraph.souza) apply true
}

//apply(plugin = "org.jetbrains.dokka")

allprojects {
    group = "sk.ai.net"
    version = "0.0.6-SNAPSHOT"
}

// Task to run all tests in the project
tasks.register("allTests") {
    group = "verification"
    description = "Runs all tests in the project"

    // Depend on all test tasks from all subprojects
    dependsOn(subprojects.map { it.tasks.matching { task -> task.name.contains("test", ignoreCase = true) && task.name.contains("compile", ignoreCase = true).not() } })
}

moduleGraphConfig {
    readmePath.set("./Modules.md")
    heading = "### Module Graph"
}

/*
tasks.register<org.jetbrains.dokka.gradle.DokkaMultiModuleTask>("dokkaHtmlMultiModule") {
    outputDirectory.set(buildDir.resolve("dokka"))
}

 */
