import com.vanniktech.maven.publish.MavenPublishBaseExtension
import com.vanniktech.maven.publish.SonatypeHost
import java.net.URI

plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.binaryCompatibility) apply false
    alias(libs.plugins.modulegraph.souza) apply true
    alias(libs.plugins.mavenPublish) apply false
}

allprojects {
    version = "0.0.5"
    group = "sk.ai.net"

    repositories {
        google()
        mavenCentral()
    }
}

allprojects {
    plugins.withId("com.vanniktech.maven.publish.base") {
        configure<MavenPublishBaseExtension> {
            repositories {
                maven {
                    name = "GitHubPackages"
                    url = uri("https://maven.pkg.github.com/sk-ai-net/skainet")
                    credentials {
                        username = System.getenv("GITHUB_ACTOR")
                        password = System.getenv("GITHUB_TOKEN")
                    }
                }
            }
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
    }
}

moduleGraphConfig {
    readmePath.set("./Modules.md")
    heading = "### Module Graph"
}
