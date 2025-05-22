# KDoc preprocessing

This project follows the same approach as [Kotlin/DataFrame](https://github.com/Kotlin/dataframe/blob/master/KDOC_PREPROCESSING.md) for preprocessing KDoc comments before generating documentation with Dokka.

The Dokka plugin is applied to all modules. Run `./gradlew dokkaHtml` to generate HTML documentation under each module's `build/dokka` directory.
