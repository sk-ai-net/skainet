package sk.ai.net.io.tokenizer

import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder


/**
 * Tokenizer model, loosely based on Huggingface's Tokenizer format.
 * This class also holds the prompt templates for chat models.
 *
 * @see [Huggingface Tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html)
 * @see [Chat Templating](https://huggingface.co/docs/transformers/main/en/chat_templating#templates-for-chat-models)
 * @see PromptSupport
 */
@Serializable
data class TokenizerModel(
    val type: String,
    val unkToken: String? = null,
    val fuseUnk: Boolean = false,
    val byteFallback: Boolean = false,
    val vocabLookup: Map<String, Long>,
    val ignoreMerges: Boolean = false,
    val merges: Map<String, Long> = emptyMap()
) {

    private var preTokenizer: PreTokenizer? = null
    private var normalizer: Normalizer? = null
    private val addedTokens: BiMap<String, Long> = BiMap()
    private val specialTokens: BiMap<String, Long> = BiMap()
    private var addedTokenPattern: Regex? = null
    private var legacy: Boolean = false
    private var promptTemplates: Map<String, String>? = null
    private var hasToolSupport: Boolean = false
    private var eosToken: String = ""
    private var bosToken: String = ""

    /** Returns the pre-tokenizer instance. */
    fun preTokenizer(): PreTokenizer? = preTokenizer

    /** Sets the pre-tokenizer and updates the legacy flag. */
    fun setPreTokenizer(preTokenizer: PreTokenizer?) {
        this.preTokenizer = preTokenizer
        this.legacy = preTokenizer?.isLegacy ?: false
    }

    /** Returns the normalizer instance. */
    fun normalizer(): Normalizer? = normalizer

    /** Sets the normalizer. */
    fun setNormalizer(normalizer: Normalizer?) {
        this.normalizer = normalizer
    }

    /** Sets the list of added tokens and updates related properties. */
    fun setAddedTokens(addedTokens: List<AddedToken>) {
        if (addedTokens.isNotEmpty()) {
            addedTokens.forEach { token ->
                this.addedTokens.put(token.content, token.id)
                if (token.special) {
                    this.specialTokens.put(token.content, token.id)
                }
            }
            this.addedTokenPattern = Regex(addedTokens.joinToString("|") { Regex.escape(it.content) })
        }
    }

    /** Returns whether merges should be ignored. */
    fun ignoreMerges(): Boolean = ignoreMerges

    /** Returns the map of added tokens. */
    fun addedTokens(): Map<String, Long> = addedTokens.asMap()

    /** Returns the pattern for added tokens. */
    fun addedTokenPattern(): Regex? = addedTokenPattern

    /** Returns whether the tokenizer is in legacy mode. */
    fun isLegacy(): Boolean = legacy

    /** Sets the legacy mode flag. */
    fun setLegacy(legacy: Boolean) {
        this.legacy = legacy
    }

    /** Returns the prompt templates. */
    fun promptTemplates(): Map<String, String>? = promptTemplates

    /** Sets the prompt templates and updates tool support flag. */
    fun setPromptTemplates(promptTemplates: Map<String, String>?) {
        this.promptTemplates = promptTemplates
        this.hasToolSupport = promptTemplates?.values?.any { it.contains("tools", ignoreCase = true) } ?: false
    }

    /** Returns whether the tokenizer supports tools. */
    fun hasToolSupport(): Boolean = hasToolSupport

    /** Sets the end-of-sequence token. */
    fun setEosToken(eosToken: String) {
        this.eosToken = eosToken
    }

    /** Returns the end-of-sequence token. */
    fun eosToken(): String = eosToken

    /** Sets the beginning-of-sequence token. */
    fun setBosToken(bosToken: String) {
        this.bosToken = bosToken
    }

    /** Returns the beginning-of-sequence token. */
    fun bosToken(): String = bosToken

    /** Checks if a token ID is a special token. */
    fun isSpecialToken(token: Long): Boolean = specialTokens.containsValue(token)

    /** Checks if a token string is a special token. */
    fun isSpecialToken(token: String): Boolean = specialTokens.containsKey(token)
}

/** Represents an added token with its content, ID, and special status. */
@Serializable
data class AddedToken(
    val content: String,
    val id: Long,
    val special: Boolean = false
)

/** Normalizer class to handle text normalization. */
@Serializable
data class Normalizer(
    val type: String,
    val normalizers: List<NormalizerItem> = emptyList()
) {
    /** Normalizes a sentence by applying a sequence of normalizers. */
    fun normalize(sentence: String): String {
        if (normalizers.isEmpty()) return sentence
        require(type.equals("Sequence", ignoreCase = true)) { "Invalid normalizer type: $type" }
        return normalizers.fold(sentence) { acc, item -> item.normalize(acc) }
    }
}

/** Individual normalization operation. */
@Serializable
data class NormalizerItem(
    val type: String,
    val prepend: String? = null,
    val pattern: Map<String, String>? = null,
    val content: String? = null
) {
    private companion object {
        val logger = LoggerFactory.getLogger(NormalizerItem::class.java)
    }

    /** Normalizes a sentence based on the type. */
    fun normalize(sentence: String): String = when (type) {
        "Replace" -> replace(sentence)
        "Prepend" -> prepend(sentence)
        "NFC", "NFKC", "NFD", "NFKD" -> formNormalize(sentence)
        else -> throw IllegalArgumentException("Invalid normalizer type: $type")
    }

    /** Applies Unicode normalization forms. */
    private fun formNormalize(sentence: String): String {
        val form = when (type) {
            "NFC" -> Normalizer.Form.NFC
            "NFKC" -> Normalizer.Form.NFKC
            "NFD" -> Normalizer.Form.NFD
            "NFKD" -> Normalizer.Form.NFKD
            else -> throw IllegalArgumentException("Invalid normalization form: $type")
        }
        return Normalizer.normalize(sentence, form)
    }

    /** Replaces patterns in the sentence. */
    private fun replace(sentence: String): String {
        var result = sentence
        pattern?.forEach { (key, value) ->
            if (key.equals("String", ignoreCase = true)) {
                result = result.replace(Regex(value), content ?: "")
            } else {
                logger.warn("Ignoring unknown pattern key: $key")
            }
        }
        return result
    }

    /** Prepends text to the sentence. */
    private fun prepend(sentence: String): String = (prepend ?: "") + sentence
}

/** Pre-tokenizer class to split text into tokens. */
@Serializable
data class PreTokenizer(
    val type: String,
    val replacement: String? = null,
    val prependScheme: String? = null,
    val pretokenizers: List<PretokenizerItem> = emptyList()
) {
    /** Indicates if the pre-tokenizer is legacy based on ByteLevel usage. */
    val isLegacy: Boolean = pretokenizers.any { it.type == "ByteLevel" }

    /** Pre-tokenizes a sentence into a list of strings. */
    fun pretokenize(sentence: String): List<String> {
        if (type.equals("MetaSpace", ignoreCase = true)) {
            val modifiedSentence = if (prependScheme?.equals("first", ignoreCase = true) == true) " $sentence" else sentence
            return listOf(modifiedSentence.replace(Regex("[ \t]+"), replacement ?: " "))
        }
        if (pretokenizers.isEmpty()) return listOf(sentence)
        require(type.equals("Sequence", ignoreCase = true)) { "Invalid pre-tokenizer type: $type" }
        return pretokenizers.fold(listOf(sentence)) { acc, item -> acc.flatMap { item.pretokenize(it) } }
    }
}

/** Individual pre-tokenization operation. */
@Serializable
data class PretokenizerItem(
    val type: String,
    val pattern: Pattern? = null,
    val behavior: String? = null,
    val invert: Boolean? = null,
    val individualDigits: Boolean? = null,
    val addPrefixSpace: Boolean? = null,
    val trimOffsets: Boolean? = null,
    val useRegex: Boolean? = null
) {
    /** Pre-tokenizes a sentence based on the type. */
    fun pretokenize(sentence: String): List<String> = when (type) {
        "Split" -> splitRegex(sentence)
        "Digits" -> splitDigits(sentence)
        "ByteLevel" -> byteLevel(sentence)
        else -> throw IllegalArgumentException("Invalid pre-tokenizer type: $type")
    }

    /** Handles ByteLevel pre-tokenization (simplified). */
    private fun byteLevel(sentence: String): List<String> {
        return listOf(sentence.codePoints()
            .map { BPETokenizer.alteredBytes.getOrDefault(it, it) }
            .joinToString("") { it.toString() })
    }

    /** Splits the sentence using a regex pattern. */
    private fun splitRegex(sentence: String): List<String> {
        return pattern?.regex?.split(sentence)?.filter { it.isNotEmpty() } ?: listOf(sentence)
    }

    /** Splits the sentence on digit boundaries. */
    private fun splitDigits(sentence: String): List<String> {
        return sentence.split(Regex("(?<=\\D)(?=\\d)|(?<=\\d)(?=\\D)")).filter { it.isNotEmpty() }
    }
}

/** Pattern class to hold a Regex object. */
@Serializable
data class Pattern(
    val regex: Regex
) {
    constructor(regex: String) : this(Regex(regex))
}

/** Custom BiMap implementation for bidirectional mapping. */
class BiMap<K, V> {
    private val keyToValue: MutableMap<K, V> = mutableMapOf()
    private val valueToKey: MutableMap<V, K> = mutableMapOf()

    /** Adds a key-value pair to the BiMap. */
    fun put(key: K, value: V) {
        keyToValue[key] = value
        valueToKey[value] = key
    }

    /** Retrieves the value for a key. */
    fun getValue(key: K): V? = keyToValue[key]

    /** Retrieves the key for a value. */
    fun getKey(value: V): K? = valueToKey[value]

    /** Checks if a key exists. */
    fun containsKey(key: K): Boolean = keyToValue.containsKey(key)

    /** Checks if a value exists. */
    fun containsValue(value: V): Boolean = valueToKey.containsValue(value)

    /** Returns the key-to-value map. */
    fun asMap(): Map<K, V> = keyToValue.toMap()
}

/** Custom serializer for Regex objects. */
object RegexSerializer : KSerializer<Regex> {
    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor("Regex", PrimitiveKind.STRING)

    override fun serialize(encoder: Encoder, value: Regex) {
        encoder.encodeString(value.pattern)
    }

    override fun deserialize(decoder: Decoder): Regex {
        return Regex(decoder.decodeString())
    }
}