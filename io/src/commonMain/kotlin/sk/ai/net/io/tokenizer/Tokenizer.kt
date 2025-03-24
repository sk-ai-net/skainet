package sk.ai.net.io.tokenizer

/**
 * Tokenizer interface
 */
interface Tokenizer {
    /**
     * Tokenize a sentence
     * @param sentence
     * @return list of token strings
     */
    fun tokenize(sentence: String?): List<String?>?

    /**
     * Encode a sentence into a list of token ids
     * @param sentence
     * @return list of token ids
     */
    fun encode(sentence: String?): LongArray?

    /**
     * Decode a token id into its string representation
     * @param id
     * @return token string
     */
    fun decode(id: Long): String?

    /**
     * Decode a list of token ids into their string representation
     * @param ids list of token ids
     * @return list of token strings
     */
    fun decode(ids: LongArray?): String?

    /**
     * Get the model for this tokenizer (expert mode)
     * @return tokenizer model
     */
    val model: TokenizerModel?
}
