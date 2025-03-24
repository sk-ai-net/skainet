package sk.ai.net.io.tokenizer

/**
 * Byte Pair Encoding tokenizer
 */
class BPETokenizer protected constructor(modelRoot: TokenizerModel) : Tokenizer {
    override val model: TokenizerModel = modelRoot
    //protected val decodeBuffer: ByteBuffer = ByteBuffer.allocate(4)

    fun getModel(): TokenizerModel {
        return model
    }

    fun tokenize(sentence: String): List<String?> {
        if (sentence.isEmpty()) return emptyList()




        if (model.preTokenizer() == null && model.addedTokenPattern() == null) Collections.singletonList(sentence)

        val sentencePieces: List<String> = ArrayList()
        if (model.addedTokenPattern() != null) {
            // Split the sentence into pieces using the added token pattern
            // Any non-added token is split into pieces using the pre-tokenizer
            val pieces: Array<String> = TokenizerModel.Companion.split(model.addedTokenPattern(), sentence, 0, true)
            for (piece in pieces) {
                if (!piece.isEmpty()) {
                    if (model.addedTokens().containsKey(piece)) sentencePieces.add(piece)
                    else if (model.preTokenizer() != null) sentencePieces.addAll(
                        model.preTokenizer().pretokenize(piece)
                    )
                    else sentencePieces.add(piece)
                }
            }
        } else if (model.preTokenizer() != null) {
            sentencePieces.addAll(model.preTokenizer().pretokenize(sentence))
        } else {
            sentencePieces.add(sentence)
        }

        return sentencePieces


        return emptyList()
    }

    protected fun preProcess(sentence: String?): String? {
        return sentence
    }

    fun encode(rawSentence: String): LongArray {
        val sentencePieces = tokenize(rawSentence)
        val allTokens: List<Long?> = ArrayList()

        for (sentence in sentencePieces) {
            var sentence: String = sentence!!
            if (model.addedTokens() != null && model.addedTokens().containsKey(sentence)) {
                allTokens.add(model.addedTokens().get(sentence))
                continue
            }
            val tokens: List<Long?> = ArrayList()
            sentence = preProcess(sentence)!!
            val codes: IntArray = sentence.codePoints().toArray()
            for (i in codes.indices) {
                val c: String? = Character.toString(codes[i])
                val id: Long? = model.vocabLookup.get(c)
                if (id != null) {
                    // we found this codepoint in vocab, add it as a token
                    // logger.debug("{} -> {}", c, id);
                    tokens.add(id)
                } else {
                    if (model.byteFallback) {
                        // byte_fallback encoding: just encode each byte as a token
                        val code = Character.toString(codes[i])
                        val chars: ByteArray = code.getBytes(StandardCharsets.UTF_8)
                        for (k in chars.indices) {
                            val token = encodeCharacterAsToken(chars[k])
                            // logger.debug("byte {} -> {}", Byte.toUnsignedInt(chars[k]), token);
                            tokens.add(token)
                        }
                    } else {
                        if (model.unkToken != null) {
                            tokens.add(model.vocabLookup.get(model.unkToken))
                        }
                    }
                }
            }

            // merge the best consecutive tuple each iteration,
            // until we can't find any more pairs to merge
            while (true) {
                var bestId: Long = -1
                var bestIdx: Long = -1
                var bestRank = Long.MAX_VALUE

                for (i in 0..<tokens.size() - 1) {
                    // check if we can merge the pair (tokens[i], tokens[i+1])
                    val token1 = decodeInternal(tokens.get(i)!!)
                    val token2 = decodeInternal(tokens.get(i + 1)!!)

                    val merge2: String? = String.format("%s %s", token1, token2)
                    val merge3: String? = String.format("%s%s", token1, token2)

                    if (model.merges.containsKey(merge2)) {
                        val id: Long? = model.vocabLookup.get(merge3)
                        if (id != null) {
                            // Check if this merge has a better rank (i.e., lower rank number)
                            val rank: Long = model.merges.get(merge2)!!
                            if (rank < bestRank) {
                                // this merge pair exists in vocab! record its position
                                bestId = id
                                bestIdx = i.toLong()
                                bestRank = rank
                            }
                        }
                    }
                }

                if (bestIdx == -1L) {
                    break // we couldn't find any more pairs to merge, so we're done
                }

                // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
                tokens.set(bestIdx.toInt(), bestId)
                // delete token at position best_idx+1, shift the entire sequence back 1
                tokens.remove(bestIdx.toInt() + 1)
            }

            allTokens.addAll(tokens)
        }

        return allTokens.stream().mapToLong({ s -> s }).toArray()
    }

    protected fun postProcessToken(decoded: String?): String? {
        var decoded = decoded
        if (decoded == null) decoded = model.unkToken

        return decoded
    }

    override fun tokenize(sentence: String?): List<String?>? {
        return if (sentence == null) null else tokenize(sentence)
    }

    override fun encode(sentence: String?): LongArray? {
        TODO("Not yet implemented")
    }

    override fun decode(id: Long): String {
        return maybeDecodeTokenAsCharacter(id).map({ c ->
            // We have a continuation byte or are buffering them
            if (Character.isUnicodeIdentifierPart(c) || decodeBuffer.remaining() < 4) {
                decodeBuffer.put(c.charValue() as Byte)

                // Unicode symbol is ready
                if (decodeBuffer.remaining() === 0) {
                    val s = String(decodeBuffer.array())
                    decodeBuffer.rewind()
                    return@map s
                }

                return@map ""
            }
            Character.toString(c)
        }).orElseGet({ postProcessToken(model.vocabLookup.inverse().get(id)) })
    }

    protected abstract fun encodeCharacterAsToken(c: Byte): Long

    protected abstract fun maybeDecodeTokenAsCharacter(id: Long): Optional<Character?>?

    // Only used for merging
    protected fun decodeInternal(id: Long): String {
        return maybeDecodeTokenAsCharacter(id).map(Object::toString).orElseGet({
            var s: String? = model.vocabLookup.inverse().get(id)
            if (s == null) s = model.unkToken
            s
        })
    }

    protected fun postProcess(sentence: String?): String? {
        return sentence
    }

    override fun decode(ids: LongArray?): String? {
        return ""
        // return postProcess(Arrays.stream(ids).mapToObj(this::decode).collect(Collectors.joining()))
    }

    companion object {
        var alteredBytes: BiMap<Int?, Int?>? // Codepoint and Token mapping needed for legacy mode

        init {
            // https://github.com/openai/gpt-2/blob/master/src/encoder.py#L19
            val tmpAlteredBytes: BiMap<Integer?, Integer?> = HashBiMap.create()
            var i = 0
            for (c in 0..255) {
                if ((c < '!'.code || c > '~'.code) && (c < '¡'.code || c > '¬'.code) && (c < '®'.code || c > 'ÿ'.code)) {
                    val codepoint = (i++ + 256)
                    tmpAlteredBytes.put(c, codepoint)
                }
            }

            alteredBytes =
                ImmutableBiMap.copyOf(tmpAlteredBytes)
        }
    }
}
