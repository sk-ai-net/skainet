/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package sk.ai.net.io.tokenizer

import com.github.tjake.jlama.safetensors.SafeTensorSupport
import com.github.tjake.jlama.safetensors.prompt.PromptSupport
import com.google.common.base.Preconditions
import java.io.IOException
import java.nio.file.Path
import java.util.Arrays
import java.util.Optional
import java.util.stream.Collectors
import java.util.stream.Stream
import kotlin.invoke

/**
 * WordPiece tokenizer
 * @see [...](https://github.com/google-research/bert/blob/master/tokenization.py)
 */
class WordPieceTokenizer(modelRoot: Path) : Tokenizer {
    protected val model: TokenizerModel? = null
    protected val promptSupport: PromptSupport? = null
    protected val sepToken: Long
    protected val clsToken: Long
    protected val unkToken: Long

    init {
        Preconditions.checkArgument(
            modelRoot.resolve("tokenizer.json").toFile().exists(),
            "No tokenizer.json found in " + modelRoot
        )

        try {
            this.model = SafeTensorSupport.loadTokenizer(modelRoot)
            Preconditions.checkArgument(
                model.type == null || model.type.equalsIgnoreCase("WordPiece"),
                "Invalid model type: " + model.type
            )

            this.promptSupport = PromptSupport(model)
        } catch (e: IOException) {
            throw RuntimeException(e)
        }

        this.sepToken =
            model.vocabLookup.get(sepString)
        this.clsToken =
            model.vocabLookup.get(clsString)
        this.unkToken =
            model.vocabLookup.get(unkString)
    }

    @Override
    fun getModel(): TokenizerModel {
        return model
    }

    @Override
    fun tokenize(sentence: String): List<String?> {
        var sentence = sentence
        sentence = preProcess(sentence)

        val whitespaceSplits: Array<String?>? = sentence.split("\\s+")

        val tokens: List<String?> = ArrayList()
        tokens.add(clsString)

        val stringList: List<String?>? = Arrays.stream(whitespaceSplits)
            .flatMap({ str: String -> this.splitByPunctuation(str) })
            .map({ str -> if (str.length() > 200) model.unkToken else str })
            .flatMap({ str ->
                var isBad = false
                val subTokens: List<String?> = ArrayList()

                var start = 0
                while (start < str.length()) {
                    var end: Int = str.length()
                    var curSubStr: String? = null
                    while (start < end) {
                        var substr: String? = str.substring(start, end)
                        if (start > 0) substr = "##" + substr
                        if (model.vocabLookup.containsKey(substr)) {
                            curSubStr = substr
                            break
                        }
                        end -= 1
                    }
                    if (curSubStr == null) {
                        isBad = true
                        break
                    }

                    subTokens.add(curSubStr)
                    start = end
                }

                if (isBad) subTokens.add(model.unkToken)
                subTokens.stream()
            })
            .collect(Collectors.toList())

        tokens.addAll(stringList)
        tokens.add(sepString)

        return tokens
    }

    protected fun preProcess(sentence: String): String {
        var sentence = sentence
        sentence = sentence.toLowerCase().strip()

        return cleanText(sentence)
    }

    fun cleanText(sentence: String): String {
        return sentence.codePoints().map({ c ->
            if (c === 0 || c === 0xfffd || isControl(
                    c
                )
            ) return@map -1
            if (Character.isWhitespace(c)) return@map ' '
            c
        }).filter({ c -> c !== -1 }).mapToObj(Character::toString).collect(Collectors.joining())
    }

    fun splitByPunctuation(str: String): Stream<String?> {
        val result: List<String?> = ArrayList()

        var start = 0

        var offset = 0
        while (offset < str.length()) {
            val codepoint: Int = str.codePointAt(offset)

            if (isPunctuation(codepoint)) {
                if (offset != start) {
                    result.add(str.substring(start, offset))
                }
                result.add(str.substring(offset, offset + Character.charCount(codepoint)))
                start = offset + Character.charCount(codepoint)
            }

            offset += Character.charCount(codepoint)
        }

        // Add the remaining part if there's any
        if (start != str.length()) {
            result.add(str.substring(start))
        }

        return result.stream()
    }

    @Override
    fun encode(sentence: String): LongArray {
        return tokenize(sentence).stream().mapToLong({ s -> model.vocabLookup.get(s) }).toArray()
    }

    protected fun postProcessToken(decoded: String): String? {
        if (decoded.startsWith("##")) return decoded.substring(2)

        return " " + decoded
    }

    @Override
    fun decode(id: Long): String? {
        return postProcessToken(model.vocabLookup.inverse().get(id))
    }

    protected fun postProcess(sentence: String): String {
        return sentence.strip()
    }

    @Override
    fun decode(ids: LongArray?): String {
        return postProcess(Arrays.stream(ids).mapToObj(this::decode).collect(Collectors.joining()))
    }

    @Override
    fun promptSupport(): Optional<PromptSupport?> {
        return if (model.promptTemplates().isPresent()) Optional.of(promptSupport) else Optional.empty()
    }

    companion object {
        protected val sepString: String = "[SEP]"
        protected val clsString: String = "[CLS]"
        protected val unkString: String = "[UNK]"

        fun isControl(c: Integer?): Boolean {
            // These are technically control characters but we count them as whitespace characters.
            if (c == '\t' || c == '\n' || c == '\r') return false

            return Character.isISOControl(c)
        }

        fun isPunctuation(cp: Integer?): Boolean {
            if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
                return true
            }

            val t: Int = Character.getType(cp)
            if (t >= 20 && t <= 24) return true

            return false
        }
    }
}
