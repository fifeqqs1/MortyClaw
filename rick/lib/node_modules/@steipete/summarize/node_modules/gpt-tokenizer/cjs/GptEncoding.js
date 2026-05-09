"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GptEncoding = void 0;
/* eslint-disable @typescript-eslint/member-ordering */
/* eslint-disable no-param-reassign */
const BytePairEncodingCore_js_1 = require("./BytePairEncodingCore.js");
const constants_js_1 = require("./constants.js");
const functionCalling_js_1 = require("./functionCalling.js");
const mapping_js_1 = require("./mapping.js");
const modelParams_js_1 = require("./modelParams.js");
const specialTokens_js_1 = require("./specialTokens.js");
const utfUtil_js_1 = require("./utfUtil.js");
const util_js_1 = require("./util.js");
class GptEncoding {
    static EndOfPrompt = specialTokens_js_1.EndOfPrompt;
    static EndOfText = specialTokens_js_1.EndOfText;
    static FimMiddle = specialTokens_js_1.FimMiddle;
    static FimPrefix = specialTokens_js_1.FimPrefix;
    static FimSuffix = specialTokens_js_1.FimSuffix;
    modelName;
    modelSpec;
    bytePairEncodingCoreProcessor;
    specialTokensEncoder;
    specialTokensSet;
    allSpecialTokenRegex;
    defaultSpecialTokenConfig;
    chatFormatter;
    countChatCompletionTokens;
    vocabularySize;
    constructor({ bytePairRankDecoder: mergeableBytePairRanks, specialTokensEncoder, expectedVocabularySize, modelName, modelSpec, chatFormatter, ...rest }) {
        this.specialTokensEncoder = specialTokensEncoder;
        this.specialTokensSet = new Set(this.specialTokensEncoder.keys());
        this.allSpecialTokenRegex = (0, util_js_1.getSpecialTokenRegex)(this.specialTokensSet);
        this.bytePairEncodingCoreProcessor = new BytePairEncodingCore_js_1.BytePairEncodingCore({
            bytePairRankDecoder: mergeableBytePairRanks,
            specialTokensEncoder,
            ...rest,
        });
        this.defaultSpecialTokenConfig = this.processSpecialTokens();
        const maxTokenValue = Math.max(mergeableBytePairRanks.length - 1, (0, util_js_1.getMaxValueFromMap)(specialTokensEncoder));
        this.vocabularySize =
            this.bytePairEncodingCoreProcessor.mergeableBytePairRankCount +
                specialTokensEncoder.size;
        if (expectedVocabularySize !== undefined) {
            if (this.vocabularySize !== expectedVocabularySize) {
                throw new Error('The number of mergeable tokens and special tokens must be equal to expectedVocabularySize.');
            }
            if (maxTokenValue !== expectedVocabularySize - 1) {
                throw new Error(`The model encodings are invalid. The maximum token value must be equal to expectedVocabularySize - 1. Currently ${maxTokenValue}, expected ${expectedVocabularySize - 1}`);
            }
        }
        this.encode = this.encode.bind(this);
        this.decode = this.decode.bind(this);
        this.encodeGenerator = this.encodeGenerator.bind(this);
        this.decodeGenerator = this.decodeGenerator.bind(this);
        this.decodeAsyncGenerator = this.decodeAsyncGenerator.bind(this);
        this.decodeAsync = this.decodeAsync.bind(this);
        this.isWithinTokenLimit = this.isWithinTokenLimit.bind(this);
        this.encodeChat = this.encodeChat.bind(this);
        this.encodeChatGenerator = this.encodeChatGenerator.bind(this);
        this.countTokens = this.countTokens.bind(this);
        this.setMergeCacheSize = this.setMergeCacheSize.bind(this);
        this.clearMergeCache = this.clearMergeCache.bind(this);
        this.estimateCost = this.estimateCost.bind(this);
        if (modelSpec?.supported_features?.includes('function_calling')) {
            this.countChatCompletionTokens =
                this.countChatCompletionTokensInternal.bind(this);
        }
        this.modelName = modelName;
        this.modelSpec = modelSpec;
        this.chatFormatter = chatFormatter ?? 'chatml';
    }
    *encodeHarmonyChatGenerator(chat, encodeOptions) {
        const harmonyStart = this.specialTokensEncoder.get(specialTokens_js_1.HarmonyStart);
        const harmonyMessage = this.specialTokensEncoder.get(specialTokens_js_1.HarmonyMessage);
        const harmonyEnd = this.specialTokensEncoder.get(specialTokens_js_1.HarmonyEnd);
        const harmonyReturn = this.specialTokensEncoder.get(specialTokens_js_1.HarmonyReturn);
        const harmonyCall = this.specialTokensEncoder.get(specialTokens_js_1.HarmonyCall);
        const harmonyChannel = this.specialTokensEncoder.get(specialTokens_js_1.HarmonyChannel);
        const harmonyConstrain = this.specialTokensEncoder.get(specialTokens_js_1.HarmonyConstrain);
        if (harmonyStart === undefined ||
            harmonyMessage === undefined ||
            harmonyEnd === undefined ||
            harmonyReturn === undefined ||
            harmonyCall === undefined ||
            harmonyChannel === undefined ||
            harmonyConstrain === undefined) {
            throw new Error('Harmony chat format requires dedicated special tokens.');
        }
        const encodeHeaderText = (text) => text.length > 0 ? this.encode(text) : [];
        const resolveTerminatorToken = (terminator) => {
            switch (terminator) {
                case '<|return|>':
                    return harmonyReturn;
                case '<|call|>':
                    return harmonyCall;
                // eslint-disable-next-line unicorn/no-useless-switch-case
                case '<|end|>':
                default:
                    return harmonyEnd;
            }
        };
        for (const message of chat) {
            if (message.content === undefined) {
                throw new Error('Content must be defined for all messages.');
            }
            const roleOrName = message.name ?? message.role ?? 'user';
            yield [harmonyStart];
            yield encodeHeaderText(roleOrName);
            const recipientInRole = message.recipient &&
                (message.recipientPlacement === 'role' || !message.channel);
            const recipientInChannel = message.recipient && !recipientInRole;
            if (recipientInRole) {
                yield encodeHeaderText(` to=${message.recipient}`);
            }
            if (message.channel) {
                yield [harmonyChannel];
                yield encodeHeaderText(message.channel);
                if (recipientInChannel) {
                    yield encodeHeaderText(` to=${message.recipient}`);
                }
            }
            if (message.constraint) {
                yield [harmonyConstrain];
                yield encodeHeaderText(message.constraint);
            }
            yield [harmonyMessage];
            yield* this.encodeGenerator(message.content, encodeOptions);
            yield [resolveTerminatorToken(message.terminator)];
        }
        const assistantPrime = encodeOptions?.primeWithAssistantResponse ?? 'assistant';
        if (assistantPrime.length > 0) {
            yield [harmonyStart];
            yield encodeHeaderText(assistantPrime);
        }
    }
    static getEncodingApi(encodingName, getMergeableRanks) {
        const modelParams = (0, modelParams_js_1.getEncodingParams)(encodingName, getMergeableRanks);
        return new GptEncoding(modelParams);
    }
    static getEncodingApiForModel(modelName, getMergeableRanks, modelSpec) {
        const encodingName = mapping_js_1.modelToEncodingMap[modelName] ?? mapping_js_1.DEFAULT_ENCODING;
        const modelParams = (0, modelParams_js_1.getEncodingParams)(encodingName, getMergeableRanks);
        return new GptEncoding({ ...modelParams, modelName, modelSpec });
    }
    processSpecialTokens({ allowedSpecial, disallowedSpecial, } = {}) {
        let regexPattern;
        if (allowedSpecial === constants_js_1.ALL_SPECIAL_TOKENS ||
            allowedSpecial?.has(constants_js_1.ALL_SPECIAL_TOKENS)) {
            allowedSpecial = new Set(this.specialTokensSet);
            const allowedSpecialSet = allowedSpecial;
            if (disallowedSpecial === constants_js_1.ALL_SPECIAL_TOKENS) {
                throw new Error('allowedSpecial and disallowedSpecial cannot both be set to "all".');
            }
            if (typeof disallowedSpecial === 'object') {
                // remove any special tokens that are disallowed
                disallowedSpecial.forEach((val) => allowedSpecialSet.delete(val));
            }
            else {
                // all special tokens are allowed, and no 'disallowedSpecial' is provided
                disallowedSpecial = new Set();
            }
        }
        if (!disallowedSpecial ||
            disallowedSpecial === constants_js_1.ALL_SPECIAL_TOKENS ||
            disallowedSpecial.has(constants_js_1.ALL_SPECIAL_TOKENS)) {
            // by default, all special tokens are disallowed
            disallowedSpecial = new Set(this.specialTokensSet);
            const disallowedSpecialSet = disallowedSpecial;
            if (allowedSpecial?.size) {
                allowedSpecial.forEach((val) => disallowedSpecialSet.delete(val));
                // disallowed takes precedence over allowed
                disallowedSpecial.forEach((val) => allowedSpecial.delete(val));
                if (disallowedSpecial.size > 0) {
                    regexPattern = (0, util_js_1.getSpecialTokenRegex)(disallowedSpecial);
                }
            }
            else {
                regexPattern = this.allSpecialTokenRegex;
            }
        }
        return { allowedSpecial, regexPattern };
    }
    encodeGenerator(lineToEncode, encodeOptions) {
        const specialTokenConfig = encodeOptions
            ? this.processSpecialTokens(encodeOptions)
            : this.defaultSpecialTokenConfig;
        if (specialTokenConfig.regexPattern) {
            const match = lineToEncode.match(specialTokenConfig.regexPattern);
            if (match !== null) {
                throw new Error(`Disallowed special token found: ${match[0]}`);
            }
        }
        return this.bytePairEncodingCoreProcessor.encodeNativeGenerator(lineToEncode, specialTokenConfig.allowedSpecial);
    }
    encode(lineToEncode, encodeOptions) {
        const specialTokenConfig = encodeOptions
            ? this.processSpecialTokens(encodeOptions)
            : this.defaultSpecialTokenConfig;
        if (specialTokenConfig.regexPattern) {
            const match = lineToEncode.match(specialTokenConfig.regexPattern);
            if (match !== null) {
                throw new Error(`Disallowed special token found: ${match[0]}`);
            }
        }
        return this.bytePairEncodingCoreProcessor.encodeNative(lineToEncode, specialTokenConfig.allowedSpecial);
    }
    /**
     * Progressively tokenizes an OpenAI chat.
     * Warning: gpt-3.5-turbo and gpt-4 chat format may change over time.
     * Returns tokens assuming the 'gpt-3.5-turbo-0301' / 'gpt-4-0314' format.
     * Based on OpenAI's guidelines: https://github.com/openai/openai-python/blob/main/chatml.md
     * Also mentioned in section 6 of this document: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
     * @param encodeOptions Options controlling how special tokens are handled.
     */
    *encodeChatGenerator(chat, model = this.modelName, encodeOptions) {
        if (!model) {
            throw new Error('Model name must be provided either during initialization or passed in to the method.');
        }
        const params = mapping_js_1.chatModelParams[model];
        if (!params) {
            throw new Error(`Model '${model}' does not support chat.`);
        }
        if (this.chatFormatter === 'harmony') {
            yield* this.encodeHarmonyChatGenerator(chat, encodeOptions);
            return;
        }
        const chatStartToken = this.specialTokensEncoder.get(specialTokens_js_1.ImStart);
        const chatEndToken = this.specialTokensEncoder.get(specialTokens_js_1.ImEnd);
        if (chatStartToken === undefined || chatEndToken === undefined) {
            throw new Error(`Model '${model}' does not support chat.`);
        }
        const allowedSpecial = new Set([specialTokens_js_1.ImSep]);
        const { messageSeparator, roleSeparator } = params;
        const encodedMessageSeparator = messageSeparator.length > 0 ? this.encode(messageSeparator) : [];
        const encodedRoleSeparator = roleSeparator.length > 0
            ? this.encode(roleSeparator, { allowedSpecial })
            : [];
        const nameCache = new Map();
        for (const { role = 'system', name = role, content } of chat) {
            if (content === undefined) {
                throw new Error('Content must be defined for all messages.');
            }
            yield [chatStartToken];
            const encodedName = nameCache.get(name) ?? this.encode(name);
            nameCache.set(name, encodedName);
            yield encodedName;
            if (encodedRoleSeparator.length > 0) {
                yield encodedRoleSeparator;
            }
            yield* this.encodeGenerator(content, encodeOptions);
            yield [chatEndToken];
            yield encodedMessageSeparator;
        }
        // every reply is primed with <|start|>assistant<|message|>
        const assistantPrime = encodeOptions?.primeWithAssistantResponse ?? 'assistant';
        if (assistantPrime.length > 0) {
            yield [chatStartToken];
            yield* this.encodeGenerator(assistantPrime, encodeOptions);
        }
        if (encodedRoleSeparator.length > 0) {
            yield encodedRoleSeparator;
        }
    }
    /**
     * Encodes a chat into a single array of tokens.
     * Warning: gpt-3.5-turbo and gpt-4 chat format may change over time.
     * Returns tokens assuming the 'gpt-3.5-turbo-0301' / 'gpt-4-0314' format.
     * Based on OpenAI's guidelines: https://github.com/openai/openai-python/blob/main/chatml.md
     * Also mentioned in section 6 of this document: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
     * @param encodeOptions Options controlling how special tokens are handled.
     */
    encodeChat(chat, model = this.modelName, encodeOptions) {
        return [...this.encodeChatGenerator(chat, model, encodeOptions)].flat();
    }
    /**
     * Checks whether the provided input stays within the provided token limit.
     * @param input The string or chat messages to evaluate.
     * @param tokenLimit The maximum allowed number of tokens.
     * @param encodeOptions Options controlling how special tokens are handled.
     * @returns {false | number} false if token limit is exceeded, otherwise the number of tokens
     */
    isWithinTokenLimit(input, tokenLimit, encodeOptions) {
        const tokenGenerator = typeof input === 'string'
            ? this.encodeGenerator(input, encodeOptions)
            : this.encodeChatGenerator(input, undefined, encodeOptions);
        let count = 0;
        for (const tokens of tokenGenerator) {
            count += tokens.length;
            if (count > tokenLimit) {
                return false;
            }
        }
        return count;
    }
    /**
     * Counts the number of tokens in the input.
     * @param input The string or chat messages to evaluate.
     * @param encodeOptions Options controlling how special tokens are handled.
     * @returns {number} The number of tokens.
     */
    countTokens(input, encodeOptions) {
        if (typeof input === 'string') {
            const specialTokenConfig = encodeOptions
                ? this.processSpecialTokens(encodeOptions)
                : this.defaultSpecialTokenConfig;
            if (specialTokenConfig.regexPattern) {
                const match = input.match(specialTokenConfig.regexPattern);
                if (match !== null) {
                    throw new Error(`Disallowed special token found: ${match[0]}`);
                }
            }
            return this.bytePairEncodingCoreProcessor.countNative(input, specialTokenConfig.allowedSpecial);
        }
        const tokenGenerator = this.encodeChatGenerator(input, undefined, encodeOptions);
        let count = 0;
        for (const tokens of tokenGenerator) {
            count += tokens.length;
        }
        return count;
    }
    countStringTokens(text) {
        if (!text) {
            return 0;
        }
        return this.bytePairEncodingCoreProcessor.countNative(text);
    }
    countChatCompletionTokensInternal(request) {
        return (0, functionCalling_js_1.computeChatCompletionTokenCount)(request, (text) => this.countStringTokens(text));
    }
    setMergeCacheSize(size) {
        this.bytePairEncodingCoreProcessor.setMergeCacheSize(size);
    }
    clearMergeCache() {
        this.bytePairEncodingCoreProcessor.clearMergeCache();
    }
    decode(inputTokensToDecode) {
        return this.bytePairEncodingCoreProcessor.decodeNative(inputTokensToDecode);
    }
    *decodeGenerator(inputTokensToDecode) {
        const decodedByteGenerator = this.bytePairEncodingCoreProcessor.decodeNativeGenerator(inputTokensToDecode);
        let buffer = '';
        for (const decodedPart of decodedByteGenerator) {
            buffer +=
                typeof decodedPart === 'string'
                    ? decodedPart
                    : BytePairEncodingCore_js_1.decoder.decode(decodedPart, { stream: true });
            if (buffer.length === 0 || (0, utfUtil_js_1.endsWithIncompleteUtfPairSurrogate)(buffer)) {
                // Keep the high surrogate in the buffer and continue with the next token
                // eslint-disable-next-line no-continue
                continue;
            }
            else {
                yield buffer;
                // reset buffer
                buffer = '';
            }
        }
        // Yield any remaining characters in the buffer
        if (buffer.length > 0) {
            yield buffer;
        }
    }
    async *decodeAsyncGenerator(inputTokensToDecode) {
        const decodedByteGenerator = this.bytePairEncodingCoreProcessor.decodeNativeAsyncIterable(inputTokensToDecode);
        let buffer = '';
        for await (const decodedPart of decodedByteGenerator) {
            buffer +=
                typeof decodedPart === 'string'
                    ? decodedPart
                    : BytePairEncodingCore_js_1.decoder.decode(decodedPart, { stream: true });
            if (buffer.length === 0 || (0, utfUtil_js_1.endsWithIncompleteUtfPairSurrogate)(buffer)) {
                // Keep the high surrogate in the buffer and continue with the next token
                // eslint-disable-next-line no-continue
                continue;
            }
            else {
                yield buffer;
                // reset buffer
                buffer = '';
            }
        }
        // Yield any remaining characters in the buffer
        if (buffer.length > 0) {
            yield buffer;
        }
    }
    async decodeAsync(inputTokensToDecode) {
        const decodedByteGenerator = this.bytePairEncodingCoreProcessor.decodeNativeAsyncIterable(inputTokensToDecode);
        let buffer = '';
        for await (const decodedPart of decodedByteGenerator) {
            buffer +=
                typeof decodedPart === 'string'
                    ? decodedPart
                    : BytePairEncodingCore_js_1.decoder.decode(decodedPart, { stream: true });
        }
        return buffer;
    }
    /**
     * Estimates the cost of processing a given token count using the model's pricing.
     *
     * @param tokenCount - The number of tokens to estimate cost for
     * @returns Cost estimate object with applicable price components (input, output, batchInput, batchOutput)
     */
    estimateCost(tokenCount, modelSpec = this.modelSpec) {
        if (!modelSpec) {
            throw new Error('Model spec must be provided either during initialization or passed in to the method.');
        }
        if (!modelSpec.price_data) {
            throw new Error(`No cost information available for model: ${modelSpec.name}`);
        }
        const priceDataPerMillion = modelSpec.price_data;
        const result = {};
        // Calculate cost per token and multiply by token count
        // eslint-disable-next-line no-magic-numbers
        const millionTokens = tokenCount / 1_000_000;
        if (priceDataPerMillion.main) {
            result.main = {
                input: priceDataPerMillion.main.input &&
                    priceDataPerMillion.main.input * millionTokens,
                output: priceDataPerMillion.main.output &&
                    priceDataPerMillion.main.output * millionTokens,
                cached_input: priceDataPerMillion.main.cached_input &&
                    priceDataPerMillion.main.cached_input * millionTokens,
                cached_output: priceDataPerMillion.main.cached_output &&
                    priceDataPerMillion.main.cached_output * millionTokens,
            };
        }
        if (priceDataPerMillion.batch) {
            result.batch = {
                input: priceDataPerMillion.batch.input &&
                    priceDataPerMillion.batch.input * millionTokens,
                output: priceDataPerMillion.batch.output &&
                    priceDataPerMillion.batch.output * millionTokens,
                cached_input: priceDataPerMillion.batch.cached_input &&
                    priceDataPerMillion.batch.cached_input * millionTokens,
                cached_output: priceDataPerMillion.batch.cached_output &&
                    priceDataPerMillion.batch.cached_output * millionTokens,
            };
        }
        return result;
    }
}
exports.GptEncoding = GptEncoding;
//# sourceMappingURL=GptEncoding.js.map