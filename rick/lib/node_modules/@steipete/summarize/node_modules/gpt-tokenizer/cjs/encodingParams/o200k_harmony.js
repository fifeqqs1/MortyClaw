"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.O200KHarmony = O200KHarmony;
const specialTokens_js_1 = require("../specialTokens.js");
const constants_js_1 = require("./constants.js");
const RESERVED_TOKEN_RANGE_START = 200_013;
const RESERVED_TOKEN_RANGE_END = 201_088; // exclusive upper bound per tiktoken
const STATIC_SPECIAL_TOKEN_ENTRIES = [
    [specialTokens_js_1.HarmonyStartOfText, 199_998],
    [specialTokens_js_1.EndOfText, 199_999],
    ['<|reserved_200000|>', 200_000],
    ['<|reserved_200001|>', 200_001],
    [specialTokens_js_1.HarmonyReturn, 200_002],
    [specialTokens_js_1.HarmonyConstrain, 200_003],
    ['<|reserved_200004|>', 200_004],
    [specialTokens_js_1.HarmonyChannel, 200_005],
    [specialTokens_js_1.HarmonyStart, 200_006],
    [specialTokens_js_1.HarmonyEnd, 200_007],
    [specialTokens_js_1.HarmonyMessage, 200_008],
    ['<|reserved_200009|>', 200_009],
    ['<|reserved_200010|>', 200_010],
    ['<|reserved_200011|>', 200_011],
    [specialTokens_js_1.HarmonyCall, 200_012],
];
function O200KHarmony(bytePairRankDecoder) {
    const specialTokensEncoder = new Map(STATIC_SPECIAL_TOKEN_ENTRIES);
    for (let tokenId = RESERVED_TOKEN_RANGE_START; tokenId < RESERVED_TOKEN_RANGE_END; tokenId += 1) {
        specialTokensEncoder.set(`<|reserved_${tokenId}|>`, tokenId);
    }
    specialTokensEncoder.set(specialTokens_js_1.EndOfPrompt, 200_018);
    return {
        tokenSplitRegex: constants_js_1.O200K_TOKEN_SPLIT_REGEX,
        bytePairRankDecoder,
        specialTokensEncoder,
        chatFormatter: 'harmony',
    };
}
//# sourceMappingURL=o200k_harmony.js.map