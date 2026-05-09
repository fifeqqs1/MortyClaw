"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createO200KSpecialTokenMap = void 0;
exports.O200KBase = O200KBase;
const specialTokens_js_1 = require("../specialTokens.js");
const constants_js_1 = require("./constants.js");
const O200K_BASE_SPECIAL_TOKEN_ENTRIES = [
    [specialTokens_js_1.EndOfText, 199_999],
    [specialTokens_js_1.FimPrefix, 200_000],
    [specialTokens_js_1.FimMiddle, 200_001],
    [specialTokens_js_1.FimSuffix, 200_002],
    [specialTokens_js_1.ImStart, 200_003],
    [specialTokens_js_1.ImEnd, 200_004],
    [specialTokens_js_1.ImSep, 200_005],
    [specialTokens_js_1.EndOfPrompt, 200_006],
];
const createO200KSpecialTokenMap = () => new Map(O200K_BASE_SPECIAL_TOKEN_ENTRIES);
exports.createO200KSpecialTokenMap = createO200KSpecialTokenMap;
function O200KBase(bytePairRankDecoder) {
    return {
        tokenSplitRegex: constants_js_1.O200K_TOKEN_SPLIT_REGEX,
        bytePairRankDecoder,
        specialTokensEncoder: (0, exports.createO200KSpecialTokenMap)(),
    };
}
//# sourceMappingURL=o200k_base.js.map