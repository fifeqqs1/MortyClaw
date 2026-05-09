import { EndOfPrompt, EndOfText, FimMiddle, FimPrefix, FimSuffix, ImEnd, ImSep, ImStart, } from '../specialTokens.js';
import { O200K_TOKEN_SPLIT_REGEX } from './constants.js';
const O200K_BASE_SPECIAL_TOKEN_ENTRIES = [
    [EndOfText, 199_999],
    [FimPrefix, 200_000],
    [FimMiddle, 200_001],
    [FimSuffix, 200_002],
    [ImStart, 200_003],
    [ImEnd, 200_004],
    [ImSep, 200_005],
    [EndOfPrompt, 200_006],
];
export const createO200KSpecialTokenMap = () => new Map(O200K_BASE_SPECIAL_TOKEN_ENTRIES);
export function O200KBase(bytePairRankDecoder) {
    return {
        tokenSplitRegex: O200K_TOKEN_SPLIT_REGEX,
        bytePairRankDecoder,
        specialTokensEncoder: createO200KSpecialTokenMap(),
    };
}
//# sourceMappingURL=o200k_base.js.map