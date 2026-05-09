import { EndOfPrompt, EndOfText, FimMiddle, FimPrefix, FimSuffix, ImEnd, ImSep, ImStart, } from '../specialTokens.js';
import { CL100K_TOKEN_SPLIT_REGEX } from './constants.js';
export function Cl100KBase(bytePairRankDecoder) {
    const specialTokenMapping = new Map([
        [EndOfText, 100_257],
        [FimPrefix, 100_258],
        [FimMiddle, 100_259],
        [FimSuffix, 100_260],
        [ImStart, 100_264],
        [ImEnd, 100_265],
        [ImSep, 100_266],
        [EndOfPrompt, 100_276],
    ]);
    return {
        tokenSplitRegex: CL100K_TOKEN_SPLIT_REGEX,
        bytePairRankDecoder,
        specialTokensEncoder: specialTokenMapping,
    };
}
//# sourceMappingURL=cl100k_base.js.map