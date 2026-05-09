"use strict";
/**
 * @vitest-environment edge-runtime
 */
Object.defineProperty(exports, "__esModule", { value: true });
// eslint-disable-next-line import/no-extraneous-dependencies
const vitest_1 = require("vitest");
const GptEncoding_js_1 = require("./GptEncoding.js");
const resolveEncoding_js_1 = require("./resolveEncoding.js");
(0, vitest_1.describe)('edge-runtime', () => {
    const encoding = GptEncoding_js_1.GptEncoding.getEncodingApi('o200k_base', resolveEncoding_js_1.resolveEncoding);
    const { decode, encode, isWithinTokenLimit } = encoding;
    (0, vitest_1.test)('simple text', () => {
        const str = 'This is some text';
        const encoded = [2_500, 382, 1_236, 2_201];
        (0, vitest_1.expect)(encode(str)).toEqual(encoded);
        (0, vitest_1.expect)(decode(encode(str))).toEqual(str);
        (0, vitest_1.expect)(isWithinTokenLimit(str, 3)).toBe(false);
        (0, vitest_1.expect)(isWithinTokenLimit(str, 5)).toBe(encoded.length);
    });
});
//# sourceMappingURL=otherRuntimes.test.js.map