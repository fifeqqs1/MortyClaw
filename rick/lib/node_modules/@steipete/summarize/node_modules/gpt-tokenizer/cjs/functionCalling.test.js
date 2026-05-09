"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
// eslint-disable-next-line import/no-extraneous-dependencies
const vitest_1 = require("vitest");
const functionCallingTestCases_js_1 = require("./fixtures/functionCallingTestCases.js");
const gpt_4o_js_1 = __importDefault(require("./model/gpt-4o.js"));
const { countChatCompletionTokens } = gpt_4o_js_1.default;
if (!countChatCompletionTokens) {
    throw new Error('Function calling token counting is not available for gpt-4o');
}
(0, vitest_1.describe)('countChatCompletionTokens', () => {
    (0, vitest_1.it)('matches known token counts', () => {
        functionCallingTestCases_js_1.functionCallingTestCases.forEach((testCase) => {
            const { tokens, ...request } = testCase;
            (0, vitest_1.expect)(countChatCompletionTokens(request)).toBe(tokens);
        });
    });
});
//# sourceMappingURL=functionCalling.test.js.map