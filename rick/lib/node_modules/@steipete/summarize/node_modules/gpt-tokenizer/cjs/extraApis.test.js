"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
// eslint-disable-next-line import/no-extraneous-dependencies
const vitest_1 = require("vitest");
const gpt_3_5_turbo_js_1 = __importDefault(require("./model/gpt-3.5-turbo.js"));
const sampleText = 'This is a test message.';
const sampleChat = [
    {
        role: 'system',
        content: 'You are a helpful assistant.',
    },
    {
        role: 'user',
        content: 'Hello, how are you?',
    },
    {
        role: 'assistant',
        content: 'I am doing well, thank you for asking.',
    },
];
(0, vitest_1.describe)('countTokens', () => {
    (0, vitest_1.describe)('text input', () => {
        (0, vitest_1.it)('counts tokens in empty string', () => {
            (0, vitest_1.expect)(gpt_3_5_turbo_js_1.default.countTokens('')).toBe(0);
        });
        (0, vitest_1.it)('counts tokens in simple text', () => {
            (0, vitest_1.expect)(gpt_3_5_turbo_js_1.default.countTokens(sampleText)).toBe(gpt_3_5_turbo_js_1.default.encode(sampleText).length);
        });
        (0, vitest_1.it)('counts tokens in text with special characters', () => {
            const textWithSpecial = 'Hello 👋 world! 🌍';
            (0, vitest_1.expect)(gpt_3_5_turbo_js_1.default.countTokens(textWithSpecial)).toBe(gpt_3_5_turbo_js_1.default.encode(textWithSpecial).length);
        });
    });
    (0, vitest_1.describe)('chat input', () => {
        (0, vitest_1.it)('counts tokens in empty chat', () => {
            (0, vitest_1.expect)(gpt_3_5_turbo_js_1.default.countTokens([])).toBe(3); // Due to assistant prompt tokens
        });
        (0, vitest_1.it)('counts tokens in sample chat', () => {
            (0, vitest_1.expect)(gpt_3_5_turbo_js_1.default.countTokens(sampleChat)).toBe(gpt_3_5_turbo_js_1.default.encodeChat(sampleChat).length);
        });
        (0, vitest_1.it)('matches token counts from encode methods', () => {
            const tokens = gpt_3_5_turbo_js_1.default.encodeChat(sampleChat);
            const count = gpt_3_5_turbo_js_1.default.countTokens(sampleChat);
            (0, vitest_1.expect)(count).toBe(tokens.length);
        });
        (0, vitest_1.it)('counts tokens in single message chat', () => {
            const singleMessage = [
                {
                    role: 'user',
                    content: 'Hello world',
                },
            ];
            (0, vitest_1.expect)(gpt_3_5_turbo_js_1.default.countTokens(singleMessage)).toBe(gpt_3_5_turbo_js_1.default.encodeChat(singleMessage).length);
        });
    });
});
//# sourceMappingURL=extraApis.test.js.map