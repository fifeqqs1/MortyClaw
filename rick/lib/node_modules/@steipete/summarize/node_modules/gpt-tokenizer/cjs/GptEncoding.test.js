"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
// eslint-disable-next-line import/no-extraneous-dependencies
const vitest_1 = require("vitest");
const constants_js_1 = require("./constants.js");
const GptEncoding_js_1 = require("./GptEncoding.js");
const mapping_js_1 = require("./mapping.js");
const models = __importStar(require("./models.js"));
const resolveEncoding_js_1 = require("./resolveEncoding.js");
const specialTokens_js_1 = require("./specialTokens.js");
(0, vitest_1.describe)('generated model exports', () => {
    (0, vitest_1.test)('gpt-5 re-exports the chat token counter helper', async () => {
        const mod = await Promise.resolve().then(() => __importStar(require('./model/gpt-5.js')));
        const encoding = mod.default;
        (0, vitest_1.expect)('countChatCompletionTokens' in mod).toBe(true);
        (0, vitest_1.expect)(mod.countChatCompletionTokens).toBe(encoding.countChatCompletionTokens);
    });
    (0, vitest_1.test)('gpt-3.5-turbo-0613 omits the chat token counter helper', async () => {
        const mod = await Promise.resolve().then(() => __importStar(require('./model/gpt-3.5-turbo-0613.js')));
        (0, vitest_1.expect)('countChatCompletionTokens' in mod).toBe(false);
    });
});
const sharedResults = {
    space: [220],
    tab: [197],
    'This is some text': [1_212, 318, 617, 2_420],
    indivisible: [521, 452, 12_843],
    'hello 👋 world 🌍': [31_373, 50_169, 233, 995, 12_520, 234, 235],
    decodedHelloWorldTokens: ['hello', ' ', '👋', ' world', ' ', '🌍'],
    'toString constructor hasOwnProperty valueOf': [
        1_462, 10_100, 23_772, 468, 23_858, 21_746, 1_988, 5_189,
    ],
    'hello, I am a text, and I have commas. a,b,c': [
        31_373, 11, 314, 716, 257, 2_420, 11, 290, 314, 423, 725, 292, 13, 257, 11,
        65, 11, 66,
    ],
};
const o200kBaseResults = {
    space: [220],
    tab: [197],
    'This is some text': [2_500, 382, 1_236, 2_201],
    indivisible: [521, 349, 181_386],
    'hello 👋 world 🌍': [24_912, 61_138, 233, 2_375, 130_321, 235],
    decodedHelloWorldTokens: ['hello', ' ', '👋', ' world', ' ', '🌍'],
    'toString constructor hasOwnProperty valueOf': [
        935, 916, 9_220, 853, 18_555, 3_895, 1_432, 2_566,
    ],
    'hello, I am a text, and I have commas. a,b,c': [
        24_912, 11, 357, 939, 261, 2_201, 11, 326, 357, 679, 179_663, 13, 261,
        17_568, 22_261,
    ],
};
const results = {
    o200k_base: o200kBaseResults,
    o200k_harmony: o200kBaseResults,
    cl100k_base: {
        space: [220],
        tab: [197],
        'This is some text': [2_028, 374, 1_063, 1_495],
        indivisible: [485, 344, 23_936],
        'hello 👋 world 🌍': [15_339, 62_904, 233, 1_917, 11_410, 234, 235],
        decodedHelloWorldTokens: ['hello', ' ', '👋', ' world', ' ', '🌍'],
        'toString constructor hasOwnProperty valueOf': [
            6_712, 4_797, 706, 19_964, 907, 2_173,
        ],
        'hello, I am a text, and I have commas. a,b,c': [
            15_339, 11, 358, 1_097, 264, 1_495, 11, 323, 358, 617, 77_702, 13, 264,
            8_568, 10_317,
        ],
    },
    p50k_base: sharedResults,
    p50k_edit: sharedResults,
    r50k_base: sharedResults,
};
const offsetPrompts = [
    // Basic prompt with "hello world"
    'hello world',
    // Basic prompt with special token end of text token
    `hello world${specialTokens_js_1.EndOfText} green cow`,
    // Chinese text: "我非常渴望与人工智能一起工作"
    '我非常渴望与人工智能一起工作',
    // Contains the interesting tokens b'\xe0\xae\xbf\xe0\xae' and b'\xe0\xaf\x8d\xe0\xae'
    // in which \xe0 is the start of a 3-byte UTF-8 character
    'நடிகர் சூர்யா',
    // Contains the interesting token b'\xa0\xe9\x99\xa4'
    // in which \xe9 is the start of a 3-byte UTF-8 character and \xa0 is a continuation byte
    ' Ġ除',
];
// eslint-disable-next-line @typescript-eslint/no-use-before-define
const testPlans = loadTestPlans();
vitest_1.describe.each(mapping_js_1.encodingNames)('%s', (encodingName) => {
    const encoding = GptEncoding_js_1.GptEncoding.getEncodingApi(encodingName, resolveEncoding_js_1.resolveEncoding);
    const { decode, decodeGenerator, decodeAsyncGenerator, encode, isWithinTokenLimit, countTokens, } = encoding;
    (0, vitest_1.describe)('encode and decode', () => {
        vitest_1.test.each(offsetPrompts)('offset prompt: %s', (str) => {
            (0, vitest_1.expect)(decode(encode(str, { allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS }))).toEqual(str);
        });
    });
    (0, vitest_1.describe)('basic functionality', () => {
        const result = results[encodingName];
        (0, vitest_1.test)('empty string', () => {
            const str = '';
            (0, vitest_1.expect)(encode(str)).toEqual([]);
            (0, vitest_1.expect)(decode(encode(str))).toEqual(str);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 0)).toBe(0);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 3)).toBe(0);
        });
        (0, vitest_1.test)('space', () => {
            const str = ' ';
            (0, vitest_1.expect)(encode(str)).toEqual(result.space);
            (0, vitest_1.expect)(decode(encode(str))).toEqual(str);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 3)).toBe(1);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 0)).toBe(false);
        });
        (0, vitest_1.test)('tab', () => {
            const str = '\t';
            (0, vitest_1.expect)(encode(str)).toEqual(result.tab);
            (0, vitest_1.expect)(decode(encode(str))).toEqual(str);
        });
        (0, vitest_1.test)('simple text', () => {
            const str = 'This is some text';
            (0, vitest_1.expect)(encode(str)).toEqual(result[str]);
            (0, vitest_1.expect)(decode(encode(str))).toEqual(str);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 3)).toBe(false);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 5)).toBe(result[str].length);
        });
        (0, vitest_1.test)('multi-token word', () => {
            const str = 'indivisible';
            (0, vitest_1.expect)(encode(str)).toEqual(result.indivisible);
            (0, vitest_1.expect)(decode(encode(str))).toEqual(str);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 3)).toBe(result.indivisible.length);
        });
        (0, vitest_1.test)('emojis', () => {
            const str = 'hello 👋 world 🌍';
            (0, vitest_1.expect)(encode(str)).toEqual(result[str]);
            (0, vitest_1.expect)(decode(encode(str))).toEqual(str);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 4)).toBe(false);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 400)).toBe(result[str].length);
        });
        (0, vitest_1.test)('keeps contractions intact in o200k encodings', () => {
            if (encodingName !== 'o200k_base' && encodingName !== 'o200k_harmony') {
                return;
            }
            const str = "What's";
            (0, vitest_1.expect)(encode(str)).toEqual([45_350]);
        });
        (0, vitest_1.test)('decode token-by-token via generator', () => {
            const str = 'hello 👋 world 🌍';
            const generator = decodeGenerator(result[str]);
            result.decodedHelloWorldTokens.forEach((token) => {
                (0, vitest_1.expect)(generator.next().value).toBe(token);
            });
        });
        (0, vitest_1.test)('encodes and decodes special tokens', () => {
            const str = `hello ${specialTokens_js_1.EndOfText} world`;
            const encoded = encode(str, {
                allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS,
            });
            (0, vitest_1.expect)(decode(encoded)).toEqual(str);
        });
        (0, vitest_1.test)('isWithinTokenLimit handles special tokens when allowed', () => {
            const str = `hello ${specialTokens_js_1.EndOfText} world`;
            (0, vitest_1.expect)(() => isWithinTokenLimit(str, 100)).toThrowError(/Disallowed special token found/);
            const allowedCount = isWithinTokenLimit(str, 100, {
                allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS,
            });
            (0, vitest_1.expect)(allowedCount).toBe(encode(str, { allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS }).length);
        });
        (0, vitest_1.test)('countTokens handles special tokens when allowed', () => {
            const str = `hello ${specialTokens_js_1.EndOfText} world`;
            (0, vitest_1.expect)(() => countTokens(str)).toThrowError(/Disallowed special token found/);
            (0, vitest_1.expect)(countTokens(str, { allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS })).toBe(encode(str, { allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS }).length);
        });
        async function* getHelloWorldTokensAsync() {
            const str = 'hello 👋 world 🌍';
            for (const token of result[str]) {
                // eslint-disable-next-line no-await-in-loop
                yield await Promise.resolve(token);
            }
        }
        (0, vitest_1.test)('decode token-by-token via async generator', async () => {
            const generator = decodeAsyncGenerator(getHelloWorldTokensAsync());
            const decoded = [...result.decodedHelloWorldTokens];
            for await (const value of generator) {
                (0, vitest_1.expect)(value).toEqual(decoded.shift());
            }
        });
        (0, vitest_1.test)('properties of Object', () => {
            const str = 'toString constructor hasOwnProperty valueOf';
            (0, vitest_1.expect)(encode(str)).toEqual(result[str]);
            (0, vitest_1.expect)(decode(encode(str))).toEqual(str);
        });
        (0, vitest_1.test)('text with commas', () => {
            const str = 'hello, I am a text, and I have commas. a,b,c';
            (0, vitest_1.expect)(decode(encode(str))).toEqual(str);
            (0, vitest_1.expect)(encode(str)).toStrictEqual(result[str]);
            (0, vitest_1.expect)(isWithinTokenLimit(str, result[str].length - 1)).toBe(false);
            (0, vitest_1.expect)(isWithinTokenLimit(str, 300)).toBe(result[str].length);
        });
    });
    if (testPlans[encodingName].length > 0) {
        (0, vitest_1.describe)('test plan', () => {
            testPlans[encodingName].forEach(({ sample, encoded }) => {
                (0, vitest_1.test)(`encodes ${sample}`, () => {
                    (0, vitest_1.expect)(encode(sample)).toEqual(encoded);
                });
                (0, vitest_1.test)(`decodes ${sample}`, () => {
                    (0, vitest_1.expect)(decode(encoded)).toEqual(sample);
                });
            });
        });
    }
});
const chatModelNames = Object.keys(mapping_js_1.chatModelParams);
const exampleMessages = [
    {
        role: 'system',
        content: 'You are a helpful, pattern-following assistant that translates corporate jargon into plain English.',
    },
    {
        role: 'system',
        name: 'example_user',
        content: 'New synergies will help drive top-line growth.',
    },
    {
        role: 'system',
        name: 'example_assistant',
        content: 'Things working well together will increase revenue.',
    },
    {
        role: 'system',
        name: 'example_user',
        content: "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
    },
    {
        role: 'system',
        name: 'example_assistant',
        content: "Let's talk later when we're less busy about how to do better.",
    },
    {
        role: 'user',
        content: "This late pivot means we don't have time to boil the ocean for the client deliverable.",
    },
];
vitest_1.describe.each(chatModelNames)('%s', async (modelName) => {
    const encoding = await Promise.resolve(`${`./model/${modelName}.ts`}`).then(s => __importStar(require(s))).then((mod) => mod.default);
    const encodedExampleChat = encoding.encodeChat(exampleMessages);
    const expectedEncodedLength = encodedExampleChat.length;
    (0, vitest_1.describe)('chat functionality', () => {
        (0, vitest_1.test)('encodes a chat correctly', () => {
            (0, vitest_1.expect)(encodedExampleChat).toHaveLength(expectedEncodedLength);
            (0, vitest_1.expect)(encodedExampleChat).toMatchSnapshot();
            const decoded = encoding.decode(encodedExampleChat);
            (0, vitest_1.expect)(decoded).toMatchSnapshot();
        });
        (0, vitest_1.test)('isWithinTokenLimit: false', () => {
            const isWithinTokenLimit = encoding.isWithinTokenLimit(exampleMessages, 50);
            (0, vitest_1.expect)(isWithinTokenLimit).toBe(false);
        });
        (0, vitest_1.test)('isWithinTokenLimit: true (number)', () => {
            const isWithinTokenLimit = encoding.isWithinTokenLimit(exampleMessages, expectedEncodedLength);
            (0, vitest_1.expect)(isWithinTokenLimit).toBe(expectedEncodedLength);
        });
        (0, vitest_1.test)('isWithinTokenLimit allows special tokens in chat when configured', () => {
            const chatWithSpecial = [
                { role: 'user', content: `Hello ${specialTokens_js_1.EndOfText} world` },
            ];
            (0, vitest_1.expect)(() => encoding.isWithinTokenLimit(chatWithSpecial, 100)).toThrowError(/Disallowed special token found/);
            const allowedCount = encoding.isWithinTokenLimit(chatWithSpecial, 100, {
                allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS,
            });
            const encoded = encoding.encodeChat(chatWithSpecial, undefined, {
                allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS,
            });
            (0, vitest_1.expect)(allowedCount).toBe(encoded.length);
        });
        (0, vitest_1.test)('countTokens allows special tokens in chat when configured', () => {
            const chatWithSpecial = [
                { role: 'user', content: `Hello ${specialTokens_js_1.EndOfText} world` },
            ];
            (0, vitest_1.expect)(() => encoding.countTokens(chatWithSpecial)).toThrowError(/Disallowed special token found/);
            (0, vitest_1.expect)(encoding.countTokens(chatWithSpecial, {
                allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS,
            })).toBe(encoding.encodeChat(chatWithSpecial, undefined, {
                allowedSpecial: constants_js_1.ALL_SPECIAL_TOKENS,
            }).length);
        });
    });
});
(0, vitest_1.describe)('estimateCost functionality', async () => {
    const gpt4oEncoding = await Promise.resolve().then(() => __importStar(require(`./model/gpt-4o.js`))).then((mod) => mod.default);
    const gpt35Encoding = await Promise.resolve().then(() => __importStar(require(`./model/gpt-3.5-turbo.js`))).then((mod) => mod.default);
    (0, vitest_1.test)('estimates cost correctly for gpt-4o model', () => {
        const tokenCount = 1_000;
        const cost = gpt4oEncoding.estimateCost(tokenCount);
        (0, vitest_1.expect)(cost).toMatchInlineSnapshot(`
      {
        "batch": {
          "cached_input": undefined,
          "cached_output": undefined,
          "input": 0.005,
          "output": 0.015,
        },
        "main": {
          "cached_input": undefined,
          "cached_output": undefined,
          "input": 0.01,
          "output": 0.03,
        },
      }
    `);
    });
    (0, vitest_1.test)('estimates cost correctly for gpt-3.5-turbo model', () => {
        const tokenCount = 1_000;
        const cost = gpt35Encoding.estimateCost(tokenCount);
        (0, vitest_1.expect)(cost).toMatchInlineSnapshot(`
      {
        "batch": {
          "cached_input": undefined,
          "cached_output": undefined,
          "input": 0.00025,
          "output": 0.00075,
        },
        "main": {
          "cached_input": undefined,
          "cached_output": undefined,
          "input": 0.0005,
          "output": 0.0015,
        },
      }
    `);
    });
    (0, vitest_1.test)('allows overriding model name', () => {
        const tokenCount = 1_000;
        // Use gpt-4o encoding but override with gpt-3.5-turbo model name
        const cost = gpt4oEncoding.estimateCost(tokenCount, models['gpt-3.5-turbo']);
        (0, vitest_1.expect)(cost).toMatchInlineSnapshot(`
      {
        "batch": {
          "cached_input": undefined,
          "cached_output": undefined,
          "input": 0.00025,
          "output": 0.00075,
        },
        "main": {
          "cached_input": undefined,
          "cached_output": undefined,
          "input": 0.0005,
          "output": 0.0015,
        },
      }
    `);
    });
    (0, vitest_1.test)('throws error when model name is not provided', () => {
        const encoding = GptEncoding_js_1.GptEncoding.getEncodingApi('cl100k_base', resolveEncoding_js_1.resolveEncoding);
        const tokenCount = 1_000;
        // No model name was provided during initialization or function call
        (0, vitest_1.expect)(() => encoding.estimateCost(tokenCount)).toThrow('Model spec must be provided either during initialization or passed in to the method.');
    });
    (0, vitest_1.test)('only includes properties that exist for the model', () => {
        // Find a model that only has input cost but no output cost
        const modelWithInputOnly = Object.entries(models).find(([_, model]) => 'price_data' in model &&
            model.price_data?.main?.input !== undefined &&
            (!('output' in model.price_data.main) ||
                model.price_data?.main?.output === undefined));
        if (modelWithInputOnly) {
            const [modelName] = modelWithInputOnly;
            const cost = gpt4oEncoding.estimateCost(1_000, models[modelName]);
            (0, vitest_1.expect)(cost.main?.input).toBeDefined();
            (0, vitest_1.expect)(cost.main?.output).toBeUndefined();
        }
        else {
            // Skip test if we can't find an appropriate model
            console.log('Skipping test: no model with input-only cost found');
        }
    });
});
function loadTestPlans() {
    const testPlanPath = path_1.default.join(__dirname, '../data/TestPlans.txt');
    const testPlanData = fs_1.default.readFileSync(testPlanPath, 'utf8');
    const tests = {
        cl100k_base: [],
        p50k_base: [],
        p50k_edit: [],
        r50k_base: [],
        o200k_base: [],
        o200k_harmony: [],
    };
    testPlanData.split('\n\n').forEach((testPlan) => {
        const [encodingNameLine, sampleLine, encodedLine] = testPlan.split('\n');
        if (!encodingNameLine || !sampleLine || !encodedLine)
            return;
        const encodingName = encodingNameLine.split(': ')[1];
        tests[encodingName].push({
            sample: sampleLine.split(': ').slice(1).join(': ') ?? '',
            encoded: JSON.parse(encodedLine.split(': ')[1] ?? '[]'),
        });
    });
    return tests;
}
//# sourceMappingURL=GptEncoding.test.js.map