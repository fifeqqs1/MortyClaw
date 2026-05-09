"use strict";
/* eslint-disable camelcase */
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.chatModelParams = exports.modelToEncodingMap = exports.encodingNames = exports.DEFAULT_ENCODING = exports.o200k_harmony = exports.o200k_base = exports.r50k_base = exports.p50k_edit = exports.p50k_base = exports.cl100k_base = void 0;
const modelsChatEnabled_gen_js_1 = require("./modelsChatEnabled.gen.js");
const encodingsMap = __importStar(require("./modelsMap.js"));
const specialTokens_js_1 = require("./specialTokens.js");
exports.cl100k_base = 'cl100k_base';
exports.p50k_base = 'p50k_base';
exports.p50k_edit = 'p50k_edit';
exports.r50k_base = 'r50k_base';
exports.o200k_base = 'o200k_base';
exports.o200k_harmony = 'o200k_harmony';
exports.DEFAULT_ENCODING = exports.o200k_base;
exports.encodingNames = [
    exports.p50k_base,
    exports.r50k_base,
    exports.p50k_edit,
    exports.cl100k_base,
    exports.o200k_base,
    exports.o200k_harmony,
];
/**
 * maps model names to encoding names
 * if a model is not listed, it uses the default encoding for new models
 * which is `o200k_base`
 */
exports.modelToEncodingMap = Object.fromEntries(Object.entries(encodingsMap).flatMap(([encodingName, models]) => models.map((modelName) => [modelName, encodingName])));
const gpt3params = {
    messageSeparator: '\n',
    roleSeparator: '\n',
};
const gpt4params = {
    messageSeparator: '',
    roleSeparator: specialTokens_js_1.ImSep,
};
exports.chatModelParams = Object.fromEntries(modelsChatEnabled_gen_js_1.chatEnabledModels.flatMap((modelName) => modelName.startsWith('gpt-3.5')
    ? [[modelName, gpt3params]]
    : [[modelName, gpt4params]]));
//# sourceMappingURL=mapping.js.map