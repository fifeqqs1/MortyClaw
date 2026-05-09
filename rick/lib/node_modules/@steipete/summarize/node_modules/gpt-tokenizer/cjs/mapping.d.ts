import { chatEnabledModels } from './modelsChatEnabled.gen.js';
import * as encodingsMap from './modelsMap.js';
export declare const cl100k_base = "cl100k_base";
export declare const p50k_base = "p50k_base";
export declare const p50k_edit = "p50k_edit";
export declare const r50k_base = "r50k_base";
export declare const o200k_base = "o200k_base";
export declare const o200k_harmony = "o200k_harmony";
export declare const DEFAULT_ENCODING = "o200k_base";
export type EncodingName = keyof typeof encodingsMap;
export declare const encodingNames: ["p50k_base", "r50k_base", "p50k_edit", "cl100k_base", "o200k_base", "o200k_harmony"];
/**
 * maps model names to encoding names
 * if a model is not listed, it uses the default encoding for new models
 * which is `o200k_base`
 */
export declare const modelToEncodingMap: Record<ModelName, EncodingName>;
export interface ChatParameters {
    messageSeparator: string;
    roleSeparator: string;
}
export type ModelName = keyof typeof import('./models.js');
export type ChatModelName = (typeof chatEnabledModels)[number];
export declare const chatModelParams: Record<ChatModelName, ChatParameters>;
