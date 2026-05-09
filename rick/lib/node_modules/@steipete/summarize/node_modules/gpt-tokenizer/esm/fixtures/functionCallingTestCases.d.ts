import type { ChatCompletionRequest } from '../GptEncoding.js';
export type FunctionCallingTestCase = ChatCompletionRequest & {
    tokens: number;
};
export declare const functionCallingTestCases: FunctionCallingTestCase[];
