export type HarmonyTerminator = '<|end|>' | '<|return|>' | '<|call|>';
export interface ChatMessageFunctionCall {
    name?: string;
    arguments?: string;
}
export interface ChatMessage {
    role?: 'system' | 'user' | 'assistant' | 'developer' | (string & {});
    name?: string;
    content: string;
    function_call?: ChatMessageFunctionCall;
    /** Harmony-only: channel label such as `analysis`, `commentary`, or `final`. */
    channel?: string;
    /** Harmony-only: recipient metadata, e.g. `functions.get_weather` or `assistant`. */
    recipient?: string;
    /** Controls where the recipient metadata is rendered in Harmony headers. Defaults to `channel`. */
    recipientPlacement?: 'role' | 'channel';
    /** Harmony-only: constraint label, e.g. `json`. */
    constraint?: string;
    /** Harmony-only: overrides the closing token, defaults to `<|end|>`. */
    terminator?: HarmonyTerminator;
}
export interface EncodeChatOptions {
    primeWithAssistantResponse?: string;
}
export type ChatCompletionFunctionType = 'string' | 'integer' | 'number' | 'boolean' | 'null' | 'array' | 'object';
export interface ChatCompletionStringProperty {
    type: Extract<ChatCompletionFunctionType, 'string'>;
    description?: string;
    enum?: string[];
}
export interface ChatCompletionNumberProperty {
    type: Extract<ChatCompletionFunctionType, 'integer' | 'number'>;
    description?: string;
    minimum?: number;
    maximum?: number;
    enum?: (number | string)[];
}
export interface ChatCompletionBooleanProperty {
    type: Extract<ChatCompletionFunctionType, 'boolean'>;
    description?: string;
}
export interface ChatCompletionNullProperty {
    type: Extract<ChatCompletionFunctionType, 'null'>;
    description?: string;
}
export interface ChatCompletionArrayProperty {
    type: Extract<ChatCompletionFunctionType, 'array'>;
    description?: string;
    items?: ChatCompletionFunctionProperty;
}
export interface ChatCompletionObjectProperty {
    type: Extract<ChatCompletionFunctionType, 'object'>;
    description?: string;
    required?: string[];
    properties?: Record<string, ChatCompletionFunctionProperty>;
}
export type ChatCompletionFunctionProperty = ChatCompletionStringProperty | ChatCompletionNumberProperty | ChatCompletionBooleanProperty | ChatCompletionNullProperty | ChatCompletionArrayProperty | ChatCompletionObjectProperty;
export interface ChatCompletionFunctionParameters extends ChatCompletionObjectProperty {
}
export interface ChatCompletionFunctionDefinition {
    name: string;
    description?: string;
    parameters?: ChatCompletionFunctionParameters;
}
export type ChatCompletionFunctionCallOption = 'auto' | 'none' | {
    name: string;
};
export interface ChatCompletionRequest {
    messages: readonly ChatMessage[];
    functions?: readonly ChatCompletionFunctionDefinition[];
    function_call?: ChatCompletionFunctionCallOption;
}
export type CountStringTokens = (text: string) => number;
export declare const MESSAGE_TOKEN_OVERHEAD = 3;
export declare const MESSAGE_NAME_TOKEN_OVERHEAD = 1;
export declare const FUNCTION_ROLE_TOKEN_DISCOUNT = 2;
export declare const FUNCTION_CALL_METADATA_TOKEN_OVERHEAD = 3;
export declare const FUNCTION_DEFINITION_TOKEN_OVERHEAD = 9;
export declare const COMPLETION_REQUEST_TOKEN_OVERHEAD = 3;
export declare const FUNCTION_CALL_NAME_TOKEN_OVERHEAD = 4;
export declare const FUNCTION_CALL_NONE_TOKEN_OVERHEAD = 1;
export declare const SYSTEM_FUNCTION_TOKEN_DEDUCTION = 4;
export declare function countMessageTokens(message: ChatMessage, countStringTokens: CountStringTokens): number;
export declare function formatObjectProperties(obj: ChatCompletionObjectProperty, indent: number, formatType: (param: ChatCompletionFunctionProperty, indent: number) => string): string;
export declare function formatFunctionType(param: ChatCompletionFunctionProperty, indent: number): string;
export declare function formatFunctionDefinitions(functions: readonly ChatCompletionFunctionDefinition[]): string;
export declare function estimateTokensInFunctions(functions: readonly ChatCompletionFunctionDefinition[], countStringTokens: CountStringTokens): number;
export declare function padSystemMessage(message: ChatMessage, hasFunctions: boolean, isSystemPadded: boolean): ChatMessage;
export declare function computeChatCompletionTokenCount(request: ChatCompletionRequest, countStringTokens: CountStringTokens): number;
