"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SYSTEM_FUNCTION_TOKEN_DEDUCTION = exports.FUNCTION_CALL_NONE_TOKEN_OVERHEAD = exports.FUNCTION_CALL_NAME_TOKEN_OVERHEAD = exports.COMPLETION_REQUEST_TOKEN_OVERHEAD = exports.FUNCTION_DEFINITION_TOKEN_OVERHEAD = exports.FUNCTION_CALL_METADATA_TOKEN_OVERHEAD = exports.FUNCTION_ROLE_TOKEN_DISCOUNT = exports.MESSAGE_NAME_TOKEN_OVERHEAD = exports.MESSAGE_TOKEN_OVERHEAD = void 0;
exports.countMessageTokens = countMessageTokens;
exports.formatObjectProperties = formatObjectProperties;
exports.formatFunctionType = formatFunctionType;
exports.formatFunctionDefinitions = formatFunctionDefinitions;
exports.estimateTokensInFunctions = estimateTokensInFunctions;
exports.padSystemMessage = padSystemMessage;
exports.computeChatCompletionTokenCount = computeChatCompletionTokenCount;
exports.MESSAGE_TOKEN_OVERHEAD = 3;
exports.MESSAGE_NAME_TOKEN_OVERHEAD = 1;
exports.FUNCTION_ROLE_TOKEN_DISCOUNT = 2;
exports.FUNCTION_CALL_METADATA_TOKEN_OVERHEAD = 3;
exports.FUNCTION_DEFINITION_TOKEN_OVERHEAD = 9;
exports.COMPLETION_REQUEST_TOKEN_OVERHEAD = 3;
exports.FUNCTION_CALL_NAME_TOKEN_OVERHEAD = 4;
exports.FUNCTION_CALL_NONE_TOKEN_OVERHEAD = 1;
exports.SYSTEM_FUNCTION_TOKEN_DEDUCTION = 4;
const NEWLINE = '\n';
function countMessageTokens(message, countStringTokens) {
    let tokens = 0;
    if (message.role) {
        tokens += countStringTokens(message.role);
    }
    if (message.content) {
        tokens += countStringTokens(message.content);
    }
    if (message.name) {
        tokens += countStringTokens(message.name) + exports.MESSAGE_NAME_TOKEN_OVERHEAD;
    }
    if (message.function_call) {
        const { name, arguments: args } = message.function_call;
        if (name) {
            tokens += countStringTokens(name);
        }
        if (args) {
            tokens += countStringTokens(args);
        }
        tokens += exports.FUNCTION_CALL_METADATA_TOKEN_OVERHEAD;
    }
    tokens += exports.MESSAGE_TOKEN_OVERHEAD;
    if (message.role === 'function') {
        tokens -= exports.FUNCTION_ROLE_TOKEN_DISCOUNT;
    }
    return tokens;
}
function formatObjectProperties(obj, indent, formatType) {
    if (!obj.properties) {
        return '';
    }
    const lines = [];
    const requiredParams = new Set(obj.required ?? []);
    const indentString = ' '.repeat(indent);
    for (const [name, param] of Object.entries(obj.properties)) {
        if (param.description && indent < 2) {
            lines.push(`${indentString}// ${param.description}`);
        }
        const isRequired = requiredParams.has(name);
        const formattedType = formatType(param, indent);
        lines.push(`${indentString}${name}${isRequired ? '' : '?'}: ${formattedType},`);
    }
    return lines.join('\n');
}
function formatFunctionType(param, indent) {
    switch (param.type) {
        case 'string':
            return (param.enum?.map((value) => JSON.stringify(value)).join(' | ') ??
                'string');
        case 'integer':
        case 'number':
            return param.enum?.map((value) => `${value}`).join(' | ') ?? 'number';
        case 'boolean':
            return 'boolean';
        case 'null':
            return 'null';
        case 'array':
            return param.items
                ? `${formatFunctionType(param.items, indent)}[]`
                : 'any[]';
        case 'object': {
            const inner = formatObjectProperties(param, indent + 2, formatFunctionType);
            const closingIndent = ' '.repeat(indent);
            return `{
${inner}
${closingIndent}}`;
        }
        default:
            return 'any';
    }
}
function formatFunctionDefinitions(functions) {
    const lines = ['namespace functions {', ''];
    for (const fn of functions) {
        if (fn.description) {
            lines.push(`// ${fn.description}`);
        }
        const { parameters } = fn;
        const properties = parameters?.properties;
        if (!parameters || !properties || Object.keys(properties).length === 0) {
            lines.push(`type ${fn.name} = () => any;`);
        }
        else {
            lines.push(`type ${fn.name} = (_: {`);
            const formattedProperties = formatObjectProperties(parameters, 0, formatFunctionType);
            if (formattedProperties.length > 0) {
                lines.push(formattedProperties);
            }
            lines.push('}) => any;');
        }
        lines.push('');
    }
    lines.push('} // namespace functions');
    return lines.join('\n');
}
function estimateTokensInFunctions(functions, countStringTokens) {
    const formatted = formatFunctionDefinitions(functions);
    let tokens = countStringTokens(formatted);
    tokens += exports.FUNCTION_DEFINITION_TOKEN_OVERHEAD;
    return tokens;
}
function padSystemMessage(message, hasFunctions, isSystemPadded) {
    if (!hasFunctions || isSystemPadded || message.role !== 'system') {
        return message;
    }
    if (!message.content || message.content.endsWith(NEWLINE)) {
        return message;
    }
    return {
        ...message,
        content: `${message.content}${NEWLINE}`,
    };
}
function computeChatCompletionTokenCount(request, countStringTokens) {
    const { messages, functions, function_call: functionCall } = request;
    const hasFunctions = Boolean(functions && functions.length > 0);
    let paddedSystem = false;
    let total = 0;
    for (const message of messages) {
        const messageToCount = padSystemMessage(message, hasFunctions, paddedSystem);
        if (messageToCount !== message && message.role === 'system') {
            paddedSystem = true;
        }
        else if (message.role === 'system' && hasFunctions && !paddedSystem) {
            paddedSystem = true;
        }
        total += countMessageTokens(messageToCount, countStringTokens);
    }
    total += exports.COMPLETION_REQUEST_TOKEN_OVERHEAD;
    if (hasFunctions && functions) {
        total += estimateTokensInFunctions(functions, countStringTokens);
        if (messages.some((message) => message.role === 'system')) {
            total -= exports.SYSTEM_FUNCTION_TOKEN_DEDUCTION;
        }
    }
    if (functionCall && functionCall !== 'auto') {
        if (functionCall === 'none') {
            total += exports.FUNCTION_CALL_NONE_TOKEN_OVERHEAD;
        }
        else if (typeof functionCall === 'object' && functionCall.name) {
            total +=
                countStringTokens(functionCall.name) + exports.FUNCTION_CALL_NAME_TOKEN_OVERHEAD;
        }
    }
    return total;
}
//# sourceMappingURL=functionCalling.js.map