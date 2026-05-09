"use strict";
/* eslint-disable no-magic-numbers */
/* eslint-disable @typescript-eslint/no-unused-vars */
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
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports["text-search-davinci-query-001"] = exports["text-search-davinci-doc-001"] = exports["text-similarity-davinci-001"] = exports["text-search-curie-query-001"] = exports["text-search-curie-doc-001"] = exports["text-similarity-curie-001"] = exports["code-search-babbage-text-001"] = exports["code-search-babbage-code-001"] = exports["text-search-babbage-query-001"] = exports["text-search-babbage-doc-001"] = exports["text-similarity-babbage-001"] = exports["text-search-ada-query-001"] = exports["text-search-ada-doc-001"] = exports["text-similarity-ada-001"] = exports["text-davinci-edit-001"] = exports["code-search-ada-text-001"] = exports["code-search-ada-code-001"] = exports["cushman-codex"] = exports["code-cushman-002"] = exports["code-cushman-001"] = exports["code-davinci-edit-001"] = exports["davinci-codex"] = exports["code-davinci-002"] = exports["code-davinci-001"] = exports["davinci"] = exports["curie"] = exports["babbage"] = exports["ada"] = exports["text-davinci-003"] = exports["text-davinci-002"] = exports["text-davinci-001"] = exports["text-curie-001"] = exports["text-babbage-001"] = exports["text-ada-001"] = exports["gpt-4-32k"] = exports["gpt-4-1106-preview"] = exports["gpt-3.5-turbo-0613"] = exports["gpt-3.5"] = exports["gpt-3.5-0301"] = void 0;
const models_gen_js_1 = require("./models.gen.js");
// export all codegen-based models:
__exportStar(require("./models.gen.js"), exports);
//
// --- BELOW ARE MODELS THAT WERE MISSING FROM DATASET, BUT PRESENT IN "OTHER MODELS" SECTION ---
// https://platform.openai.com/docs/pricing#other-models
//
//  - gpt-3.5-0301
const gpt_3_5_0301_spec = {
    ...models_gen_js_1["gpt-3.5-turbo-0125"],
    name: 'gpt-3.5-0301',
    slug: 'gpt-3-5-0301',
    supported_endpoints: ['chat_completions', 'responses'],
    price_data: {
        main: { input: 1.5, output: 2 },
        batch: { input: 0.75, output: 1 },
    },
};
exports["gpt-3.5-0301"] = gpt_3_5_0301_spec;
exports["gpt-3.5"] = gpt_3_5_0301_spec;
//  - gpt-3.5-turbo-0613
const gpt_3_5_turbo_0613_spec = {
    ...models_gen_js_1["gpt-3.5-turbo-0125"],
    name: 'gpt-3.5-turbo-0613',
    slug: 'gpt-3-5-turbo-0613',
    supported_endpoints: ['chat_completions', 'responses', 'batch'],
    price_data: {
        main: { input: 1.5, output: 2 },
        batch: { input: 0.75, output: 1 },
    },
};
exports["gpt-3.5-turbo-0613"] = gpt_3_5_turbo_0613_spec;
//  - gpt-4-1106-preview
const gpt_4_1106_preview_spec = {
    ...models_gen_js_1["gpt-4-0613"],
    name: 'gpt-4-1106-preview',
    slug: 'gpt-4-1106-preview',
    performance: 2,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 128e3,
    max_output_tokens: 4_096,
    supported_features: ['fine_tuning'],
    supported_endpoints: ['chat_completions', 'responses', 'assistants'],
    price_data: {
        main: { input: 10, output: 30 },
        batch: { input: 5, output: 15 },
    },
};
exports["gpt-4-1106-preview"] = gpt_4_1106_preview_spec;
//  - gpt-4-32k
const gpt_4_32k_config = {
    name: 'gpt-4-32k',
    slug: 'gpt-4-32k',
    display_name: 'GPT-4-32k',
    current_snapshot: 'gpt-4-32k',
    tagline: 'Legacy GPT-4 model with a 32k context window',
    description: 'Legacy version of GPT-4 with a 32,768 token context window.',
    type: 'chat',
    snapshots: ['gpt-4-32k'],
    point_to: 'gpt-4o',
    deprecated: true,
};
const gpt_4_32k_spec = {
    ...models_gen_js_1["gpt-4-0613"],
    name: 'gpt-4-32k',
    slug: 'gpt-4-32k',
    context_window: 32_768,
    max_output_tokens: 8_192,
    supported_endpoints: [
        'chat_completions',
        'responses',
        'assistants',
    ],
    price_data: {
        main: { input: 60, output: 120 },
        batch: { input: 30, output: 60 },
    },
};
exports["gpt-4-32k"] = gpt_4_32k_spec;
//
// --- BELOW ARE LEGACY, NO LONGER SUPPORTED MODELS ---
//
// --- text-ada-001 ---
const text_ada_config = {
    name: 'text-ada',
    slug: 'text-ada',
    display_name: 'Text Ada 001',
    current_snapshot: 'text-ada-001',
    tagline: 'Text Ada 001',
    description: 'Model for lightweight tasks.',
    type: 'other', // Legacy completions model
    snapshots: ['text-ada-001'],
    point_to: 'gpt-3.5-turbo-instruct',
    deprecated: true,
};
const text_ada_001_spec = {
    name: 'text-ada-001',
    slug: 'text-ada-001',
    performance: 1, // Assuming lowest performance for old models
    latency: 3, // Assuming higher latency for old models
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.4 } },
};
exports["text-ada-001"] = text_ada_001_spec;
// --- text-babbage-001 ---
const text_babbage_config = {
    name: 'text-babbage',
    slug: 'text-babbage',
    display_name: 'Text Babbage 001',
    current_snapshot: 'text-babbage-001',
    tagline: 'Text Babbage 001',
    description: 'Model for efficient processing.',
    type: 'other', // Legacy completions model
    snapshots: ['text-babbage-001'],
    point_to: 'gpt-3.5-turbo-instruct',
    deprecated: true,
};
const text_babbage_001_spec = {
    name: 'text-babbage-001',
    slug: 'text-babbage-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.5 } },
};
exports["text-babbage-001"] = text_babbage_001_spec;
// --- text-curie-001 ---
const text_curie_config = {
    name: 'text-curie',
    slug: 'text-curie',
    display_name: 'Text Curie 001',
    current_snapshot: 'text-curie-001',
    tagline: 'Text Curie 001',
    description: 'Mid-range model for various tasks.',
    type: 'other', // Legacy completions model
    snapshots: ['text-curie-001'],
    point_to: 'gpt-3.5-turbo-instruct',
    deprecated: true,
};
const text_curie_001_spec = {
    name: 'text-curie-001',
    slug: 'text-curie-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 2 } },
};
exports["text-curie-001"] = text_curie_001_spec;
// --- text-davinci (001, 002, 003) ---
const text_davinci_config = {
    name: 'text-davinci',
    slug: 'text-davinci',
    display_name: 'Text Davinci',
    current_snapshot: 'text-davinci-003', // Points to the latest of these legacy ones
    tagline: 'Legacy high-performance text generation models',
    description: 'Legacy high-performance model for complex tasks. Includes 001, 002, and 003 versions.',
    type: 'other', // Legacy completions model
    snapshots: ['text-davinci-003', 'text-davinci-002', 'text-davinci-001'],
    point_to: 'gpt-3.5-turbo-instruct',
    deprecated: true,
};
const text_davinci_001_spec = {
    name: 'text-davinci-001',
    slug: 'text-davinci-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 20 } },
};
exports["text-davinci-001"] = text_davinci_001_spec;
const text_davinci_002_spec = {
    name: 'text-davinci-002',
    slug: 'text-davinci-002',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 4_000, // text-davinci-002/003 had larger context
    max_output_tokens: 4_000,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 20 } },
};
exports["text-davinci-002"] = text_davinci_002_spec;
const text_davinci_003_spec = {
    name: 'text-davinci-003',
    slug: 'text-davinci-003',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 4_000,
    max_output_tokens: 4_000,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 20 } },
};
exports["text-davinci-003"] = text_davinci_003_spec;
// --- ada ---
const ada_config = {
    name: 'ada',
    slug: 'ada',
    display_name: 'Ada',
    current_snapshot: 'ada', // No numbered version, so snapshot is 'ada'
    tagline: 'Ada - Base model for lightweight tasks',
    description: 'Base model for lightweight tasks.',
    type: 'other', // Legacy base model for completions
    snapshots: ['ada'],
    point_to: 'babbage-002',
    deprecated: true,
};
const ada_spec = {
    name: 'ada', // Spec name is 'ada'
    slug: 'ada',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.4 } },
};
exports["ada"] = ada_spec;
// --- babbage ---
const babbage_config = {
    name: 'babbage',
    slug: 'babbage',
    display_name: 'Babbage',
    current_snapshot: 'babbage',
    tagline: 'Babbage - Model for efficient processing',
    description: 'Model for efficient processing.',
    type: 'other', // Legacy base model for completions
    snapshots: ['babbage'],
    point_to: 'babbage-002',
    deprecated: true,
};
const babbage_spec = {
    name: 'babbage',
    slug: 'babbage',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.5 } },
};
exports["babbage"] = babbage_spec;
// --- curie ---
const curie_config = {
    name: 'curie',
    slug: 'curie',
    display_name: 'Curie',
    current_snapshot: 'curie',
    tagline: 'Curie - Mid-range model for a variety of applications',
    description: 'Mid-range model for a variety of applications.',
    type: 'other', // Legacy base model for completions
    snapshots: ['curie'],
    point_to: 'davinci-002',
    deprecated: true,
};
const curie_spec = {
    name: 'curie',
    slug: 'curie',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 2 } },
};
exports["curie"] = curie_spec;
// --- davinci ---
const davinci_config = {
    name: 'davinci',
    slug: 'davinci',
    display_name: 'Davinci',
    current_snapshot: 'davinci',
    tagline: 'Davinci - High-performance legacy model',
    description: 'High-performance legacy model.',
    type: 'other', // Legacy base model for completions
    snapshots: ['davinci'],
    point_to: 'davinci-002',
    deprecated: true,
};
const davinci_spec = {
    name: 'davinci',
    slug: 'davinci',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 20 } },
};
exports["davinci"] = davinci_spec;
// --- code-davinci (001, 002) ---
const code_davinci_config = {
    name: 'code-davinci',
    slug: 'code-davinci',
    display_name: 'Code Davinci',
    current_snapshot: 'code-davinci-002',
    tagline: 'Legacy code generation models (Davinci series)',
    description: 'Legacy coding model. Includes 001 and 002 versions.',
    type: 'other', // Code-specific
    snapshots: ['code-davinci-002', 'code-davinci-001'],
    point_to: 'gpt-4o',
    deprecated: true,
};
const code_davinci_001_spec = {
    name: 'code-davinci-001',
    slug: 'code-davinci-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] }, // Code is text
    context_window: 8_000,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 20 } },
};
exports["code-davinci-001"] = code_davinci_001_spec;
const code_davinci_002_spec = {
    name: 'code-davinci-002',
    slug: 'code-davinci-002',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_000,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 20 } },
};
exports["code-davinci-002"] = code_davinci_002_spec;
// --- davinci-codex ---
const davinci_codex_config = {
    name: 'davinci-codex', // Exact match
    slug: 'davinci-codex',
    display_name: 'Code Davinci 001 (davinci-codex)',
    current_snapshot: 'davinci-codex',
    tagline: 'Alias for Code Davinci 001.',
    description: 'Alias for Code Davinci 001. Older coding model.',
    type: 'other', // Code-specific
    snapshots: ['davinci-codex'],
    point_to: 'gpt-4o',
    deprecated: true,
};
const davinci_codex_spec = {
    name: 'davinci-codex', // Spec name matches snapshot
    slug: 'davinci-codex',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_000, // Same as code-davinci-001
    max_output_tokens: 2_048, // Same as code-davinci-001
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 20 } },
};
exports["davinci-codex"] = davinci_codex_spec;
// --- code-davinci-edit-001 ---
const code_davinci_edit_config = {
    name: 'code-davinci-edit',
    slug: 'code-davinci-edit',
    display_name: 'Code Davinci Edit 001',
    current_snapshot: 'code-davinci-edit-001',
    tagline: 'Older coding edit model.',
    description: 'Older coding edit model.',
    type: 'other', // Edit-specific
    snapshots: ['code-davinci-edit-001'],
    deprecated: true,
};
const code_davinci_edit_001_spec = {
    name: 'code-davinci-edit-001',
    slug: 'code-davinci-edit-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048, // Edit models often had smaller context
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'], // No specific 'edit' endpoint in new list
    reasoning_tokens: false,
    price_data: { main: { input: 20 } },
};
exports["code-davinci-edit-001"] = code_davinci_edit_001_spec;
// --- code-cushman (001, 002) ---
const code_cushman_config = {
    name: 'code-cushman',
    slug: 'code-cushman',
    display_name: 'Code Cushman',
    current_snapshot: 'code-cushman-002',
    tagline: 'Legacy code generation models (Cushman series)',
    description: 'Legacy model for coding. Includes 001 and 002 versions.',
    type: 'other', // Code-specific
    snapshots: ['code-cushman-002', 'code-cushman-001'],
    point_to: 'gpt-4o',
    deprecated: true,
};
const code_cushman_001_spec = {
    name: 'code-cushman-001',
    slug: 'code-cushman-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 2 } },
};
exports["code-cushman-001"] = code_cushman_001_spec;
const code_cushman_002_spec = {
    name: 'code-cushman-002',
    slug: 'code-cushman-002',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 2 } },
};
exports["code-cushman-002"] = code_cushman_002_spec;
// --- cushman-codex ---
const cushman_codex_config = {
    name: 'cushman-codex', // Exact match
    slug: 'cushman-codex',
    display_name: 'Code Cushman 001 (cushman-codex)',
    current_snapshot: 'cushman-codex',
    tagline: 'Alias for Code Cushman 001.',
    description: 'Alias for Code Cushman 001. Older model for coding tasks.',
    type: 'other', // Code-specific
    snapshots: ['cushman-codex'],
    point_to: 'gpt-4o',
    deprecated: true,
};
const cushman_codex_spec = {
    name: 'cushman-codex', // Spec name matches snapshot
    slug: 'cushman-codex',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048, // Same as code-cushman-001
    max_output_tokens: 2_048, // Same as code-cushman-001
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'],
    reasoning_tokens: false,
    price_data: { main: { input: 2 } },
};
exports["cushman-codex"] = cushman_codex_spec;
// --- code-search-ada-code-001 ---
const code_search_ada_code_config = {
    name: 'code-search-ada-code',
    slug: 'code-search-ada-code',
    display_name: 'Code Search Ada Code 001',
    current_snapshot: 'code-search-ada-code-001',
    tagline: 'Embedding model for code search.',
    description: 'Embedding model for code search.',
    type: 'other', // Embedding model
    snapshots: ['code-search-ada-code-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const code_search_ada_code_001_spec = {
    name: 'code-search-ada-code-001',
    slug: 'code-search-ada-code-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] }, // Output is an embedding vector (represented as text)
    context_window: 8_191, // maxInput for embedding models
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.4 } },
};
exports["code-search-ada-code-001"] = code_search_ada_code_001_spec;
// --- code-search-ada-text-001 ---
const code_search_ada_text_config = {
    name: 'code-search-ada-text',
    slug: 'code-search-ada-text',
    display_name: 'Code Search Ada Text 001',
    current_snapshot: 'code-search-ada-text-001',
    tagline: 'Embedding model for text search in code context.',
    description: 'Embedding model for text search in code context.',
    type: 'other', // Embedding model
    snapshots: ['code-search-ada-text-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const code_search_ada_text_001_spec = {
    name: 'code-search-ada-text-001',
    slug: 'code-search-ada-text-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.4 } },
};
exports["code-search-ada-text-001"] = code_search_ada_text_001_spec;
// --- text-davinci-edit-001 ---
const text_davinci_edit_config = {
    name: 'text-davinci-edit',
    slug: 'text-davinci-edit',
    display_name: 'Text Davinci Edit 001',
    current_snapshot: 'text-davinci-edit-001',
    tagline: 'Older text edit model.',
    description: 'Older text edit model.',
    type: 'other', // Edit-specific
    snapshots: ['text-davinci-edit-001'],
    deprecated: true,
};
const text_davinci_edit_001_spec = {
    name: 'text-davinci-edit-001',
    slug: 'text-davinci-edit-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 2_048,
    max_output_tokens: 2_048,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['completions'], // No specific 'edit' endpoint
    reasoning_tokens: false,
    price_data: { main: { input: 20 } },
};
exports["text-davinci-edit-001"] = text_davinci_edit_001_spec;
// --- text-similarity-ada-001 ---
const text_similarity_ada_config = {
    name: 'text-similarity-ada',
    slug: 'text-similarity-ada',
    display_name: 'Text Similarity Ada 001',
    current_snapshot: 'text-similarity-ada-001',
    tagline: 'Embedding model for similarity tasks.',
    description: 'Embedding model for similarity tasks.',
    type: 'other', // Embedding
    snapshots: ['text-similarity-ada-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_similarity_ada_001_spec = {
    name: 'text-similarity-ada-001',
    slug: 'text-similarity-ada-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.4 } },
};
exports["text-similarity-ada-001"] = text_similarity_ada_001_spec;
// --- text-search-ada-doc-001 ---
const text_search_ada_doc_config = {
    name: 'text-search-ada-doc',
    slug: 'text-search-ada-doc',
    display_name: 'Text Search Ada Doc 001',
    current_snapshot: 'text-search-ada-doc-001',
    tagline: 'Embedding model for document search.',
    description: 'Embedding model for document search.',
    type: 'other', // Embedding
    snapshots: ['text-search-ada-doc-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_search_ada_doc_001_spec = {
    name: 'text-search-ada-doc-001',
    slug: 'text-search-ada-doc-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.4 } },
};
exports["text-search-ada-doc-001"] = text_search_ada_doc_001_spec;
// --- text-search-ada-query-001 ---
const text_search_ada_query_config = {
    name: 'text-search-ada-query',
    slug: 'text-search-ada-query',
    display_name: 'Text Search Ada Query 001',
    current_snapshot: 'text-search-ada-query-001',
    tagline: 'Embedding model for query search.',
    description: 'Embedding model for query search.',
    type: 'other', // Embedding
    snapshots: ['text-search-ada-query-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_search_ada_query_001_spec = {
    name: 'text-search-ada-query-001',
    slug: 'text-search-ada-query-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.4 } },
};
exports["text-search-ada-query-001"] = text_search_ada_query_001_spec;
// --- text-similarity-babbage-001 ---
const text_similarity_babbage_config = {
    name: 'text-similarity-babbage',
    slug: 'text-similarity-babbage',
    display_name: 'Text Similarity Babbage 001',
    current_snapshot: 'text-similarity-babbage-001',
    tagline: 'Embedding model for similarity tasks.',
    description: 'Embedding model for similarity tasks.',
    type: 'other', // Embedding
    snapshots: ['text-similarity-babbage-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_similarity_babbage_001_spec = {
    name: 'text-similarity-babbage-001',
    slug: 'text-similarity-babbage-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.5 } },
};
exports["text-similarity-babbage-001"] = text_similarity_babbage_001_spec;
// --- text-search-babbage-doc-001 ---
const text_search_babbage_doc_config = {
    name: 'text-search-babbage-doc',
    slug: 'text-search-babbage-doc',
    display_name: 'Text Search Babbage Doc 001',
    current_snapshot: 'text-search-babbage-doc-001',
    tagline: 'Embedding model for document search.',
    description: 'Embedding model for document search.',
    type: 'other', // Embedding
    snapshots: ['text-search-babbage-doc-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_search_babbage_doc_001_spec = {
    name: 'text-search-babbage-doc-001',
    slug: 'text-search-babbage-doc-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.5 } },
};
exports["text-search-babbage-doc-001"] = text_search_babbage_doc_001_spec;
// --- text-search-babbage-query-001 ---
const text_search_babbage_query_config = {
    name: 'text-search-babbage-query',
    slug: 'text-search-babbage-query',
    display_name: 'Text Search Babbage Query 001',
    current_snapshot: 'text-search-babbage-query-001',
    tagline: 'Embedding model for query search.',
    description: 'Embedding model for query search.',
    type: 'other', // Embedding
    snapshots: ['text-search-babbage-query-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_search_babbage_query_001_spec = {
    name: 'text-search-babbage-query-001',
    slug: 'text-search-babbage-query-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.5 } },
};
exports["text-search-babbage-query-001"] = text_search_babbage_query_001_spec;
// --- code-search-babbage-code-001 ---
const code_search_babbage_code_config = {
    name: 'code-search-babbage-code',
    slug: 'code-search-babbage-code',
    display_name: 'Code Search Babbage Code 001',
    current_snapshot: 'code-search-babbage-code-001',
    tagline: 'Embedding model for code search.',
    description: 'Embedding model for code search.',
    type: 'other', // Embedding
    snapshots: ['code-search-babbage-code-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const code_search_babbage_code_001_spec = {
    name: 'code-search-babbage-code-001',
    slug: 'code-search-babbage-code-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.5 } },
};
exports["code-search-babbage-code-001"] = code_search_babbage_code_001_spec;
// --- code-search-babbage-text-001 ---
const code_search_babbage_text_config = {
    name: 'code-search-babbage-text',
    slug: 'code-search-babbage-text',
    display_name: 'Code Search Babbage Text 001',
    current_snapshot: 'code-search-babbage-text-001',
    tagline: 'Embedding model for text search in code context.',
    description: 'Embedding model for text search in code context.',
    type: 'other', // Embedding
    snapshots: ['code-search-babbage-text-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const code_search_babbage_text_001_spec = {
    name: 'code-search-babbage-text-001',
    slug: 'code-search-babbage-text-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 0.5 } },
};
exports["code-search-babbage-text-001"] = code_search_babbage_text_001_spec;
// --- text-similarity-curie-001 ---
const text_similarity_curie_config = {
    name: 'text-similarity-curie',
    slug: 'text-similarity-curie',
    display_name: 'Text Similarity Curie 001',
    current_snapshot: 'text-similarity-curie-001',
    tagline: 'Embedding model for similarity tasks.',
    description: 'Embedding model for similarity tasks.',
    type: 'other', // Embedding
    snapshots: ['text-similarity-curie-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_similarity_curie_001_spec = {
    name: 'text-similarity-curie-001',
    slug: 'text-similarity-curie-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 2 } },
};
exports["text-similarity-curie-001"] = text_similarity_curie_001_spec;
// --- text-search-curie-doc-001 ---
const text_search_curie_doc_config = {
    name: 'text-search-curie-doc',
    slug: 'text-search-curie-doc',
    display_name: 'Text Search Curie Doc 001',
    current_snapshot: 'text-search-curie-doc-001',
    tagline: 'Embedding model for document search.',
    description: 'Embedding model for document search.',
    type: 'other', // Embedding
    snapshots: ['text-search-curie-doc-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_search_curie_doc_001_spec = {
    name: 'text-search-curie-doc-001',
    slug: 'text-search-curie-doc-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 2 } },
};
exports["text-search-curie-doc-001"] = text_search_curie_doc_001_spec;
// --- text-search-curie-query-001 ---
const text_search_curie_query_config = {
    name: 'text-search-curie-query',
    slug: 'text-search-curie-query',
    display_name: 'Text Search Curie Query 001',
    current_snapshot: 'text-search-curie-query-001',
    tagline: 'Embedding model for query search.',
    description: 'Embedding model for query search.',
    type: 'other', // Embedding
    snapshots: ['text-search-curie-query-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_search_curie_query_001_spec = {
    name: 'text-search-curie-query-001',
    slug: 'text-search-curie-query-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 2 } },
};
exports["text-search-curie-query-001"] = text_search_curie_query_001_spec;
// --- text-similarity-davinci-001 ---
const text_similarity_davinci_config = {
    name: 'text-similarity-davinci',
    slug: 'text-similarity-davinci',
    display_name: 'Text Similarity Davinci 001',
    current_snapshot: 'text-similarity-davinci-001',
    tagline: 'Embedding model for similarity tasks.',
    description: 'Embedding model for similarity tasks.',
    type: 'other', // Embedding
    snapshots: ['text-similarity-davinci-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_similarity_davinci_001_spec = {
    name: 'text-similarity-davinci-001',
    slug: 'text-similarity-davinci-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 200 } },
};
exports["text-similarity-davinci-001"] = text_similarity_davinci_001_spec;
// --- text-search-davinci-doc-001 ---
const text_search_davinci_doc_config = {
    name: 'text-search-davinci-doc',
    slug: 'text-search-davinci-doc',
    display_name: 'Text Search Davinci Doc 001',
    current_snapshot: 'text-search-davinci-doc-001',
    tagline: 'Embedding model for document search.',
    description: 'Embedding model for document search.',
    type: 'other', // Embedding
    snapshots: ['text-search-davinci-doc-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_search_davinci_doc_001_spec = {
    name: 'text-search-davinci-doc-001',
    slug: 'text-search-davinci-doc-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 200 } },
};
exports["text-search-davinci-doc-001"] = text_search_davinci_doc_001_spec;
// --- text-search-davinci-query-001 ---
const text_search_davinci_query_config = {
    name: 'text-search-davinci-query',
    slug: 'text-search-davinci-query',
    display_name: 'Text Search Davinci Query 001',
    current_snapshot: 'text-search-davinci-query-001',
    tagline: 'Embedding model for query search.',
    description: 'Embedding model for query search.',
    type: 'other', // Embedding
    snapshots: ['text-search-davinci-query-001'],
    point_to: 'text-embedding-3-small',
    deprecated: true,
};
const text_search_davinci_query_001_spec = {
    name: 'text-search-davinci-query-001',
    slug: 'text-search-davinci-query-001',
    performance: 1,
    latency: 3,
    modalities: { input: ['text'], output: ['text'] },
    context_window: 8_191,
    knowledge_cutoff: new Date(Date.UTC(2_021, 9 - 1, 1)), // 2021-09
    supported_endpoints: ['embeddings'],
    reasoning_tokens: false,
    price_data: { main: { input: 200 } },
};
exports["text-search-davinci-query-001"] = text_search_davinci_query_001_spec;
//# sourceMappingURL=models.js.map