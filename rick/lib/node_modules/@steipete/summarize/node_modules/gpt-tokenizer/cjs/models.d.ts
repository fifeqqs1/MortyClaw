export * from './models.gen.js';
declare const gpt_3_5_0301_spec: {
    readonly name: "gpt-3.5-0301";
    readonly slug: "gpt-3-5-0301";
    readonly supported_endpoints: ["chat_completions", "responses"];
    readonly price_data: {
        readonly main: {
            readonly input: 1.5;
            readonly output: 2;
        };
        readonly batch: {
            readonly input: 0.75;
            readonly output: 1;
        };
    };
    readonly performance: 1;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 16385;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning"];
    readonly reasoning_tokens: false;
};
export { gpt_3_5_0301_spec as 'gpt-3.5-0301' };
export { gpt_3_5_0301_spec as 'gpt-3.5' };
declare const gpt_3_5_turbo_0613_spec: {
    readonly name: "gpt-3.5-turbo-0613";
    readonly slug: "gpt-3-5-turbo-0613";
    readonly supported_endpoints: ["chat_completions", "responses", "batch"];
    readonly price_data: {
        readonly main: {
            readonly input: 1.5;
            readonly output: 2;
        };
        readonly batch: {
            readonly input: 0.75;
            readonly output: 1;
        };
    };
    readonly performance: 1;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 16385;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning"];
    readonly reasoning_tokens: false;
};
export { gpt_3_5_turbo_0613_spec as 'gpt-3.5-turbo-0613' };
declare const gpt_4_1106_preview_spec: {
    readonly name: "gpt-4-1106-preview";
    readonly slug: "gpt-4-1106-preview";
    readonly performance: 2;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 4096;
    readonly supported_features: ["fine_tuning"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants"];
    readonly price_data: {
        readonly main: {
            readonly input: 10;
            readonly output: 30;
        };
        readonly batch: {
            readonly input: 5;
            readonly output: 15;
        };
    };
    readonly knowledge_cutoff: Date;
    readonly reasoning_tokens: false;
};
export { gpt_4_1106_preview_spec as 'gpt-4-1106-preview' };
declare const gpt_4_32k_spec: {
    readonly name: "gpt-4-32k";
    readonly slug: "gpt-4-32k";
    readonly context_window: 32768;
    readonly max_output_tokens: 8192;
    readonly supported_endpoints: ["chat_completions", "responses", "assistants"];
    readonly price_data: {
        readonly main: {
            readonly input: 60;
            readonly output: 120;
        };
        readonly batch: {
            readonly input: 30;
            readonly output: 60;
        };
    };
    readonly performance: 2;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning", "streaming"];
    readonly reasoning_tokens: false;
};
export { gpt_4_32k_spec as 'gpt-4-32k' };
declare const text_ada_001_spec: {
    readonly name: "text-ada-001";
    readonly slug: "text-ada-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.4;
        };
    };
};
export { text_ada_001_spec as 'text-ada-001' };
declare const text_babbage_001_spec: {
    readonly name: "text-babbage-001";
    readonly slug: "text-babbage-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.5;
        };
    };
};
export { text_babbage_001_spec as 'text-babbage-001' };
declare const text_curie_001_spec: {
    readonly name: "text-curie-001";
    readonly slug: "text-curie-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 2;
        };
    };
};
export { text_curie_001_spec as 'text-curie-001' };
declare const text_davinci_001_spec: {
    readonly name: "text-davinci-001";
    readonly slug: "text-davinci-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
        };
    };
};
export { text_davinci_001_spec as 'text-davinci-001' };
declare const text_davinci_002_spec: {
    readonly name: "text-davinci-002";
    readonly slug: "text-davinci-002";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 4000;
    readonly max_output_tokens: 4000;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
        };
    };
};
export { text_davinci_002_spec as 'text-davinci-002' };
declare const text_davinci_003_spec: {
    readonly name: "text-davinci-003";
    readonly slug: "text-davinci-003";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 4000;
    readonly max_output_tokens: 4000;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
        };
    };
};
export { text_davinci_003_spec as 'text-davinci-003' };
declare const ada_spec: {
    readonly name: "ada";
    readonly slug: "ada";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.4;
        };
    };
};
export { ada_spec as 'ada' };
declare const babbage_spec: {
    readonly name: "babbage";
    readonly slug: "babbage";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.5;
        };
    };
};
export { babbage_spec as 'babbage' };
declare const curie_spec: {
    readonly name: "curie";
    readonly slug: "curie";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 2;
        };
    };
};
export { curie_spec as 'curie' };
declare const davinci_spec: {
    readonly name: "davinci";
    readonly slug: "davinci";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
        };
    };
};
export { davinci_spec as 'davinci' };
declare const code_davinci_001_spec: {
    readonly name: "code-davinci-001";
    readonly slug: "code-davinci-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8000;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
        };
    };
};
export { code_davinci_001_spec as 'code-davinci-001' };
declare const code_davinci_002_spec: {
    readonly name: "code-davinci-002";
    readonly slug: "code-davinci-002";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8000;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
        };
    };
};
export { code_davinci_002_spec as 'code-davinci-002' };
declare const davinci_codex_spec: {
    readonly name: "davinci-codex";
    readonly slug: "davinci-codex";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8000;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
        };
    };
};
export { davinci_codex_spec as 'davinci-codex' };
declare const code_davinci_edit_001_spec: {
    readonly name: "code-davinci-edit-001";
    readonly slug: "code-davinci-edit-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
        };
    };
};
export { code_davinci_edit_001_spec as 'code-davinci-edit-001' };
declare const code_cushman_001_spec: {
    readonly name: "code-cushman-001";
    readonly slug: "code-cushman-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 2;
        };
    };
};
export { code_cushman_001_spec as 'code-cushman-001' };
declare const code_cushman_002_spec: {
    readonly name: "code-cushman-002";
    readonly slug: "code-cushman-002";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 2;
        };
    };
};
export { code_cushman_002_spec as 'code-cushman-002' };
declare const cushman_codex_spec: {
    readonly name: "cushman-codex";
    readonly slug: "cushman-codex";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 2;
        };
    };
};
export { cushman_codex_spec as 'cushman-codex' };
declare const code_search_ada_code_001_spec: {
    readonly name: "code-search-ada-code-001";
    readonly slug: "code-search-ada-code-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.4;
        };
    };
};
export { code_search_ada_code_001_spec as 'code-search-ada-code-001' };
declare const code_search_ada_text_001_spec: {
    readonly name: "code-search-ada-text-001";
    readonly slug: "code-search-ada-text-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.4;
        };
    };
};
export { code_search_ada_text_001_spec as 'code-search-ada-text-001' };
declare const text_davinci_edit_001_spec: {
    readonly name: "text-davinci-edit-001";
    readonly slug: "text-davinci-edit-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 2048;
    readonly max_output_tokens: 2048;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
        };
    };
};
export { text_davinci_edit_001_spec as 'text-davinci-edit-001' };
declare const text_similarity_ada_001_spec: {
    readonly name: "text-similarity-ada-001";
    readonly slug: "text-similarity-ada-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.4;
        };
    };
};
export { text_similarity_ada_001_spec as 'text-similarity-ada-001' };
declare const text_search_ada_doc_001_spec: {
    readonly name: "text-search-ada-doc-001";
    readonly slug: "text-search-ada-doc-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.4;
        };
    };
};
export { text_search_ada_doc_001_spec as 'text-search-ada-doc-001' };
declare const text_search_ada_query_001_spec: {
    readonly name: "text-search-ada-query-001";
    readonly slug: "text-search-ada-query-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.4;
        };
    };
};
export { text_search_ada_query_001_spec as 'text-search-ada-query-001' };
declare const text_similarity_babbage_001_spec: {
    readonly name: "text-similarity-babbage-001";
    readonly slug: "text-similarity-babbage-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.5;
        };
    };
};
export { text_similarity_babbage_001_spec as 'text-similarity-babbage-001' };
declare const text_search_babbage_doc_001_spec: {
    readonly name: "text-search-babbage-doc-001";
    readonly slug: "text-search-babbage-doc-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.5;
        };
    };
};
export { text_search_babbage_doc_001_spec as 'text-search-babbage-doc-001' };
declare const text_search_babbage_query_001_spec: {
    readonly name: "text-search-babbage-query-001";
    readonly slug: "text-search-babbage-query-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.5;
        };
    };
};
export { text_search_babbage_query_001_spec as 'text-search-babbage-query-001' };
declare const code_search_babbage_code_001_spec: {
    readonly name: "code-search-babbage-code-001";
    readonly slug: "code-search-babbage-code-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.5;
        };
    };
};
export { code_search_babbage_code_001_spec as 'code-search-babbage-code-001' };
declare const code_search_babbage_text_001_spec: {
    readonly name: "code-search-babbage-text-001";
    readonly slug: "code-search-babbage-text-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.5;
        };
    };
};
export { code_search_babbage_text_001_spec as 'code-search-babbage-text-001' };
declare const text_similarity_curie_001_spec: {
    readonly name: "text-similarity-curie-001";
    readonly slug: "text-similarity-curie-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 2;
        };
    };
};
export { text_similarity_curie_001_spec as 'text-similarity-curie-001' };
declare const text_search_curie_doc_001_spec: {
    readonly name: "text-search-curie-doc-001";
    readonly slug: "text-search-curie-doc-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 2;
        };
    };
};
export { text_search_curie_doc_001_spec as 'text-search-curie-doc-001' };
declare const text_search_curie_query_001_spec: {
    readonly name: "text-search-curie-query-001";
    readonly slug: "text-search-curie-query-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 2;
        };
    };
};
export { text_search_curie_query_001_spec as 'text-search-curie-query-001' };
declare const text_similarity_davinci_001_spec: {
    readonly name: "text-similarity-davinci-001";
    readonly slug: "text-similarity-davinci-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 200;
        };
    };
};
export { text_similarity_davinci_001_spec as 'text-similarity-davinci-001' };
declare const text_search_davinci_doc_001_spec: {
    readonly name: "text-search-davinci-doc-001";
    readonly slug: "text-search-davinci-doc-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 200;
        };
    };
};
export { text_search_davinci_doc_001_spec as 'text-search-davinci-doc-001' };
declare const text_search_davinci_query_001_spec: {
    readonly name: "text-search-davinci-query-001";
    readonly slug: "text-search-davinci-query-001";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8191;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["embeddings"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 200;
        };
    };
};
export { text_search_davinci_query_001_spec as 'text-search-davinci-query-001' };
