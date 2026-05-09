declare const babbage_002_spec: {
    readonly name: "babbage-002";
    readonly slug: "babbage-002";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning"];
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.4;
            readonly output: 0.4;
        };
        readonly batch: {
            readonly input: 0.2;
            readonly output: 0.2;
        };
    };
};
export { babbage_002_spec as 'babbage-002' };
declare const chatgpt_4o_latest_spec: {
    readonly name: "chatgpt-4o-latest";
    readonly slug: "chatgpt-4o-latest";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "predicted_outputs", "image_input"];
    readonly supported_endpoints: ["chat_completions", "responses"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 5;
            readonly output: 15;
        };
    };
};
export { chatgpt_4o_latest_spec as 'chatgpt-4o-latest' };
declare const codex_mini_latest_spec: {
    readonly name: "codex-mini-latest";
    readonly slug: "codex-mini-latest";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 200000;
    readonly max_output_tokens: 100000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "function_calling", "image_input", "prompt_caching", "evals", "stored_completions"];
    readonly supported_endpoints: ["responses"];
    readonly reasoning_tokens: true;
};
export { codex_mini_latest_spec as 'codex-mini-latest' };
declare const computer_use_preview_2025_03_11_spec: {
    readonly name: "computer-use-preview-2025-03-11";
    readonly slug: "computer-use-preview-2025-03-11";
    readonly performance: 2;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 8192;
    readonly max_output_tokens: 1024;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["function_calling"];
    readonly supported_endpoints: ["responses", "batch"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 3;
            readonly output: 12;
        };
        readonly batch: {
            readonly input: 1.5;
            readonly output: 6;
        };
    };
};
export { computer_use_preview_2025_03_11_spec as 'computer-use-preview-2025-03-11' };
export { computer_use_preview_2025_03_11_spec as 'computer-use-preview' };
declare const dall_e_2_spec: {
    readonly name: "dall-e-2";
    readonly slug: "dall-e-2";
    readonly performance: 1;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["image"];
    };
    readonly supported_endpoints: ["image_generation", "image_edit"];
    readonly supported_features: ["inpainting"];
    readonly reasoning_tokens: false;
};
export { dall_e_2_spec as 'dall-e-2' };
declare const dall_e_3_spec: {
    readonly name: "dall-e-3";
    readonly slug: "dall-e-3";
    readonly performance: 3;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["image"];
    };
    readonly supported_endpoints: ["image_generation"];
    readonly reasoning_tokens: false;
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
};
export { dall_e_3_spec as 'dall-e-3' };
declare const davinci_002_spec: {
    readonly name: "davinci-002";
    readonly slug: "davinci-002";
    readonly performance: 1;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning"];
    readonly supported_endpoints: ["completions"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 2;
            readonly output: 2;
        };
        readonly batch: {
            readonly input: 1;
            readonly output: 1;
        };
    };
};
export { davinci_002_spec as 'davinci-002' };
declare const gpt_3_5_turbo_16k_0613_spec: {
    readonly name: "gpt-3.5-turbo-16k-0613";
    readonly slug: "gpt-3-5-turbo-16k-0613";
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
    readonly supported_endpoints: ["chat_completions", "responses", "batch"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 3;
            readonly output: 4;
        };
        readonly batch: {
            readonly input: 1.5;
            readonly output: 2;
        };
    };
};
export { gpt_3_5_turbo_16k_0613_spec as 'gpt-3.5-turbo-16k-0613' };
declare const gpt_3_5_turbo_instruct_spec: {
    readonly name: "gpt-3.5-turbo-instruct";
    readonly slug: "gpt-3-5-turbo-instruct";
    readonly performance: 1;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 4096;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning"];
    readonly supported_endpoints: ["chat_completions", "responses"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 1.5;
            readonly output: 2;
        };
    };
};
export { gpt_3_5_turbo_instruct_spec as 'gpt-3.5-turbo-instruct' };
declare const gpt_3_5_turbo_0125_spec: {
    readonly name: "gpt-3.5-turbo-0125";
    readonly slug: "gpt-3-5-turbo-0125";
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
    readonly supported_endpoints: ["chat_completions", "responses", "batch", "fine_tuning"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 0.5;
            readonly output: 1.5;
        };
        readonly batch: {
            readonly input: 0.25;
            readonly output: 0.75;
        };
    };
};
export { gpt_3_5_turbo_0125_spec as 'gpt-3.5-turbo-0125' };
declare const gpt_3_5_turbo_1106_spec: {
    readonly name: "gpt-3.5-turbo-1106";
    readonly slug: "gpt-3-5-turbo-1106";
    readonly deprecated: true;
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
    readonly supported_endpoints: ["chat_completions", "responses", "batch", "fine_tuning"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 1;
            readonly output: 2;
        };
        readonly batch: {
            readonly input: 0.5;
            readonly output: 1;
        };
    };
};
export { gpt_3_5_turbo_1106_spec as 'gpt-3.5-turbo-1106' };
export { gpt_3_5_turbo_0125_spec as 'gpt-3.5-turbo' };
declare const gpt_4_0125_preview_spec: {
    readonly name: "gpt-4-0125-preview";
    readonly slug: "gpt-4-0125-preview";
    readonly deprecated: true;
    readonly performance: 2;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4_0125_preview_spec as 'gpt-4-0125-preview' };
declare const gpt_4_1106_vision_preview_spec: {
    readonly name: "gpt-4-1106-vision-preview";
    readonly slug: "gpt-4-1106-vision-preview";
    readonly performance: 2;
    readonly latency: 3;
    readonly deprecated: true;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning", "streaming"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4_1106_vision_preview_spec as 'gpt-4-1106-vision-preview' };
export { gpt_4_0125_preview_spec as 'gpt-4-turbo-preview' };
declare const gpt_4_turbo_2024_04_09_spec: {
    readonly name: "gpt-4-turbo-2024-04-09";
    readonly slug: "gpt-4-turbo-2024-04-09";
    readonly performance: 2;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "function_calling", "image_input"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4_turbo_2024_04_09_spec as 'gpt-4-turbo-2024-04-09' };
export { gpt_4_turbo_2024_04_09_spec as 'gpt-4-turbo' };
declare const gpt_4_1_mini_2025_04_14_spec: {
    readonly name: "gpt-4.1-mini-2025-04-14";
    readonly slug: "gpt-4.1-mini-2025-04-14";
    readonly performance: 3;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 1047576;
    readonly max_output_tokens: 32768;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["predicted_outputs", "streaming", "function_calling", "fine_tuning", "file_search", "file_uploads", "web_search", "structured_outputs", "image_input"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch", "fine_tuning"];
    readonly reasoning_tokens: false;
};
export { gpt_4_1_mini_2025_04_14_spec as 'gpt-4.1-mini-2025-04-14' };
export { gpt_4_1_mini_2025_04_14_spec as 'gpt-4.1-mini' };
declare const gpt_4_1_nano_2025_04_14_spec: {
    readonly name: "gpt-4.1-nano-2025-04-14";
    readonly slug: "gpt-4.1-nano-2025-04-14";
    readonly performance: 2;
    readonly latency: 5;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 1047576;
    readonly max_output_tokens: 32768;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["predicted_outputs", "streaming", "function_calling", "file_search", "file_uploads", "structured_outputs", "image_input", "prompt_caching", "fine_tuning"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch", "fine_tuning"];
    readonly reasoning_tokens: false;
};
export { gpt_4_1_nano_2025_04_14_spec as 'gpt-4.1-nano-2025-04-14' };
export { gpt_4_1_nano_2025_04_14_spec as 'gpt-4.1-nano' };
declare const gpt_4_1_2025_04_14_spec: {
    readonly name: "gpt-4.1-2025-04-14";
    readonly slug: "gpt-4.1-2025-04-14";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 1047576;
    readonly max_output_tokens: 32768;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "predicted_outputs", "distillation", "function_calling", "file_search", "file_uploads", "image_input", "web_search", "fine_tuning", "prompt_caching"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch", "fine_tuning"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4_1_2025_04_14_spec as 'gpt-4.1-2025-04-14' };
export { gpt_4_1_2025_04_14_spec as 'gpt-4.1' };
declare const gpt_4_5_preview_2025_02_27_spec: {
    readonly name: "gpt-4.5-preview-2025-02-27";
    readonly slug: "gpt-4.5-preview-2025-02-27";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["function_calling", "structured_outputs", "streaming", "system_messages", "evals", "prompt_caching", "image_input"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch"];
    readonly reasoning_tokens: false;
};
export { gpt_4_5_preview_2025_02_27_spec as 'gpt-4.5-preview-2025-02-27' };
export { gpt_4_5_preview_2025_02_27_spec as 'gpt-4.5-preview' };
declare const gpt_4_0613_spec: {
    readonly name: "gpt-4-0613";
    readonly slug: "gpt-4-0613";
    readonly performance: 2;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8192;
    readonly max_output_tokens: 8192;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning", "streaming"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch", "fine_tuning"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 30;
            readonly output: 60;
        };
        readonly batch: {
            readonly input: 15;
            readonly output: 30;
        };
    };
};
export { gpt_4_0613_spec as 'gpt-4-0613' };
declare const gpt_4_0314_spec: {
    readonly name: "gpt-4-0314";
    readonly slug: "gpt-4-0314";
    readonly deprecated: true;
    readonly performance: 2;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 8192;
    readonly max_output_tokens: 8192;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["fine_tuning", "streaming"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants"];
    readonly reasoning_tokens: false;
    readonly price_data: {
        readonly main: {
            readonly input: 30;
            readonly output: 60;
        };
        readonly batch: {
            readonly input: 15;
            readonly output: 30;
        };
    };
};
export { gpt_4_0314_spec as 'gpt-4-0314' };
export { gpt_4_0613_spec as 'gpt-4' };
declare const gpt_4o_audio_preview_2025_06_03_spec: {
    readonly name: "gpt-4o-audio-preview-2025-06-03";
    readonly slug: "gpt-4o-audio-preview-2025-06-03";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "function_calling"];
    readonly supported_endpoints: ["chat_completions"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_audio_preview_2025_06_03_spec as 'gpt-4o-audio-preview-2025-06-03' };
declare const gpt_4o_audio_preview_2024_12_17_spec: {
    readonly name: "gpt-4o-audio-preview-2024-12-17";
    readonly slug: "gpt-4o-audio-preview-2024-12-17";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "function_calling"];
    readonly supported_endpoints: ["chat_completions"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_audio_preview_2024_12_17_spec as 'gpt-4o-audio-preview-2024-12-17' };
declare const gpt_4o_audio_preview_2024_10_01_spec: {
    readonly name: "gpt-4o-audio-preview-2024-10-01";
    readonly slug: "gpt-4o-audio-preview-2024-10-01";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "function_calling"];
    readonly supported_endpoints: ["chat_completions"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_audio_preview_2024_10_01_spec as 'gpt-4o-audio-preview-2024-10-01' };
export { gpt_4o_audio_preview_2025_06_03_spec as 'gpt-4o-audio-preview' };
declare const gpt_4o_mini_audio_preview_2024_12_17_spec: {
    readonly name: "gpt-4o-mini-audio-preview-2024-12-17";
    readonly slug: "gpt-4o-mini-audio-preview-2024-12-17";
    readonly performance: 2;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "function_calling"];
    readonly supported_endpoints: ["chat_completions"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_mini_audio_preview_2024_12_17_spec as 'gpt-4o-mini-audio-preview-2024-12-17' };
export { gpt_4o_mini_audio_preview_2024_12_17_spec as 'gpt-4o-mini-audio-preview' };
declare const gpt_4o_mini_realtime_preview_2024_12_17_spec: {
    readonly name: "gpt-4o-mini-realtime-preview-2024-12-17";
    readonly slug: "gpt-4o-mini-realtime-preview-2024-12-17";
    readonly performance: 2;
    readonly latency: 5;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 16000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["function_calling", "prompt_caching"];
    readonly supported_endpoints: ["realtime"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_mini_realtime_preview_2024_12_17_spec as 'gpt-4o-mini-realtime-preview-2024-12-17' };
export { gpt_4o_mini_realtime_preview_2024_12_17_spec as 'gpt-4o-mini-realtime-preview' };
declare const gpt_4o_mini_search_preview_2025_03_11_spec: {
    readonly name: "gpt-4o-mini-search-preview-2025-03-11";
    readonly slug: "gpt-4o-mini-search-preview-2025-03-11";
    readonly performance: 2;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "image_input"];
    readonly supported_endpoints: ["chat_completions"];
    readonly reasoning_tokens: false;
};
export { gpt_4o_mini_search_preview_2025_03_11_spec as 'gpt-4o-mini-search-preview-2025-03-11' };
export { gpt_4o_mini_search_preview_2025_03_11_spec as 'gpt-4o-mini-search-preview' };
declare const gpt_4o_mini_transcribe_spec: {
    readonly name: "gpt-4o-mini-transcribe";
    readonly slug: "gpt-4o-mini-transcribe";
    readonly performance: 3;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["audio", "text"];
        readonly output: ["text"];
    };
    readonly context_window: 16000;
    readonly max_output_tokens: 2000;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["transcription", "realtime"];
    readonly reasoning_tokens: false;
};
export { gpt_4o_mini_transcribe_spec as 'gpt-4o-mini-transcribe' };
declare const gpt_4o_mini_tts_spec: {
    readonly name: "gpt-4o-mini-tts";
    readonly slug: "gpt-4o-mini-tts";
    readonly performance: 4;
    readonly latency: 4;
    readonly current_snapshot: "gpt-4o-mini-tts";
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["audio"];
    };
    readonly supported_endpoints: ["speech_generation"];
    readonly reasoning_tokens: false;
};
export { gpt_4o_mini_tts_spec as 'gpt-4o-mini-tts' };
declare const gpt_4o_mini_2024_07_18_spec: {
    readonly name: "gpt-4o-mini-2024-07-18";
    readonly slug: "gpt-4o-mini-2024-07-18";
    readonly performance: 2;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["predicted_outputs", "streaming", "function_calling", "fine_tuning", "file_search", "file_uploads", "web_search", "structured_outputs", "image_input"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch", "fine_tuning"];
    readonly reasoning_tokens: false;
};
export { gpt_4o_mini_2024_07_18_spec as 'gpt-4o-mini-2024-07-18' };
export { gpt_4o_mini_2024_07_18_spec as 'gpt-4o-mini' };
declare const gpt_4o_realtime_preview_2025_06_03_spec: {
    readonly name: "gpt-4o-realtime-preview-2025-06-03";
    readonly slug: "gpt-4o-realtime-preview-2025-06-03";
    readonly performance: 3;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 32000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["function_calling", "prompt_caching"];
    readonly supported_endpoints: ["realtime"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_realtime_preview_2025_06_03_spec as 'gpt-4o-realtime-preview-2025-06-03' };
declare const gpt_4o_realtime_preview_2024_12_17_spec: {
    readonly name: "gpt-4o-realtime-preview-2024-12-17";
    readonly slug: "gpt-4o-realtime-preview-2024-12-17";
    readonly performance: 3;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 16000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["function_calling", "prompt_caching"];
    readonly supported_endpoints: ["realtime"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_realtime_preview_2024_12_17_spec as 'gpt-4o-realtime-preview-2024-12-17' };
declare const gpt_4o_realtime_preview_2024_10_01_spec: {
    readonly name: "gpt-4o-realtime-preview-2024-10-01";
    readonly slug: "gpt-4o-realtime-preview-2024-10-01";
    readonly performance: 2;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 16000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["function_calling", "prompt_caching"];
    readonly supported_endpoints: ["realtime"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_realtime_preview_2024_10_01_spec as 'gpt-4o-realtime-preview-2024-10-01' };
export { gpt_4o_realtime_preview_2025_06_03_spec as 'gpt-4o-realtime-preview' };
declare const gpt_4o_search_preview_2025_03_11_spec: {
    readonly name: "gpt-4o-search-preview-2025-03-11";
    readonly slug: "gpt-4o-search-preview-2025-03-11";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "image_input"];
    readonly supported_endpoints: ["chat_completions"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_search_preview_2025_03_11_spec as 'gpt-4o-search-preview-2025-03-11' };
export { gpt_4o_search_preview_2025_03_11_spec as 'gpt-4o-search-preview' };
declare const gpt_4o_transcribe_spec: {
    readonly name: "gpt-4o-transcribe";
    readonly slug: "gpt-4o-transcribe";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["audio", "text"];
        readonly output: ["text"];
    };
    readonly context_window: 16000;
    readonly max_output_tokens: 2000;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["transcription", "realtime"];
    readonly reasoning_tokens: false;
};
export { gpt_4o_transcribe_spec as 'gpt-4o-transcribe' };
declare const gpt_4o_2024_11_20_spec: {
    readonly name: "gpt-4o-2024-11-20";
    readonly slug: "gpt-4o-2024-11-20";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "predicted_outputs", "distillation", "function_calling", "file_search", "file_uploads", "image_input", "web_search"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_2024_11_20_spec as 'gpt-4o-2024-11-20' };
declare const gpt_4o_2024_08_06_spec: {
    readonly name: "gpt-4o-2024-08-06";
    readonly slug: "gpt-4o-2024-08-06";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "predicted_outputs", "distillation", "file_search", "file_uploads", "fine_tuning", "function_calling", "image_input", "web_search"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch", "fine_tuning"];
    readonly reasoning_tokens: false;
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
};
export { gpt_4o_2024_08_06_spec as 'gpt-4o-2024-08-06' };
declare const gpt_4o_2024_05_13_spec: {
    readonly name: "gpt-4o-2024-05-13";
    readonly slug: "gpt-4o-2024-05-13";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "function_calling", "fine_tuning", "file_search", "file_uploads", "image_input", "web_search", "predicted_outputs"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch"];
    readonly reasoning_tokens: false;
};
export { gpt_4o_2024_05_13_spec as 'gpt-4o-2024-05-13' };
export { gpt_4o_2024_08_06_spec as 'gpt-4o' };
declare const gpt_5_chat_latest_spec: {
    readonly name: "gpt-5-chat-latest";
    readonly slug: "gpt-5-chat-latest";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly max_input_tokens: 272000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["function_calling", "structured_outputs", "streaming", "image_input"];
    readonly supported_endpoints: ["chat_completions", "responses"];
    readonly reasoning_tokens: false;
};
export { gpt_5_chat_latest_spec as 'gpt-5-chat-latest' };
declare const gpt_5_codex_spec: {
    readonly name: "gpt-5-codex";
    readonly slug: "gpt-5-codex";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 400000;
    readonly max_output_tokens: 128000;
    readonly max_input_tokens: 272000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "function_calling", "image_input", "web_search", "prompt_caching"];
    readonly supported_endpoints: ["responses"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 1.25;
            readonly output: 10;
        };
    };
};
export { gpt_5_codex_spec as 'gpt-5-codex' };
declare const gpt_5_mini_2025_08_07_spec: {
    readonly name: "gpt-5-mini-2025-08-07";
    readonly slug: "gpt-5-mini-2025-08-07";
    readonly performance: 3;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 400000;
    readonly max_output_tokens: 128000;
    readonly max_input_tokens: 272000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "function_calling", "file_search", "file_uploads", "web_search", "structured_outputs", "image_input"];
    readonly supported_endpoints: ["chat_completions", "responses", "batch"];
    readonly reasoning_tokens: true;
};
export { gpt_5_mini_2025_08_07_spec as 'gpt-5-mini-2025-08-07' };
export { gpt_5_mini_2025_08_07_spec as 'gpt-5-mini' };
declare const gpt_5_nano_2025_08_07_spec: {
    readonly name: "gpt-5-nano-2025-08-07";
    readonly slug: "gpt-5-nano-2025-08-07";
    readonly performance: 2;
    readonly latency: 5;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 400000;
    readonly max_output_tokens: 128000;
    readonly max_input_tokens: 272000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "function_calling", "file_search", "file_uploads", "structured_outputs", "image_input", "prompt_caching"];
    readonly supported_endpoints: ["chat_completions", "responses", "batch"];
    readonly reasoning_tokens: true;
};
export { gpt_5_nano_2025_08_07_spec as 'gpt-5-nano-2025-08-07' };
export { gpt_5_nano_2025_08_07_spec as 'gpt-5-nano' };
declare const gpt_5_pro_2025_10_06_spec: {
    readonly name: "gpt-5-pro-2025-10-06";
    readonly slug: "gpt-5-pro-2025-10-06";
    readonly performance: 5;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly supported_features: ["structured_outputs", "function_calling", "file_search", "file_uploads", "image_input", "web_search"];
    readonly supported_endpoints: ["responses", "batch"];
    readonly context_window: 400000;
    readonly max_output_tokens: 272000;
    readonly knowledge_cutoff: Date;
    readonly reasoning_tokens: true;
};
export { gpt_5_pro_2025_10_06_spec as 'gpt-5-pro-2025-10-06' };
export { gpt_5_pro_2025_10_06_spec as 'gpt-5-pro' };
declare const gpt_5_2025_08_07_spec: {
    readonly name: "gpt-5-2025-08-07";
    readonly slug: "gpt-5-2025-08-07";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 400000;
    readonly max_output_tokens: 128000;
    readonly max_input_tokens: 272000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "distillation", "function_calling", "file_search", "file_uploads", "image_input", "web_search", "prompt_caching"];
    readonly supported_endpoints: ["chat_completions", "responses", "batch"];
    readonly reasoning_tokens: true;
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
};
export { gpt_5_2025_08_07_spec as 'gpt-5-2025-08-07' };
export { gpt_5_2025_08_07_spec as 'gpt-5' };
declare const gpt_audio_mini_2025_10_06_spec: {
    readonly name: "gpt-audio-mini-2025-10-06";
    readonly performance: 4;
    readonly latency: 5;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly supported_endpoints: ["chat_completions"];
    readonly supported_features: ["function_calling", "prompt_caching"];
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly reasoning_tokens: false;
};
export { gpt_audio_mini_2025_10_06_spec as 'gpt-audio-mini-2025-10-06' };
export { gpt_audio_mini_2025_10_06_spec as 'gpt-audio-mini' };
declare const gpt_audio_2025_08_28_spec: {
    readonly name: "gpt-audio-2025-08-28";
    readonly slug: "gpt-audio-2025-08-28";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 16384;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "function_calling"];
    readonly supported_endpoints: ["chat_completions"];
    readonly reasoning_tokens: false;
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
};
export { gpt_audio_2025_08_28_spec as 'gpt-audio-2025-08-28' };
export { gpt_audio_2025_08_28_spec as 'gpt-audio' };
declare const gpt_image_1_mini_spec: {
    readonly name: "gpt-image-1-mini";
    readonly performance: 4;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["image"];
    };
    readonly supported_endpoints: ["image_edit", "image_generation"];
};
export { gpt_image_1_mini_spec as 'gpt-image-1-mini' };
declare const gpt_image_1_spec: {
    readonly name: "gpt-image-1";
    readonly slug: "gpt-image-1";
    readonly performance: 4;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["image"];
    };
    readonly supported_endpoints: ["image_generation", "image_edit"];
    readonly supported_features: ["inpainting"];
    readonly reasoning_tokens: false;
};
export { gpt_image_1_spec as 'gpt-image-1' };
declare const gpt_oss_120b_spec: {
    readonly name: "gpt-oss-120b";
    readonly slug: "gpt-oss-120b";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 131072;
    readonly max_output_tokens: 131072;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "function_calling"];
    readonly supported_endpoints: ["responses", "batch"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 0.0005;
            readonly cached_output: 0.0001;
            readonly output: 0.0015;
        };
        readonly batch: {
            readonly input: 0.0004;
            readonly output: 0.0012;
        };
    };
};
export { gpt_oss_120b_spec as 'gpt-oss-120b' };
declare const gpt_oss_20b_spec: {
    readonly name: "gpt-oss-20b";
    readonly slug: "gpt-oss-20b";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 131072;
    readonly max_output_tokens: 131072;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "function_calling"];
    readonly supported_endpoints: ["responses", "batch"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 0.0002;
            readonly cached_output: 0.00005;
            readonly output: 0.0006;
        };
        readonly batch: {
            readonly input: 0.00015;
            readonly output: 0.0005;
        };
    };
};
export { gpt_oss_20b_spec as 'gpt-oss-20b' };
declare const gpt_realtime_mini_2025_10_06_spec: {
    readonly name: "gpt-realtime-mini-2025-10-06";
    readonly performance: 4;
    readonly latency: 5;
    readonly modalities: {
        readonly input: ["text", "image", "audio"];
        readonly output: ["text", "audio"];
    };
    readonly supported_endpoints: ["realtime"];
    readonly supported_features: ["function_calling", "prompt_caching"];
    readonly context_window: 32000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly reasoning_tokens: false;
};
export { gpt_realtime_mini_2025_10_06_spec as 'gpt-realtime-mini-2025-10-06' };
export { gpt_realtime_mini_2025_10_06_spec as 'gpt-realtime-mini' };
declare const gpt_realtime_2025_08_28_spec: {
    readonly name: "gpt-realtime-2025-08-28";
    readonly slug: "gpt-realtime-2025-08-28";
    readonly performance: 5;
    readonly latency: 4;
    readonly modalities: {
        readonly input: ["text", "audio", "image"];
        readonly output: ["text", "audio"];
    };
    readonly context_window: 32000;
    readonly max_output_tokens: 4096;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["function_calling", "prompt_caching"];
    readonly supported_endpoints: ["realtime"];
    readonly reasoning_tokens: false;
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
};
export { gpt_realtime_2025_08_28_spec as 'gpt-realtime-2025-08-28' };
export { gpt_realtime_2025_08_28_spec as 'gpt-realtime' };
declare const o1_mini_2024_09_12_spec: {
    readonly name: "o1-mini-2024-09-12";
    readonly slug: "o1-mini-2024-09-12";
    readonly performance: 3;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 65536;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "file_search", "file_uploads"];
    readonly supported_endpoints: ["chat_completions", "assistants"];
    readonly reasoning_tokens: true;
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
};
export { o1_mini_2024_09_12_spec as 'o1-mini-2024-09-12' };
export { o1_mini_2024_09_12_spec as 'o1-mini' };
declare const o1_preview_2024_09_12_spec: {
    readonly name: "o1-preview-2024-09-12";
    readonly slug: "o1-preview-2024-09-12";
    readonly performance: 3;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 128000;
    readonly max_output_tokens: 32768;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "file_search", "function_calling", "file_uploads"];
    readonly supported_endpoints: ["chat_completions", "assistants"];
    readonly reasoning_tokens: true;
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
};
export { o1_preview_2024_09_12_spec as 'o1-preview-2024-09-12' };
export { o1_preview_2024_09_12_spec as 'o1-preview' };
declare const o1_pro_2025_03_19_spec: {
    readonly name: "o1-pro-2025-03-19";
    readonly slug: "o1-pro-2025-03-19";
    readonly performance: 4;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 200000;
    readonly max_output_tokens: 100000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["structured_outputs", "function_calling", "image_input"];
    readonly supported_endpoints: ["responses", "batch"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 150;
            readonly output: 600;
        };
        readonly batch: {
            readonly input: 75;
            readonly output: 300;
        };
    };
};
export { o1_pro_2025_03_19_spec as 'o1-pro-2025-03-19' };
export { o1_pro_2025_03_19_spec as 'o1-pro' };
declare const o1_2024_12_17_spec: {
    readonly name: "o1-2024-12-17";
    readonly slug: "o1-2024-12-17";
    readonly performance: 4;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 200000;
    readonly max_output_tokens: 100000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "file_search", "function_calling", "file_uploads", "image_input"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 15;
            readonly cached_output: 7.5;
            readonly output: 60;
        };
        readonly batch: {
            readonly input: 7.5;
            readonly output: 30;
        };
    };
};
export { o1_2024_12_17_spec as 'o1-2024-12-17' };
export { o1_2024_12_17_spec as 'o1' };
declare const o3_deep_research_2025_06_26_spec: {
    readonly name: "o3-deep-research-2025-06-26";
    readonly slug: "o3-deep-research-2025-06-26";
    readonly performance: 5;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 200000;
    readonly max_output_tokens: 100000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "file_uploads", "image_input", "prompt_caching", "evals", "stored_completions"];
    readonly supported_endpoints: ["responses", "batch"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 15;
            readonly cached_output: 7.5;
            readonly output: 60;
        };
        readonly batch: {
            readonly input: 7.5;
            readonly output: 30;
        };
    };
};
export { o3_deep_research_2025_06_26_spec as 'o3-deep-research-2025-06-26' };
export { o3_deep_research_2025_06_26_spec as 'o3-deep-research' };
declare const o3_mini_2025_01_31_spec: {
    readonly name: "o3-mini-2025-01-31";
    readonly slug: "o3-mini-2025-01-31";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly context_window: 200000;
    readonly max_output_tokens: 100000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "function_calling", "file_search", "file_uploads"];
    readonly supported_endpoints: ["chat_completions", "responses", "assistants", "batch"];
    readonly reasoning_tokens: true;
};
export { o3_mini_2025_01_31_spec as 'o3-mini-2025-01-31' };
export { o3_mini_2025_01_31_spec as 'o3-mini' };
declare const o3_pro_2025_06_10_spec: {
    readonly name: "o3-pro-2025-06-10";
    readonly slug: "o3-pro-2025-06-10";
    readonly performance: 5;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 200000;
    readonly max_output_tokens: 100000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["structured_outputs", "function_calling", "image_input"];
    readonly supported_endpoints: ["responses", "batch"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 20;
            readonly output: 80;
        };
        readonly batch: {
            readonly input: 10;
            readonly output: 40;
        };
    };
};
export { o3_pro_2025_06_10_spec as 'o3-pro-2025-06-10' };
export { o3_pro_2025_06_10_spec as 'o3-pro' };
declare const o3_2025_04_16_spec: {
    readonly name: "o3-2025-04-16";
    readonly slug: "o3-2025-04-16";
    readonly performance: 5;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 200000;
    readonly max_output_tokens: 100000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "file_search", "function_calling", "file_uploads", "image_input", "prompt_caching", "evals", "stored_completions"];
    readonly supported_endpoints: ["chat_completions", "responses", "batch"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 15;
            readonly cached_output: 7.5;
            readonly output: 60;
        };
        readonly batch: {
            readonly input: 7.5;
            readonly output: 30;
        };
    };
};
export { o3_2025_04_16_spec as 'o3-2025-04-16' };
export { o3_2025_04_16_spec as 'o3' };
declare const o4_mini_deep_research_2025_06_26_spec: {
    readonly name: "o4-mini-deep-research-2025-06-26";
    readonly slug: "o4-mini-deep-research-2025-06-26";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 200000;
    readonly max_output_tokens: 100000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "file_uploads", "image_input", "prompt_caching", "evals", "stored_completions"];
    readonly supported_endpoints: ["responses", "batch"];
    readonly reasoning_tokens: true;
    readonly price_data: {
        readonly main: {
            readonly input: 8.25;
            readonly cached_output: 2.0625;
            readonly output: 33;
        };
        readonly batch: {
            readonly input: 4.125;
            readonly output: 16.5;
        };
    };
};
export { o4_mini_deep_research_2025_06_26_spec as 'o4-mini-deep-research-2025-06-26' };
export { o4_mini_deep_research_2025_06_26_spec as 'o4-mini-deep-research' };
declare const o4_mini_2025_04_16_spec: {
    readonly name: "o4-mini-2025-04-16";
    readonly slug: "o4-mini-2025-04-16";
    readonly performance: 4;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly context_window: 200000;
    readonly max_output_tokens: 100000;
    readonly knowledge_cutoff: Date;
    readonly supported_features: ["streaming", "structured_outputs", "function_calling", "file_search", "file_uploads", "image_input", "prompt_caching", "evals", "stored_completions", "fine_tuning"];
    readonly supported_endpoints: ["chat_completions", "responses", "batch", "fine_tuning"];
    readonly reasoning_tokens: true;
};
export { o4_mini_2025_04_16_spec as 'o4-mini-2025-04-16' };
export { o4_mini_2025_04_16_spec as 'o4-mini' };
declare const omni_moderation_2024_09_26_spec: {
    readonly name: "omni-moderation-2024-09-26";
    readonly slug: "omni-moderation-2024-09-26";
    readonly performance: 3;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["text"];
    };
    readonly supported_endpoints: ["moderation"];
    readonly reasoning_tokens: false;
    readonly supported_features: ["image_input"];
};
export { omni_moderation_2024_09_26_spec as 'omni-moderation-2024-09-26' };
export { omni_moderation_2024_09_26_spec as 'omni-moderation-latest' };
declare const sora_2_pro_spec: {
    readonly name: "sora-2-pro";
    readonly slug: "sora-2-pro";
    readonly performance: 5;
    readonly latency: 1;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["video", "audio"];
    };
    readonly supported_endpoints: ["videos"];
    readonly reasoning_tokens: false;
};
export { sora_2_pro_spec as 'sora-2-pro' };
declare const sora_2_spec: {
    readonly name: "sora-2";
    readonly slug: "sora-2";
    readonly performance: 4;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text", "image"];
        readonly output: ["video", "audio"];
    };
    readonly supported_endpoints: ["videos"];
    readonly reasoning_tokens: false;
};
export { sora_2_spec as 'sora-2' };
declare const text_embedding_3_large_spec: {
    readonly name: "text-embedding-3-large";
    readonly slug: "text-embedding-3-large";
    readonly performance: 3;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly supported_endpoints: ["embeddings", "batch"];
};
export { text_embedding_3_large_spec as 'text-embedding-3-large' };
declare const text_embedding_3_small_spec: {
    readonly name: "text-embedding-3-small";
    readonly slug: "text-embedding-3-small";
    readonly performance: 2;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly supported_endpoints: ["embeddings", "batch"];
};
export { text_embedding_3_small_spec as 'text-embedding-3-small' };
declare const text_embedding_ada_002_spec: {
    readonly name: "text-embedding-ada-002";
    readonly slug: "text-embedding-ada-002";
    readonly performance: 1;
    readonly latency: 2;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly supported_endpoints: ["embeddings", "batch"];
};
export { text_embedding_ada_002_spec as 'text-embedding-ada-002' };
declare const text_moderation_007_spec: {
    readonly name: "text-moderation-007";
    readonly slug: "text-moderation-007";
    readonly performance: 2;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["text"];
    };
    readonly max_output_tokens: 32768;
    readonly knowledge_cutoff: Date;
    readonly supported_endpoints: ["moderation"];
    readonly reasoning_tokens: false;
};
export { text_moderation_007_spec as 'text-moderation-007' };
export { text_moderation_007_spec as 'text-moderation-latest' };
export { text_moderation_007_spec as 'text-moderation-stable' };
declare const tts_1_hd_spec: {
    readonly name: "tts-1-hd";
    readonly slug: "tts-1-hd";
    readonly performance: 3;
    readonly latency: 3;
    readonly current_snapshot: "tts-1-hd";
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["audio"];
    };
    readonly supported_endpoints: ["speech_generation"];
    readonly reasoning_tokens: false;
};
export { tts_1_hd_spec as 'tts-1-hd' };
declare const tts_1_spec: {
    readonly name: "tts-1";
    readonly slug: "tts-1";
    readonly performance: 2;
    readonly latency: 4;
    readonly current_snapshot: "tts-1";
    readonly modalities: {
        readonly input: ["text"];
        readonly output: ["audio"];
    };
    readonly supported_endpoints: ["speech_generation"];
    readonly reasoning_tokens: false;
};
export { tts_1_spec as 'tts-1' };
declare const whisper_1_spec: {
    readonly name: "whisper-1";
    readonly slug: "whisper-1";
    readonly performance: 2;
    readonly latency: 3;
    readonly modalities: {
        readonly input: ["audio"];
        readonly output: ["text"];
    };
    readonly supported_endpoints: ["transcription", "translation"];
    readonly reasoning_tokens: false;
};
export { whisper_1_spec as 'whisper-1' };
