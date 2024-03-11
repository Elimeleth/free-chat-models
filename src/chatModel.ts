import "tslib";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseLanguageModelParams } from "@langchain/core/language_models/base";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { BaseMessage } from "@langchain/core/messages";
import { ChatGenerationChunk, ChatResult } from "@langchain/core/outputs";

import { ApiOptions, OptionsModel } from "./types";
import { _convertDeltaToMessageChunk, chatCompletions, convertMessagesToParams } from "./utils";

/*

THIS CLASS EXTENDS FROM LANGCHIAN BaseChatModel

For general comprehension this class is an partial copy from OpenAi class
and return a OpenAi BaseChatModel

this class don't support function_call and tooling

this class is an adapter and using GPTFREE like client request

*/



class ChatModel extends BaseChatModel {
    constructor(
        private model_name: OptionsModel, 
        private fields?: ApiOptions) {
        super(fields as any ?? {})
        console.warn('This library is not fully tested and it may not work as expected.')
        if (!this.model_name) throw Error('SET A MODEL NAME')
        if (this.model_name !== 'Bing') throw Error(`${this.model_name} NOT SUPPORTED YET`)
    }

    invocationParams() {
        const params = {
            model: this.model_name,
            markdown: true,
            data: {
                system_message: this.fields?.data?.system_message || '',
                temperature: this.fields?.data?.temperature || 0.9,
                max_tokens: this.fields?.data?.max_tokens || 100,
                top_p: this.fields?.data?.top_p || 0.6,
                repetition_penalty: this.fields?.data?.repetition_penalty || 1.2,
                ...this.fields?.data
            },
            stream: false,
        };
        return params;
    }

    _combineLLMOutput?(...llmOutputs: (Record<string, any> | undefined)[]): Record<string, any> | undefined {
        return llmOutputs.reduce((acc, llmOutput) => {
            if (llmOutput && llmOutput.tokenUsage) {
                if (!acc?.tokenUsage) {
                    // @ts-ignore
                    acc.tokenUsage = {
                        completionTokens: 0,
                        promptTokens: 0,
                        totalTokens: 0,
                    };

                    return acc
                }
                acc.tokenUsage.completionTokens +=
                    llmOutput.tokenUsage.completionTokens ?? 0;
                acc.tokenUsage.promptTokens += llmOutput.tokenUsage.promptTokens ?? 0;
                acc.tokenUsage.totalTokens += llmOutput.tokenUsage.totalTokens ?? 0;
            }
            return acc;
        }, {
            tokenUsage: {
                completionTokens: 0,
                promptTokens: 0,
                totalTokens: 0,
            },
        });
    }

    _llmType(): string {
        return 'experimental';
    }

    get callKeys(): string[] {
        return [
            ...super.callKeys,
            "options",
            "promptIndex",
            "response_format",
            "seed",
        ];
    }

    async completionWithRetry(messages: any[], options: any) {
        return this.caller.call(async () => {
            try {
                const data = await chatCompletions(
                    messages,
                    options
                )
                return data;
            }
            catch (error) {
                if (error.response) {
                    // The request was made and the server responded with a status code
                    // that falls out of the range of 2xx
                    return error.response
                } else if (error.request) {
                    // The request was made but no response was received
                    // `error.request` is an instance of XMLHttpRequest in the browser and an instance of
                    // http.ClientRequest in node.js
                    return error.request
                } 
                
                return error.message
    
            }
        });
    }

    async _generate(messages: BaseMessage[], options: this["ParsedCallOptions"], runManager?: CallbackManagerForLLMRun | undefined): Promise<ChatResult> {
        const stream = this._streamResponseChunks(messages, options, runManager) as any
        const finalChunks = {};
        
        for await (const chunk of stream) {
            const index = chunk.generationInfo?.completion ?? 0;
            if (finalChunks[index] === undefined) {
                finalChunks[index] = chunk;
            } else {
                finalChunks[index] = finalChunks[index].concat(chunk);
            }
        }
    
        const generations: any = Object.entries(finalChunks)
            .sort(([aKey], [bKey]) => parseInt(aKey, 10) - parseInt(bKey, 10))
            .map(([_, value]) => value);
    
        // OpenAI does not support token usage report under stream mode,
        // fallback to estimation.
        return { generations, llmOutput: { estimatedTokenUsage: 0 } };
    }

    async *_streamResponseChunks(messages: BaseMessage[], options: this["ParsedCallOptions"], runManager?: CallbackManagerForLLMRun | undefined) {
        const messagesMapped = convertMessagesToParams(messages);
        const opts = this.invocationParams()
        
        const data = await this.completionWithRetry(messagesMapped, opts);
        
        const delta = {
            role: 'assistant',
            content: data
        }

        const chunk = _convertDeltaToMessageChunk(delta);

        const newTokenIndices = {
            prompt: 0,
            completion: 0,
        };

        const generationChunk = new ChatGenerationChunk({
            message: chunk,
            text: chunk.content as any,
            generationInfo: newTokenIndices,
        });
        yield generationChunk;
        // eslint-disable-next-line no-void
        void runManager?.handleLLMNewToken(generationChunk.text ?? "", newTokenIndices, undefined, undefined, undefined, { chunk: generationChunk });
        if (options.signal?.aborted) {
            throw new Error("AbortError");
        }
    }

}

export { ChatModel };