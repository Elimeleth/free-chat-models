
export type OptionsModel = 'Bing'
    | 'llama2'
    | 'qwen-b7'
    | 'leo'
    | 'mistral'
    | 'zephyr'

export type ApiOptions = {
    data: ApiData
    markdown: boolean
    stream: boolean
    conversation_style: string
    model: string
}

export type ApiData = {
    system_message: string
    temperature: number
    max_tokens: number
    max_new_tokens: number
    top_p: number
    top_k: number
    repetition_penalty: number
}


export type IMessages = {
    role: string;
    content: string;
}

export type IOptions = {
    model: string;
    prompt: string;
}
