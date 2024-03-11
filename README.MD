# This is an experimental class based on nexra API, any bad use is your responsability

## Usage

```ts
/* stuff code */

const chain = PromptTemplate.fromTemplate("Answer the user's {question}")
const model = new ChatModel('Bing')

const answer = await chain.pipe(model).invoke({ question: 'Which colors is thes sky' })

console.log(answer.content)
```

## Author

Elimeleth Capuano [https://github.com/elimeleth](https://github.com/elimeleth)
