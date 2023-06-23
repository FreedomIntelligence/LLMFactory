# LLMFactory
A factory to standardize and modularize  training of vertical LLMs

# Objective
End users could train their own large langauge models through **LLMFactory** without coding. The only thing users have to do is  to  percisely discribe  their need.
 
# For End Users

## Steps to get an adapted model for users
- Select a backbone, e.g., Llama and Bloom
- Optionaly select some **knowlege** modules, each is used to inject knowledge in specific field.
- Select some **function** modules, e.g., coding, medical advices, math, etc.
- Select some reward models.

After 30 mins, you will get a url to download your model weights and a serving url.

# For Developers

## data

### pretraining data

- collect plain data
- classify these data
- train Lora modules for each backbone

### finetuning data:
- distill data (converation and instruction) from GPT4 (converation from chatgpt since it is cheaper)
- collect human instruction/converation fron online source or real-scenaiors.
- classify these instruction/converation data
- quality ranking
- filtering strategies (diversity)

### reward models
- modularize reward models

We do not directly sell data, we sell models.

### plugins/tools

…

## auto-testing
- MMLU
- C-Eval


# current stage
- use a 500M bloom as a demo

# TODO list

- automatically read documents (tables/images) and extract QA pairs.


