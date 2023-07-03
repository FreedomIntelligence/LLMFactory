# LLMFactory
A factory to standardize and modularize  training of customized LLMs

# Objective
End users could train their own large langauge models through **LLMFactory** without coding. The only thing users have to do is  to  percisely discribe  their need.
 
# For End Users

## Steps to get an adapted model for users
- Select a backbone, e.g., Llama and Bloom
- Optionaly select some **knowlege** modules, each is used to inject knowledge in specific field. Or one could upload their own data.
- Select some **function** modules, e.g., coding, medical advices, math, etc.
- Select some reward models.



After 30 mins, you will get a url to download your model weights and a serving url.

# Getting started
## Installation
To get started, follow these steps to install the required packages:
1. Clone the repository:
```bash
git clone https://github.com/FreedomIntelligence/LLMFactory.git
cd LLMFactory
```
2. Install the package:
```bash
pip install .
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configure Local Resource
To configure the local resources, follow these steps:
1. Edit the Factory resource configuration file:
   - Open the file **`factory/resource.json`**.
   - Locate your local models and data.
   - Make the necessary changes.
  
2. Edit the training script template:
   - Open the file llmfactory/constants.py.
   - Adapt the script to match your actual gpt-resource environment, such as nnodes and nproc_per_node.

By following these steps, you will be able to set up and configure the necessary resources for the project.

# For Developers

```
import llmfactory

# Configure the resource in the factory/resource.json file
factory = llmfactory.Factory()

# Show available models
factory.show_available_model()
# Output:
# [Bloom]: bloom-560m, bloomz-560m, bloom-1b1, bloomz-1b1, bloomz-7b1-mt
# [Llama]: llama-7b-hf, llama-13b-hf
# [Baichuan]: baichuan-7B

# Show available data
factory.show_available_data()
# Output:
# [Local]: music, computer, medical

# Select a model from the available model set
model_config = factory.create_backbone("bloom-560m")

# Set up the data configuration
data_config = factory.prepare_data_for_training(num_data=50, data_ratios={"music": 0.4, "computer": 0.6})

# Train a new model based on the existing model and data configuration
model_config = factory.train_model(model_config, data_config, save_name="test")

# Deploy the model on the command line
factory.deploy_model_cli(model_config)

# Deploy the model using Gradio
factory.deploy_model_gradio(model_config)
```
## data


### RAG

### pretraining data (less is more, smaler models consumes less data)

- collect plain data
- classify these data
- train Lora modules for each backbone
- if you choose two Lora moduels, some further data (mixed with two domains) should be used to further pretraining
- one could upload data

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


# current stage v0.01
- use a 560M bloom as a demo;
- add ModelFactory, DataFactory for simple model/data selection.

# TODO list

- automatically read documents (tables/images) and extract QA pairs.
- parameter-efficent deployment
- an interface to upload our own json file

# Acknowledgement
- The code is mainly develped based on [LLMZoo](https://github.com/FreedomIntelligence/LLMZoo).
