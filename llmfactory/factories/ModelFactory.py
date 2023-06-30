import random
from .component import ModelType
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

class ModelFactory():
    def __init__(self, model_config):
        self.available_models = dict()
        self.model_config = model_config
        self.setup_config(model_config)

    def setup_config(self, model_config):
        for model_name, config in model_config.items():
            model_type = ModelType.get_modelType(config)
            if model_type is None:
                continue
            config["model_type"] = model_type
            self.available_models[model_name] = config

    def get_all_available(self):
        model_dict = dict()
        for model_name, config in self.available_models.items():
            model_type = config['model_type']
            if model_type not in model_dict:
                model_dict[model_type] = [model_name]
            else:
                model_dict[model_type].append(model_name)
        
        model_descriptions = []
        for model_type, model_names in model_dict.items():
            model_descriptions.append("[{}]: {}".format(model_type.name, ", ".join(model_names)))

        return "\n".join(model_descriptions)

    def create_backbone(self, model_config):
        # try to load model
        model_path = model_config['model_path']
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # config = AutoConfig.from_pretrained(model_path)
            del model
            del tokenizer
            # model_config['config'] = config
            return model_config
        except Exception as e:
            print(f"Failed to loaded model from: {model_path}\n{e}")
            return None