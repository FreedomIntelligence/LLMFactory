from enum import Enum
import os

class ModelType(Enum):
    Bloom = 1
    Llama = 2
    Baichuan = 3

    @classmethod
    def get_available_types(cls):
        return ModelType.__members__

    @property
    def fsdp_transformer_layer_cls_to_wrap(self):
        if self == ModelType.Bloom:
            return "BloomBlock"
        elif self == ModelType.Llama:
            return "LlamaDecoderLayer"
        elif self == ModelType.Baichuan:
            return "DecoderLayer"

    @classmethod
    def get_modelType(cls, model_config):
        if "bloom" in model_config['model_path']:
            return ModelType.Bloom
        elif "llama" in model_config['model_path']:
            return ModelType.Llama
        elif "baichuan" in model_config['model_path']:
            return ModelType.Baichuan
        else:
            return None


class DataType(Enum):
    Local = 1
    Huggingface = 2
    Other = 3

    @classmethod
    def get_available_types(cls):
        return DataType.__members__

    @classmethod
    def get_dataType(cls, data_config):
        if os.path.exists(data_config['data_path']):
            return DataType.Local
        if not is_valid_url(data_config['data_path']):
            return None
        if "huggingface" in data_config['data_path']:
            return DataType.Huggingface
        return DataType.Other

    @classmethod
    def download_hf(cls, data_path):
        # TODO: @liang
        return True

    @classmethod
    def download_url(cls, data_path):
        # TODO: @liang
        return True

    def download_data_if_needed(self, data_config):
        if self == DataType.Local:
            return True
        elif self == DataType.Huggingface:
            return download_hf(data_config['data_path'])
        elif self == DataType.Other:
            return download_url(data_config['data_path'])

def is_valid_url(url):
    try:
        response = requests.head(url)
        return response.status_code == requests.codes.ok
    except requests.exceptions.RequestException:
        return False

if __name__ == '__main__':
    print(list(ModelType.get_available_types()))