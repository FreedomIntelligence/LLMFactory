import json, random
from .component import DataType

class DataFactory():
    def __init__(self, data_config):
        self.available_data = dict()
        self.data_config = data_config
        self.setup_config(data_config)
    
    def setup_config(self, data_config):
        for data_name, config in data_config.items():
            data_type = DataType.get_dataType(config)
            if data_type is None:
                continue
            config["data_type"] = data_type
            self.available_data[data_name] = config

    def get_all_available(self):
        data_dict = dict()
        for data_name, config in self.available_data.items():
            data_type = config['data_type']
            if data_type not in data_dict:
                data_dict[data_type] = [data_name]
            else:
                data_dict[data_type].append(data_name)
        
        data_descriptions = []
        for data_type, data_names in data_dict.items():
            data_descriptions.append("[{}]: {}".format(data_type.name, ", ".join(data_names)))

        return "\n".join(data_descriptions)

    def prepare_data_for_training(self, num_data: int, data_ratios: dict):
        all_data = []
        for data_name, data_config in self.available_data.items():
            if data_name in data_ratios:
                data = json.load(open(data_config['data_path']))
                sub_num = int(num_data * data_ratios[data_name])
                sub_data = random.sample(data*(1+sub_num//len(data)), sub_num)
                all_data.extend(sub_data)
        return {"num_data": num_data, "data_ratios": data_ratios, "data": all_data}