import json, os, time, subprocess
from ..constants import TRAIN_SCRIPT
from ..utils import EnumEncoder
# from llmfactory.factories import DataFactory, ModelFactory
from .DataFactory import DataFactory
from .ModelFactory import ModelFactory

class Factory():
    def __init__(self, config_file="factory/resource.json"):
        assert os.path.exists(config_file), print(f"factory configuration file not found: {config_file}")

        config = json.load(open(config_file))
        self.model_factory = ModelFactory(config['model_config'])
        self.data_factory = DataFactory(config['data_config'])
        self.config = config

    def show_available_model(self):
        """
        Show all the available model for model training.
        """
        print(self.model_factory.get_all_available())

    def show_available_data(self):
        """
        Show all the available data for model training.
        """
        print(self.data_factory.get_all_available())

    def create_backbone(self, model: str):
        """
        Create model for model training.

        Args:
            model (str): The name of the selected model.
        
        Returns:
            dict: Information about the created model.
        """
        model_config = self.model_factory.available_models.get(model, None)
        if model_config is None:
            print(f"Incorrect model selection: [{model}]. Please utilize the \"show_available_models\" function to display the models that are currently accessible.")
            return None
        return self.model_factory.create_backbone(model_config)

    def prepare_data_for_training(self, num_data: int=100, data_ratios: dict={}):
        """
        Prepare data for model training.

        Args:
            num_data (int): The total number of training data.
            data_ratios (dict): The ratio of each dataset in the training data.
        
        Returns:
            dict: Information about the prepared data.
        """
        available_data = self.data_factory.available_data
        for key in data_ratios:
            if key not in available_data:
                print(f"Incorrect data selection: [{key}]. Please utilize the \"show_available_data\" function to display the data that are currently accessible.")
                return None
        if len(data_ratios) == 0:
            print(f"Incorrect data selection: [None]. Please utilize the \"show_available_data\" function to display the data that are currently accessible.")
        return self.data_factory.prepare_data_for_training(num_data, data_ratios)

    def train_model(self, model_config: dict, data_config: dict, save_name: str, save_dir: str = "trained_model"):
        """
        Trains a model using prepared data.

        Args:
            model_config (dict): A dictionary containing information about the target model.
            data_config (dict): A dictionary containing information about the prepared data.
            save_name (str): The name of the saved model.
            save_dir (str): The directory where the trained model will be saved. Default is "trained_model".

        Returns:
            dict: A dictionary containing information about the trained model.
        """
        output_dir = os.path.join(save_dir, save_name)
        os.makedirs(output_dir, exist_ok=True)

        # save data for training
        data_path = os.path.join(output_dir, "data.json")
        with open(data_path, 'w') as writer:
            json.dump(data_config['data'], writer, indent=4, ensure_ascii=False)
        del data_config['data']
        data_config['data_path'] = data_path
        print(f"Successfully save data to {data_path}")

        # save train script for training
        train_script = TRAIN_SCRIPT \
            .replace("<<model_path>>", model_config['model_path']) \
            .replace("<<data_path>>", data_path) \
            .replace("<<output_dir>>", output_dir) \
            .replace("<<fsdp_transformer_layer_cls_to_wrap>>", model_config['model_type'].fsdp_transformer_layer_cls_to_wrap)

        script_path = os.path.join(output_dir, "train.sh")
        with open(script_path, 'w') as writer:
            writer.write(train_script)
        print(f"Successfully save train script to {script_path}")

        log_path = os.path.join(output_dir, "logs", "training.log")
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        if self.config['training_config'].get("train_in_bg", False):
            process_id = subprocess.getoutput(f"nohup sh {script_path} > {log_path} 2>&1 & echo $!")
            print(f"Successfully submitted training in background, with process_id: {process_id}")
        else:
            print("Start training... ")
            process_id = None
            subprocess.call(['sh', script_path, '|', 'tee', log_path])
            print("Finished training")

        training_config = {"training_dir": output_dir, "process_id": process_id, "model_config": model_config, "data_config": data_config}
        with open(os.path.join(output_dir, "logs", "training_config.json"), 'w') as writer:
            json.dump(training_config, writer, indent=4, ensure_ascii=False, cls=EnumEncoder)

        new_model_config = {"model_path": output_dir}
        return new_model_config

    def deploy_model_gradio(self, model_config):
        log_dir = os.path.join(model_config['model_path'], "logs")
        os.makedirs(log_dir, exist_ok=True)

        print("Start launching controller...")

        # Launch a controller
        controller_log = os.path.join(log_dir, "deploy_controller.log")
        controller_process_id = subprocess.getoutput(f"nohup python -m llmfactory.deploy.webapp.controller > {controller_log} 2>&1 & echo $!")
        time.sleep(3)
        print(f"Finished launch controller {controller_process_id}, launching model worker...")

        # Launch a model worker
        worker_log = os.path.join(log_dir, "deploy_worker.log")
        worker_process_id = subprocess.getoutput(f"nohup python -m llmfactory.deploy.webapp.model_worker --model-path {model_config['model_path']} > {worker_log} 2>&1 & echo $!")
        time.sleep(3)
        print(f"Finished launch model worker {worker_process_id}, launching web server...")

        # Launch a gradio web server
        server_log = os.path.join(log_dir, "deploy_server.log")
        web_server = ['python', '-m', 'llmfactory.deploy.webapp.gradio_web_server']
        try:
           subprocess.call(web_server)
        except KeyboardInterrupt:
            pass
        finally:
            subprocess.call(['kill', '-9', str(controller_process_id)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.call(['kill', '-9', str(worker_process_id)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Model deployment exited")
        return 

    def deploy_model_cli(self, model_config):
        command = ['python', '-m', 'llmfactory.deploy.cli', '--model-path', model_config['model_path']]
        try:
            subprocess.call(command)
        except KeyboardInterrupt:
            pass
        finally:
            print("Model deployment exited")
        return