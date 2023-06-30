from llmfactory.factories import Factory

factory = Factory()
factory.show_available_data()
# [Local]: music, computer, medical
factory.show_available_model()
# [Bloom]: bloom-560m, bloomz-560m, bloom-1b1, bloomz-1b1, bloomz-7b1-mt
# [Llama]: llama-7b-hf, llama-13b-hf
model_config = factory.create_backbone("bloom-560m")
data_config = factory.prepare_data_for_training(20, {"music": 0.4, "computer": 0.6})
model_config = factory.train_model(model_config, data_config, "test")
factory.deploy_model_cli(model_config)
# factory.deploy_model_gradio(model_config)
