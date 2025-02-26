import os
import torch

from models.cgan import CGAN

class MangaColorizer():
    def __init__(
            self,
            model_name: str = 'cGAN',
            model_kwargs: dict = None,
            ):
        self.device = None
        self.model_name = None
        self.model = None
        self.model_kwargs = model_kwargs
        self.input_dir = None
        self.input_dataset = None

        MangaColorizer.supported_models = {
            'cGAN': CGAN,
            }
        MangaColorizer.supported_image_format = {
            '.jpeg',
            '.jpg',
            '.png',
        }
        
        self._determine_device()
        self.set_model_name(model_name)

    def load_model(self):
        if self.model_name is None:
            print("No model specified!")
            return
        if not MangaColorizer._check_supported_model(self.model_name):
            print(f"Model {self.model_name} not supported!")
            return
        if self.model_kwargs is not None:
            self.model = MangaColorizer.supported_models[self.model_name](**self.model_kwargs)
        else:
            self.model = MangaColorizer.supported_models[self.model_name]

    def load_dataset(self, input_dir: str = None):
        if input_dir is None:
            print("Invalid input folder specified!")
            return
        if not os.path.exists(input_dir):
            print(f"Specified input folder {input_dir} does not exist!")
            return
        self.input_dataset = MangaColorizer.traverse_dir(
            input_dir,
            MangaColorizer.supported_image_format,
            )
        self.input_dir = input_dir

    def train(self):
        if self.input_dataset is None or self.input_dir is None:
            print("Empty input dataset!")
            return
        self.model.load_data_paths(self.input_dataset)
        self.model.train()

    def infere(self, input):
        if self.model is None:
            print("No available model!")
            return
        if not self.model.is_trained():
            print("Model has not been trained!")
            return
        return self.model.infere(input=input)

    def set_model_name(self, model_name: str = None):
        if model_name is None:
            print("No model specified!")
            return
        if not MangaColorizer._check_supported_model(model_name):
            print(f"Model {model_name} not supported!")
            return
        self.model_name = model_name

    def _determine_device(self) -> None:
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'gpu'
        else:
            self.device = 'cpu'

    @staticmethod
    def traverse_dir(dir: str, extensions: set = None):
        content = []
        for dirpath, dirnames, filenames in os.walk(dir):
            if len(filenames) == 0:
                continue
            if extensions is not None:
                content.extend([
                    os.path.join(os.path.abspath(dirpath), fname)
                    for fname in filenames
                    if os.path.splitext(fname)[1] in extensions
                ])
            else:
                content.extend(filenames)
        return content

    @staticmethod
    def _check_supported_model(model_name: str):
        return model_name in MangaColorizer.supported_models.keys()