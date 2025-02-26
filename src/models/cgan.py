import os

class CGAN():
    def __init__(
            self,
            latent_dim_size: int = 10
            ):
        self._is_trained = False
        CGAN.supported_image_format = {
            '.jpeg',
            '.jpg',
            '.png',
        }
        CGAN.supported_data_types = {
            'image': CGAN.supported_image_format,
            }
        self.data_paths = []

    def load_data_paths(self, data_paths: list[str], data_type: str):
        if data_type not in CGAN.supported_data_types.keys():
            print(f"Invalid data type specified {data_type} for the {self.__class__.__name__} model!")
            return
        
        data_paths_added = set(self.data_paths)
        for path in data_paths:
            _, fext = os.path.splitext(path)
            if fext not in CGAN.supported_data_types[data_type]:
                print(f"File {path} has an unsupported format and will be skipped.")
                continue

            fullpath = os.path.abspath(path)
            if fullpath not in data_paths_added: # prevent double append
                data_paths_added.add(fullpath)
                self.data_paths.append(fullpath)
        print(f"Data paths loaded in {self.__class__.__name__} model!")

    def train(self):
        # TODO - implement
        print(f"Model {self.__class__.__name__} trained!")
        self._is_trained = True
        pass

    def infere(self, input):
        # TODO - implement, both for single image and for image folder
        print(f"Inference with model {self.__class__.__name__} successful!")
        pass

    def is_trained(self):
        return self._is_trained