class CGAN():
    def __init__(
            self,
            latent_dim_size: int = 10
            ):
        self._is_trained = False

    def load_data_paths(self, data: list[str]):
        # TODO - implement
        print(f"Data paths loaded in {self.__class__.__name__} model!")
        pass

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