import os
import random
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor

from manga_colorization_v2.denoising.denoiser import FFDNetDenoiser
from manga_colorization_v2.networks.models import Generator
from manga_colorization_v2.utils.utils import resize_pad

IMAGE_SIZE_PADDING = 32

class CGAN():
    def __init__(
            self,
            device: str = 'cpu',
            weights_file_generator: str = None,
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
        self.device = device if device in {'cpu', 'gpu', 'mps'} else None
        self.train_options = {
            'batch_size': 10,
            'epochs': 10,
            'latent_size': latent_dim_size,
        }
        self.denoiser = None
        self.current_pad = None
        self.generator = None
        self.weights_file_gen = None

        self.set_denoiser()
        self.set_colorizer(weights_file_generator)

    def set_denoiser(self, weights_dir: str = 'manga_colorization_v2/denoising/models'):
        # TODO move weights in main repo
        weights_dir_abs = os.path.abspath(weights_dir)
        if not os.path.exists(weights_dir_abs):
            print(f"Impossible to load the denoiser: invalid weights folder {weights_dir_abs}")
            return
        if self.device is None:
            print("Invalid device specified! Impossible to load the denoiser!")
            return
        self.denoiser = FFDNetDenoiser(self.device, _weights_dir=weights_dir_abs)

    def set_colorizer(self, weights_file: str = None):
        if self.device is None:
            print("Invalid device specified! Impossible to load the generator!")
            return
        if self.generator is None:
            self.generator = Generator().to(self.device)

        if weights_file is not None:
            weights_file_abs = os.path.abspath(weights_file)
            if not os.path.exists(weights_file_abs):
                print(f"Impossible to load weights for the generator in generator: invalid weights file {weights_file_abs}")
                return
            self.weights_file_gen = weights_file_abs
            self.generator.load_state_dict(torch.load(self.weights_file_gen, map_location=self.device))

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

    def infere(self, input: str = None, output_dir: str = None):
        # TODO - implement, both for single image and for image folder
        if input is None:
            print("Invalid input for the inference!")
            return
        input_abs_path = os.path.abspath(input)
        if not os.path.exists(input_abs_path):
            print(f"Invalid input specified for inference: {input_abs_path}")
            return
        
        self.set_colorizer('manga_colorization_v2/networks/generator.zip') # load weights of the generator
        input_torch = self._condition_image_input(input_abs_path)
        with torch.no_grad():
            fake_color, _ = self.generator(input_torch)
            fake_color = fake_color.detach()
        # CGAN.plot_pytorch_tensor_image(fake_color)
        output = self._condition_image_output(fake_color)
        print(f"Inference with model {self.__class__.__name__} successful!")

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            input_name, input_ext = os.path.splitext(os.path.basename(input))
            output_path = os.path.join(
                os.path.abspath(output_dir),
                input_name + "_col" + input_ext,
                )
            plt.imsave(output_path, output)
        return output

    def _condition_image_input(self, image_path: str, size = 576, apply_denoise = True, denoise_sigma = 25):
        image = plt.imread(image_path)
        if image is None:
            print(f"Impossible to read the image {image_path}")
            return
        if size % IMAGE_SIZE_PADDING != 0:
            print(f"Specified size to condition image {size} is not divisible by {IMAGE_SIZE_PADDING}: impossible to process the image with CNNs!")
            return

        if apply_denoise:
            if self.denoiser is None:
                print(f"Invalid denoiser! The image {image_path} will not be processed!")
            else:
                image = self.denoiser.get_denoised_image(image, sigma=denoise_sigma)

        image, self.current_pad = resize_pad(image, size)
        
        tensor_transformer = ToTensor()
        image_torch = tensor_transformer(image).unsqueeze(0).to(self.device)

        height, width = image_torch.shape[2], image_torch.shape[3]
        blank_torch = torch.zeros(1, 4, height, width).float().to(self.device)

        return torch.cat([image_torch, blank_torch], 1)
    
    def _condition_image_output(self, image):
        result = image[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5 # permute() for matplotlib convention

        if self.current_pad[0] != 0:
            result = result[:-self.current_pad[0]]
        if self.current_pad[1] != 0:
            result = result[:, :-self.current_pad[1]]

        return result.numpy()

    def is_trained(self):
        return self._is_trained
    
    @staticmethod
    def plot_pytorch_tensor_image(tensor_image: torch.Tensor):
        image = tensor_image.squeeze(0) # remove first dimension
        channels = image.size()[0]
        if channels == 1: # black-white image
            plt.imshow(F.to_pil_image(image))
        elif channels == 3:
            plt.imshow(F.to_pil_image(image)) # pytorch to matplotlib conversion
        else:
            print("Invalid number of channels for a tensor image!")