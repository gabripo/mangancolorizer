import os
import random
import torch
import matplotlib.pyplot as plt
import numpy
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from manga_colorization_v2.denoising.denoiser import FFDNetDenoiser
from manga_colorization_v2.networks.models import Generator
from manga_colorization_v2.utils.utils import resize_pad

from datasets.image_dataset import ImageDataset

IMAGE_SIZE_PADDING = 32

class CGAN():
    def __init__(
            self,
            device: str = 'cpu',
            weights_file_generator: str = None,
            latent_dim_size: int = 10
            ):
        self._is_trained = False
        CGAN.supported_data_types = {'image'}
        self.dataset = None
        self.device = device if device in {'cpu', 'gpu', 'mps'} else None
        self.train_options = {
            'batch_size': 1,
            'epochs': 10,
            'latent_size': latent_dim_size,
        }
        self.denoiser = None
        self.current_pad = None
        self.generator = None
        self.weights_file_gen = None
        self.discriminator = None

        self.set_denoiser()
        self.set_generator(weights_file_generator)
        self.set_discriminator()

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

    def set_generator(self, weights_file: str = None):
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

    def set_discriminator(self):
        if self.device is None:
            print("Invalid device specified! Impossible to load the generator!")
            return
        if self.discriminator is None:
            # TODO same architecture as NetD in alacGAN
            self.discriminator = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (887, 576) -> (443, 288)
                torch.nn.LeakyReLU(0.2),
                torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (443, 288) -> (221, 144)
                torch.nn.BatchNorm2d(128),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (221, 144) -> (110, 72)
                torch.nn.BatchNorm2d(256),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (110, 72) -> (55, 36)
                torch.nn.BatchNorm2d(512),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # (55, 36) -> (27, 18)
                torch.nn.BatchNorm2d(1024),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),  # (27, 18) -> (24, 15)
                torch.nn.Sigmoid()
            ).to(self.device)

    def load_data_paths(self, data_paths: list[str], data_type: str):
        if data_type not in CGAN.supported_data_types:
            print(f"Invalid data type specified {data_type} for the {self.__class__.__name__} model!")
            return
        if self.device is None:
            print("Invalid device where to load the dataset onto!")
            return
        if data_type == 'image':
            self.dataset = ImageDataset(data_paths, device=self.device)
        print(f"Data paths loaded in {self.__class__.__name__} model!")

    def train(self, num_epochs: int = 2):
        if self.dataset is None:
            print("Empty dataset: impossible to train!")
            return
        if not isinstance(self.dataset, torch.utils.data.Dataset):
            print("Invalid torch dataset: impossible to train!")
        dataloader_kwargs = {
            'collate_fn': self.dataset.collate_fn,
        }
        if 'batch_size' in self.train_options.keys():
            dataloader_kwargs['batch_size'] = self.train_options['batch_size']
        dataloader = DataLoader(dataset=self.dataset, **dataloader_kwargs)
        
        adversarial_loss = torch.nn.BCELoss()
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999),
            )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999),
            )
        
        for epoch in range(num_epochs):
            for idx, imgs in enumerate(dataloader):
                batch_size = imgs.size(0)
                img_height = imgs.size(2)
                img_width = imgs.size(3)
                print(f"epoch {epoch} - index {idx} - batch size {batch_size} - (width, height) ({img_width}, {img_height})")
                # Train discriminator
                optimizer_D.zero_grad()
                outputs = self.discriminator(imgs)
                real_labels = torch.ones(batch_size, 1, outputs.size(2), outputs.size(3)).to(self.device) # real is labeled as 1
                d_loss_real = adversarial_loss(outputs, real_labels)

                # TODO currently working for batch_size = 1 only, to be extended
                z = torch.cat([
                    self._condition_image_input(self.generate_randn_image(img_height, img_width))
                    for _ in range(batch_size)
                    ]).to(self.device)
                outputs, _ = self.generator(z)
                # TODO - check why torch.mps.driver_allocated_memory() explodes after this
                outputs.detach()
                fake_imgs = self._condition_image_output(outputs)
                outputs = self.discriminator(self._image_to_torch(fake_imgs))
                fake_labels = torch.zeros(batch_size, 1, outputs.size(2), outputs.size(3)).to(self.device) # fake is labeled as 0
                d_loss_fake = adversarial_loss(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

                # Train generator
                optimizer_G.zero_grad()
                # TODO currently working for batch_size = 1 only, to be extended
                z = torch.cat([
                    self._condition_image_input(self.generate_randn_image(img_height, img_width))
                    for _ in range(batch_size)
                    ]).to(self.device)
                outputs, _ = self.generator(z)
                outputs.detach()
                fake_imgs = self._condition_image_output(outputs)
                outputs = self.discriminator(self._image_to_torch(fake_imgs))

                real_labels = torch.ones(batch_size, 1, outputs.size(2), outputs.size(3)).to(self.device) # real is labeled as 1
                g_loss = adversarial_loss(outputs, real_labels)
                g_loss.backward()
                optimizer_G.step()

                torch.mps.empty_cache() # freeing up device memory
            
            print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
                

        print(f"Model {self.__class__.__name__} trained!")
        self._is_trained = True
        pass

    @staticmethod
    def generate_randn_image(img_height: int, img_width: int, num_channels: int = 3):
        rand_img = numpy.random.randn(img_height, img_width, num_channels).astype(numpy.float32)
        return rand_img

    def infere(self, input: str = None, output_dir: str = None):
        # TODO - implement, both for single image and for image folder
        if input is None:
            print("Invalid input for the inference!")
            return
        input_abs_path = os.path.abspath(input)
        if not os.path.exists(input_abs_path):
            print(f"Invalid input specified for inference: {input_abs_path}")
            return
        
        self.set_generator('manga_colorization_v2/networks/generator.zip') # load weights of the generator
        input_torch = self._condition_image_input_path(input_abs_path)
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

    def _condition_image_input_path(self, image_path: str, size = 576, apply_denoise = True, denoise_sigma = 25):
        image = plt.imread(image_path)
        if image is None:
            print(f"Impossible to read the image {image_path}")
            return
        return self._condition_image_input(image, size, apply_denoise, denoise_sigma)
    
    def _condition_image_input(self, image: numpy.ndarray, size = 576, apply_denoise = True, denoise_sigma = 25):
        if size % IMAGE_SIZE_PADDING != 0:
            print(f"Specified size to condition image {size} is not divisible by {IMAGE_SIZE_PADDING}: impossible to process the image with CNNs!")
            return 

        if apply_denoise:
            if self.denoiser is None:
                print(f"Invalid denoiser! The image will not be processed!")
            else:
                image = self.denoiser.get_denoised_image(image, sigma=denoise_sigma)

        image, self.current_pad = resize_pad(image, size)
        
        image_torch = self._image_to_torch(image)

        height, width = image_torch.shape[2], image_torch.shape[3]
        blank_torch = torch.zeros(1, 4, height, width).float().to(self.device)

        return torch.cat([image_torch, blank_torch], 1)
    
    def _image_to_torch(self, image_plt: numpy.ndarray, unsqueeze = True):
        tensor_transformer = ToTensor()
        if unsqueeze:
            image_torch = tensor_transformer(image_plt).unsqueeze(0).to(self.device)
        else:
            image_torch = tensor_transformer(image_plt).to(self.device)
        return image_torch
    
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