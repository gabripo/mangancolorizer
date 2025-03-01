import os
import random
import torch
import matplotlib.pyplot as plt
import numpy
import zipfile
import json
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from manga_colorization_v2.denoising.denoiser import FFDNetDenoiser
from manga_colorization_v2.networks.models import Generator
from manga_colorization_v2.utils.utils import resize_pad

from src.datasets.image_dataset import ImageDataset

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

        self.print_device()
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
            self.discriminator = Discriminator().to(self.device)

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

    def train(self, num_epochs: int = 2, save_weights: bool = True, save_losses: bool = True):
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
        
        losses = {
            'discriminator': {'epoch': [], 'real': [], 'fake': [], 'total': []},
            'generator': {'epoch': [], 'total': []}
            }
        for epoch in range(num_epochs):
            for idx, imgs in enumerate(dataloader):
                batch_size = imgs.size(0)
                img_height = imgs.size(2)
                img_width = imgs.size(3)
                print(f"epoch {epoch+1} - index {idx} - batch size {batch_size} - (width, height) ({img_width}, {img_height})")
                # Train discriminator
                optimizer_D.zero_grad()
                outputs = self.discriminator(imgs)
                real_labels = torch.ones(batch_size, 1).to(self.device) # real is labeled as 1
                d_loss_real = adversarial_loss(outputs, real_labels)

                randn_images = self.generate_randn_images(img_height, img_width, num_images=batch_size)
                z = self._from_images_to_generator_input(randn_images)
                outputs, _ = self.generator(z)
                # TODO - check why torch.mps.driver_allocated_memory() explodes after this
                fake_imgs = self._from_generator_output_to_images(outputs)
                outputs = self.discriminator(fake_imgs)
                fake_labels = torch.zeros(batch_size, 1).to(self.device) # fake is labeled as 0
                d_loss_fake = adversarial_loss(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()
                losses['discriminator']['epoch'].append(epoch+1)
                losses['discriminator']['real'].append(d_loss_real.cpu().detach().numpy())
                losses['discriminator']['fake'].append(d_loss_fake.cpu().detach().numpy())
                losses['discriminator']['total'].append(losses['discriminator']['real'][-1] + losses['discriminator']['fake'][-1])
                print(f"Loss for discriminator: {d_loss} (real {d_loss} | fake {d_loss_fake})")
                self._empty_device_cache()

                # Train generator
                optimizer_G.zero_grad()
                randn_images = self.generate_randn_images(img_height, img_width, num_images=batch_size)
                z = self._from_images_to_generator_input(randn_images)
                outputs, _ = self.generator(z)
                fake_imgs = self._from_generator_output_to_images(outputs)
                outputs = self.discriminator(fake_imgs)
                real_labels = torch.ones(batch_size, 1).to(self.device) # real is labeled as 1
                g_loss = adversarial_loss(outputs, real_labels)

                g_loss.backward()
                optimizer_G.step()
                losses['generator']['epoch'].append(epoch+1)
                losses['generator']['total'].append(g_loss.cpu().detach().numpy())
                print(f"Loss for generator: {g_loss}")
                self._empty_device_cache()

            print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

            if save_losses:
                losses_json_path = os.path.join(os.getcwd(), 'losses.json')
                with open(losses_json_path, 'w') as json_file:
                    json.dump(losses, json_file, default=CGAN.numpy_serializer, indent=4)

            if save_weights:
                self.save_weights(epoch=epoch)
                
        print(f"Model {self.__class__.__name__} trained!")
        self._is_trained = True
        pass

    @staticmethod
    def numpy_serializer(obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.generic):
            return obj.item()
        raise TypeError(f"Type {type(obj)} not serializable")

    def save_weights(self, epoch: int = None):
        save_folder = os.path.join(os.getcwd(), 'weights')
        os.makedirs(save_folder, exist_ok=True)

        generator_weights_path = os.path.join(save_folder, 'fine_tuned_generator_weights.pth')
        torch.save(self.generator.state_dict(), generator_weights_path)
        discriminator_weights_path = os.path.join(save_folder, 'fine_tuned_discriminator_weights.pth')
        torch.save(self.discriminator.state_dict(), discriminator_weights_path)

        if epoch is None:
            zipfile_name = os.path.join(save_folder, f'fine_tuned_model_weights_epoch.zip')
        else:    
            zipfile_name = os.path.join(save_folder, f'fine_tuned_model_weights_epoch_{epoch+1}.zip')
        with zipfile.ZipFile(zipfile_name, 'w') as zipf:
            zipf.write(generator_weights_path, os.path.basename(generator_weights_path))
            zipf.write(discriminator_weights_path, os.path.basename(discriminator_weights_path))

        if os.path.exists(zipfile_name):
            os.remove(generator_weights_path)
            os.remove(discriminator_weights_path)

    def _from_generator_output_to_images(self, generator_outputs: torch.Tensor, device: str = None):
        if device is None:
            device = self.device
        images = [self._condition_image_output(output.unsqueeze(0)) for output in generator_outputs]
        images_torch = torch.cat([self._image_to_torch(img) for img in images]).to(device)
        return images_torch


    def _from_images_to_generator_input(self, images: list[numpy.ndarray], device: str = None):
        if device is None:
            device = self.device
        z = torch.cat([
                self._condition_image_input(img)
                for img in images
                ]).to(self.device)
        return z

    @staticmethod
    def generate_randn_images(img_height: int, img_width: int, num_channels: int = 3, num_images: int = 1):
        rand_images = [
            numpy.random.randn(img_height, img_width, num_channels).astype(numpy.float32)
            for _ in range(num_images)
            ]
        return rand_images
    
    def _empty_device_cache(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache

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
    
    def print_device(self) -> bool:
        if self.device:
            print(f"Device for {self.__class__.__name__} is {self.device}")
        else:
            print(f"No device available for {self.__class__.__name__} !")
    
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



'''https://github.com/orashi/AlacGAN/blob/master/models/standard.py'''
class ResNeXtBottleneck(torch.nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = torch.nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = torch.nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = torch.nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = torch.nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     torch.nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = torch.nn.functional.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = torch.nn.functional.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        x = self.shortcut.forward(x)
        return x + bottleneck

class NetD(torch.nn.Module):
    def __init__(self, ndf=64):
        super(NetD, self).__init__()

        self.feed = torch.nn.Sequential(torch.nn.Conv2d(3, ndf, kernel_size=7, stride=1, padding=3, bias=False),  # 512
                                  torch.nn.LeakyReLU(0.2, True),
                                  torch.nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 256
                                  torch.nn.LeakyReLU(0.2, True),

                                  ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1, stride=2),  # 128
                                  torch.nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False),
                                  torch.nn.LeakyReLU(0.2, True),

                                  ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),  # 64
                                  torch.nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=False),
                                  torch.nn.LeakyReLU(0.2, True),

                                  ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=2)  # 32
                                  )

        self.feed2 = torch.nn.Sequential(torch.nn.Conv2d(ndf * 12, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),  # 32
                                   torch.nn.LeakyReLU(0.2, True),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 16
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 8
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 4
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   torch.nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=0, bias=False),  # 1
                                   torch.nn.LeakyReLU(0.2, True)
                                   )

        self.out = torch.nn.Linear(512, 1)

    def forward(self, color, sketch_feat):
        x = self.feed(color)

        x = self.feed2(torch.cat([x, sketch_feat], 1))

        out = self.out(x.view(color.size(0), -1))
        return out

class NetI(torch.nn.Module):
    def __init__(self):
        super(NetI, self).__init__()
        i2v_model = torch.nn.Sequential(  # Sequential,
            torch.nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            torch.nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            torch.nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            torch.nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            torch.nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            torch.nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(1024, 1539, (3, 3), (1, 1), (1, 1)),
            torch.nn.AvgPool2d((7, 7), (1, 1), (0, 0), ceil_mode=True),  # AvgPool2d,
        )
        # i2v_model.load_state_dict(torch.load(I2V_PATH))
        i2v_model = torch.nn.Sequential(
            *list(i2v_model.children())[:15]
        )
        self.model = i2v_model
        self.register_buffer('mean', torch.FloatTensor([164.76139251, 167.47864617, 181.13838569]).view(1, 3, 1, 1))

    def forward(self, images):
        images = F.avg_pool2d(images, 2, 2)
        images = images.mul(0.5).add(0.5).mul(255)
        return self.model(images.expand(-1, 3, 256, 256) - self.mean)
    
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2)
        )
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to handle varying input sizes
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x