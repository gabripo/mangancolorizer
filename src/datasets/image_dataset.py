import os
from torch import stack
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(
            self,
            image_paths: list[str],
            transformation: transforms.transforms = None,
            device: str = None,
            ):
        self.images = []
        ImageDataset.supported_image_format = {
            '.jpeg',
            '.jpg',
            '.png',
        }
        self.transformation = transformation
        self.device = device if device is not None else 'cpu'

        self._pick_images(image_paths)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.transformation is not None:
            image = self.transformation(image)
        return image
    
    def _pick_images(self, image_paths: list[str]):
        image_paths_added = set(self.images)
        for path in image_paths:
            _, fext = os.path.splitext(path)
            if fext not in ImageDataset.supported_image_format:
                print(f"File {path} has an unsupported format and will be skipped.")
                continue

            fullpath = os.path.abspath(path)
            if fullpath not in image_paths_added: # prevent double append
                image_paths_added.add(fullpath)
                self.images.append(fullpath)

    def collate_fn(self, images: list[Image]):
        images_tensors = [
            transforms.ToTensor()(image).to(self.device)
            for image in images
            ]
        return stack(images_tensors)