import os
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths: list[str]):
        self.images = []
        ImageDataset.supported_image_format = {
            '.jpeg',
            '.jpg',
            '.png',
        }

        self._pick_images(image_paths)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index])
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