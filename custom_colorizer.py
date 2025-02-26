import os
from src.colorizer import MangaColorizer

if __name__ == "__main__":
    manga_colorizer = MangaColorizer(
        model_name='cGAN',
        model_kwargs={
            'latent_dim_size': 100,
        }
    )

    input_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'input')
    manga_colorizer.load_dataset(input_dir)

    manga_colorizer.load_model()
    manga_colorizer.train()
    manga_colorizer.infere("dummy_image_path")
