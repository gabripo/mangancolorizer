import os
from src.colorizer import MangaColorizer

if __name__ == "__main__":
    manga_colorizer = MangaColorizer(
        model_name='cGAN',
        model_kwargs={
            'latent_dim_size': 100,
        }
    )

    input_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dataset')
    manga_colorizer.load_dataset(input_dir)

    manga_colorizer.load_model()
    manga_colorizer.model.train_options['batch_size'] = 20
    manga_colorizer.model.train_options['epochs'] = 50
    manga_colorizer.train(save_weights=True)
    # manga_colorizer.train() # to test double-append prevention in model
    output = manga_colorizer.infere("input/One Piece 1140 (ENG)/00000.jpg", "output")