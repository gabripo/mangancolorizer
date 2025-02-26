import os
import torch
import matplotlib.pyplot as plt

import manga_colorization_v2.colorizator

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'gpu'
    else:
        device = 'cpu'
    generator_path = os.path.abspath(os.path.join(
        os.getcwd(),
        'manga_colorization_v2',
        'networks',
        'generator.zip',
    ))
    denoiser_path = os.path.abspath(os.path.join(
        os.getcwd(),
        'manga_colorization_v2',
        'denoising',
        'models',
    ))
    colorizer = manga_colorization_v2.colorizator.MangaColorizator(
        device=device,
        generator_path=generator_path,
        denoiser_path=denoiser_path
    )

    input_dir = os.path.abspath(os.path.join(
        os.getcwd(),
        'input',
        'One Piece 1140 (ENG)',
    ))
    image_extensions = ('.jpg', '.jpeg', '.png')
    images_path = [
        os.path.join(input_dir, img_path)
        for img_path in sorted(os.listdir(input_dir))
        if img_path.endswith(image_extensions)
        ]

    img_path = images_path[0] # make things easy for now
    image_plt = plt.imread(img_path)
    colorizer.set_image(image_plt)
    image_plt_colorized = colorizer.colorize()

    img_name, img_ext = os.path.splitext(os.path.basename(img_path))
    output_dir = os.path.join(os.getcwd(), 'output', os.path.basename(os.path.dirname(img_path)))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_col_path = os.path.abspath(os.path.join(
        output_dir,
        f"{img_name}_col{img_ext}",
    ))
    plt.imsave(img_col_path, image_plt_colorized)
