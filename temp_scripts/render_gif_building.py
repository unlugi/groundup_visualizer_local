import os
import cairosvg
import glob
from PIL import Image

def svg_to_png(svg_path, png_path):
    """
    Convert an SVG image to PNG format.

    Parameters:
    svg_path (str): Path to the input SVG image.
    png_path (str): Path to save the output PNG image.
    """
    with open(svg_path, 'rb') as f:
        svg_content = f.read()
    cairosvg.svg2png(bytestring=svg_content, write_to=png_path)

def create_gif(svg_paths, output_file, duration=500):
    """
    Create a GIF from a set of SVG images.

    Parameters:
    svg_paths (list): List of paths to the input SVG images.
    output_file (str): Path to save the output GIF.
    duration (int): Duration (in milliseconds) to display each frame.
    """
    png_paths = []
    for svg_path in svg_paths:
        png_path = os.path.splitext(svg_path)[0] + '.png'
        svg_to_png(svg_path, png_path)
        png_paths.append(png_path)

    images = []
    for png_path in png_paths:
        images.append(Image.open(png_path))

    # Save as GIF
    images[0].save(output_file,
                   save_all=True,
                   append_images=images[1:],
                   optimize=False,
                   duration=duration,
                   loop=0)  # loop=0 for infinite loop, loop=1 for no loop, loop=n for n loops

    # Clean up PNG files
    for png_path in png_paths:
        os.remove(png_path)

if __name__ == "__main__":

    folder_path = 'imraj'
    image_paths = sorted(glob.glob(os.path.join(folder_path, 'user_sketch_td_*.svg'))) # List of input image paths
    output_file = "output_td.gif"  # Output GIF file name
    create_gif(image_paths, output_file)