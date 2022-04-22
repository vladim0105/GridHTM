from PIL import Image, ImageDraw, ImageFont
import numpy as np


def value_to_hsv(val, max_val=1):
    col = np.zeros(shape=(3,)).astype(np.uint8)
    col[0] = int(60 * (max_val - val))
    col[1] = 255
    col[2] = 255
    return col


def text_phantom(text, size):
    # Availability is platform dependent
    pil_font = ImageFont.load_default()

    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [size * 5, size], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size * 5 - text_width) // 2,
              (size - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    out = np.asarray(canvas)
    return out


def moving_average(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')
def trailing_average(x, N):
    return np.convolve(x, np.ones(N)/N, mode='full')

def random_bit_array(shape, num_ones):
    arr = np.zeros(shape=shape)
    arr[:num_ones] = 1
    np.random.shuffle(arr)
    return arr
