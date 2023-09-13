from config import *
from PIL import Image
import random
import os
import time
import json
from tqdm import tqdm


SAVE_ROOT = os.path.abspath(SAVE_ROOT)
WORD_LIST_PATH = os.path.abspath(WORD_LIST_PATH)


def generate_sentence(words_list, word_count):
    sentence_words = random.sample(words_list, word_count)
    sentence = ' '.join(sentence_words)
    return sentence


def generate_image(width: int, height: int, mode: str, format: str):
    image = Image.new(mode, (width, height))
    pixels = image.load()
    
    if mode == "RGB":
        for x in range(width):
            for y in range(height):
                pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    elif mode == "L":
        for x in range(width):
            for y in range(height):
                pixels[x, y] = random.randint(0, 255)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return image


def generate(words_list, len: int):
    width, height = IMAGE_SIZE
    res = []
    for _ in tqdm(range(len)):
        # text
        text_length = random.randint(MIN_LENGTH, MAX_LENGTH)
        text = generate_sentence(words_list, text_length)
        # image
        image = generate_image(width, height, IMAGE_MODE, IMAGE_TYPE)
        
        res.append((text, image))
    return res


if __name__ == "__main__":
    # get word list
    with open(WORD_LIST_PATH, 'r') as f:
        words = f.readlines()
    words = [word.strip() for word in words if word.isascii()]
    
    
    # image folder
    IMAGE_ROOT = os.path.join(SAVE_ROOT, 'images')
    if not os.path.exists(IMAGE_ROOT):
        os.makedirs(IMAGE_ROOT)

    pairs = generate(words, LENGTH)
    res = []
    for text, image in tqdm(pairs):
        # make pair data
        image_filepath = os.path.join(IMAGE_ROOT, str(int(time.time() * 1000)) + f'.{IMAGE_TYPE}')
        line = json.dumps({'text': text, 'vision': image_filepath}) + '\n'
        res.append(line)
        
        # save image
        image.save(image_filepath)
        time.sleep(.001)
    
    dataset_filepath = os.path.join(SAVE_ROOT, 'dataset.jsonl')
    with open(dataset_filepath, 'w') as f:
        f.writelines(res)
