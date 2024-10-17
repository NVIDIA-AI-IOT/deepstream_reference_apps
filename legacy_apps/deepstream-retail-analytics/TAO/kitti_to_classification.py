import os
from glob import glob
import cv2
from tqdm import tqdm


def read_label_file(filepath):
    labels = []
    coordinates = []
    
    # Read the file
    with open(filepath, "r") as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.split()
        lab, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = line
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        labels.append(lab)
        coordinates.append((x1, y1, x2, y2))
    
    return labels, coordinates


def crop_image(img, xmin, ymin, xmax, ymax):
    return img[ymin:ymax, xmin:xmax :]


def draw_bbox(img, xmin, ymin, xmax, ymax):
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
    return img


if __name__ == "__main__":
    images = glob("default/image_2/*.PNG")

    for image in tqdm(images):
        label_file, _ = os.path.splitext(os.path.basename(image))
        labels, coordinates = read_label_file(f"default/label_2/{label_file}.txt")
        img = cv2.imread(image)
        for i, label in enumerate(labels):
            img_cropped = crop_image(img, coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3])
            # img_annotated = draw_bbox(img, coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3])
            cv2.imwrite(f"classification_data/{label}/{label_file}_{i}.png", img_cropped)
            # cv2.imwrite(f"test_{i}.png", img_annotated)