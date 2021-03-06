import os
import sys
import numpy as np

# dimensions of result image
W = 1200
H = 900
CHANNEL_NUM = 3  # we work with rgb images
MAX_VALUE = 255  # max pixel value, required by ppm header


def read_image(path):
    # second line of header contains image dimensions
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # высота и ширина фрагмента изображения
    # print('This is W and H ', w, h)
    # skip 3 lines reserved for header and read image
    image = np.loadtxt(path, skiprows=3, dtype=np.uint8).reshape((h, w, CHANNEL_NUM))
    # преобразование ppm в массив с ориентиром на высоту и ширину конкретного фрагмента изображения
    # print('And this is image ', image)
    return image


def write_image(path, img):
    h, w = img.shape[:2]
    # ppm format requires header in special format
    header = f'P3\n{w} {h}\n{MAX_VALUE}\n'
    with open(path, 'w') as f:
        f.write(header)
        for r, g, b in img.reshape((-1, CHANNEL_NUM)):
            f.write(f'{r} {g} {b} ')


def solve_puzzle(tiles_folder):
    # create placeholder for result image
    # read all tiles in list
    tiles = [read_image(os.path.join(tiles_folder, t)) for t in sorted(os.listdir(tiles_folder))]
    print(tiles[0][0][0], tiles[0][0][1], tiles[0][0][-2],  tiles[0][0][-1])#, tiles[0][0][0],  tiles[0][0][0],)


    result_img = np.zeros((H, W, CHANNEL_NUM), dtype=np.uint8)
    # scan dimensions of all tiles and find minimal height and width
    dims = np.array([t.shape[:2] for t in tiles])
    #print(dims)
    h, w = np.min(dims, axis=0)
    #print('Print H, W ', h, w)
    # compute grid that will cover image
    # spacing between grid rows = min h
    # spacing between grid columns = min w
    x_nodes = np.arange(0, W, w)
    y_nodes = np.arange(0, H, h)
    xx, yy = np.meshgrid(x_nodes, y_nodes)
    # meshgrid - создание списка  массивов
    nodes = np.vstack((xx.flatten(), yy.flatten())).T
    # vstack - объединение массивов по вертикали
    # flatten - сжатие массива до одного измерения
    # метод ndarray.T транспортирует, меняет оси массива в обратном порядке
    # fill grid with tiles
    for (x, y), tile in zip(nodes, tiles):
        result_img[y: y + h, x: x + w] = tile[:h, :w]
    # добавление изображения в пустой массив по конкретным координатам
    # result_img - zeros массив состоящий из нулей
    # tile - один фрагмент из списка
    # output_path = "image.ppm"
    # write_image(output_path, result_img)


if __name__ == "__main__":
    directory = 'C:/Users/Евгений/PycharmProjects/pythonProject1/computer vision/images/0000_0005_0003/tiles'
    solve_puzzle(directory)
