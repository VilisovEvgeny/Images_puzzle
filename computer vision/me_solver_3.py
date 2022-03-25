import os
import sys
import numpy as np
import cv2

# dimensions of result image
W = 1200
H = 900
CHANNEL_NUM = 3  # we work with rgb images
MAX_VALUE = 255  # max pixel value, required by ppm header


def read_image(path):
    # second line of header contains image dimensions
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # высота и ширина фрагмента изображения
    # skip 3 lines reserved for header and read image
    image = np.loadtxt(path, skiprows=3, dtype=np.uint8).reshape((h, w, CHANNEL_NUM))
    # преобразование ppm в массив с ориентиром на высоту и ширину конкретного фрагмента изображения
    return image


def write_image(path, img):
    print(img)
    h, w = img.shape[:2]
    # ppm format requires header in special format
    header = f'P3\n{w} {h}\n{MAX_VALUE}\n'
    with open(path, 'w') as f:
        f.write(header)
        for r, g, b in img.reshape((-1, CHANNEL_NUM)):
            f.write(f'{r} {g} {b} ')


# solve_puzzle осуществляет сбор цельного изображения из уже отсортированных и упорядоченных фрагментов
# на данный момент эта функция не подключена, поскольку фрагменты упорядочиваются не корректно
def solve_puzzle(tiles_folder):
    # create placeholder for result image
    # read all tiles in list
    tiles = [read_image(os.path.join(tiles_folder, t)) for t in sorted(os.listdir(tiles_folder))]
    result_img = np.zeros((H, W, CHANNEL_NUM), dtype=np.uint8)
    # scan dimensions of all tiles and find minimal height and width
    dims = np.array([t.shape[:2] for t in tiles])
    h, w = np.min(dims, axis=0)
    print('Print H, W ', h, w)
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
    output_path = "image.ppm"
    write_image(output_path, result_img)


# путь к файлу с фрагментами изображения
directory = 'C:/Users/Евгений/PycharmProjects/pythonProject1/computer vision/data/0000_0000_0000/tiles'

# список, содержащий фрагменты в виде массивов RGB
tiles_array = np.array([read_image(os.path.join(directory, t)) for t in sorted(os.listdir(directory))])

LENGTH_T = len(tiles_array)
W_TILES_NUMBER = int(W/((W * H / LENGTH_T) ** 0.5))
H_TILES_NUMBER = int(H/((W * H / LENGTH_T) ** 0.5))


# добавление крайних пикселей со стороны фрагмента
def side_gener(tile):
    # side = np.array((tile[0][0],
    #                  tile[0][1],
    #                  tile[0][2],
    #                  #tile[0][(len(tile[0]) % 2) - 1],
    #                  #tile[0][len(tile[0]) % 2],
    #                  tile[0][-3],
    #                  tile[0][-2],
    #                  tile[0][-1]))
    side = tile[0]
    return side


# создание словаря фрагмента, содержашего данные о каждой стороне фрагмента,
# о его повороте при подстановке и о наиболее подходящем "соседе" с каждой стороны
def get_tiles_data_dict(tiles):
    tiles_dict_list = []

    for i, tile in zip(range(len(tiles)), tiles):
        i = {
            "rot": 0,
            0: side_gener(tile),
            1: side_gener(np.rot90(tile, k=1)),
            2: side_gener(np.rot90(tile, k=2)),
            3: side_gener(np.rot90(tile, k=3)),
            "mates": {
                0: [0, 0, 100000000000],
                1: [0, 0, 100000000000],
                2: [0, 0, 100000000000],
                3: [0, 0, 100000000000],
            }
        }
        tiles_dict_list.append(i)

    return tiles_dict_list


# нахождение разницы между сторонами первого и второго фрагментов

def find_difference(side1, side2):
    difference = 0
    side1 = list(side1.flatten())
    side2 = side2[::-1]
    side2 = list(side2.flatten())

    for el1, el2 in zip(side1, side2):
        difference += abs(int(el1) - int(el2))
    return difference


# перебор фрагментов и поиск наиболее подходящего "соседа" с каждой стороны
# подходящим считается та сторона, того фрагмента, разница между которой наименьшая из всех вариантов
def compatibility_gener(tiles):
    for i in range(len(tiles)):
        print("\n")
        for j in range(len(tiles)):
            if not i == j:
                for k in range(4):
                    if find_difference(tiles[i][0], tiles[j][k]) < tiles[i]["mates"][0][2]:
                        tiles[i]["mates"][0] = [j, k, find_difference(tiles[i][0], tiles[j][k])]
                    if find_difference(tiles[i][1], tiles[j][k]) < tiles[i]["mates"][1][2]:
                        tiles[i]["mates"][1] = [j, k, find_difference(tiles[i][1], tiles[j][k])]
                    if find_difference(tiles[i][2], tiles[j][k]) < tiles[i]["mates"][2][2]:
                        tiles[i]["mates"][2] = [j, k, find_difference(tiles[i][2], tiles[j][k])]
                    if find_difference(tiles[i][3], tiles[j][k]) < tiles[i]["mates"][3][2]:
                        tiles[i]["mates"][3] = [j, k, find_difference(tiles[i][3], tiles[j][k])]
            print(i, tiles[i]["mates"])


# получение координат соседних фрагментов относительно основного
def get_positions(main_cord, rot=0):
    up = [main_cord[0] - 1, main_cord[1]]
    left = [main_cord[0], main_cord[1] - 1]
    down = [main_cord[0] + 1, main_cord[1]]
    right = [main_cord[0], main_cord[1] + 1]
    direction = [up, right, down, left, up, right, down, left, up, right, down, left]
    return direction[4+rot: 8+rot]


def assemble_image(tiles_dict, field, current_position, used_list, tile=0):
    # tile - индекс фрагмента от которого начинается сборка пазла
    if len(used_list) == LENGTH_T - 1:

        field[current_position[0]][current_position[1]] = tile
        return field
    else:
        field[current_position[0]][current_position[1]] = tile
        el1 = tiles_dict[tile]

        used_list.append(tile)
        directions = get_positions(current_position, el1["rot"])
        for i, position in zip(range(0, 4), directions):

            # el2 - фрагмент, прилегающий к основному
            el2_ind = tiles_dict[tile]["mates"][i][0]
            el2_side = tiles_dict[tile]["mates"][i][1]
            el2 = tiles_dict[el2_ind]
            if tiles_dict[tile]["mates"][i][2] == el2["mates"][el2_side][2] and (el2_ind not in used_list)\
                    and field[position[0]][position[1]] == -1:
                rot = (-(el2_side - i)) % 4 + el1["rot"]
                tiles_dict[el2_ind]["rot"] = rot
                field[position[0]][position[1]] = el2_ind
                assemble_image(tiles_dict, field, position, used_list, el2_ind)
    return field


def solvation(tiles):
    tiles_dict = get_tiles_data_dict(tiles)
    compatibility_gener(tiles_dict)
    tiles_field = np.full((W_TILES_NUMBER*2 + 2, W_TILES_NUMBER*2 + 2), -1, dtype=np.int16)
    first_position = [W_TILES_NUMBER + 1, W_TILES_NUMBER + 1]
    used_list = []
    ordered_tiles = assemble_image(tiles_dict, tiles_field, first_position, used_list)
    print(ordered_tiles)
    return tiles_dict


a = solvation(tiles_array)
