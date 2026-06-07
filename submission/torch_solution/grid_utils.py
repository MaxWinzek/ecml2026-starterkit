MOVEMENT_ARRAY = ((-1, 0), (0, 1), (1, 0), (0, -1))


def get_new_position(position, movement):
    return (position[0] + MOVEMENT_ARRAY[movement][0], position[1] + MOVEMENT_ARRAY[movement][1])
