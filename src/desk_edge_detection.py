def find_horizontal_edge_y(lines, tolerance=5):
    if lines is None:
        return None

    for line in lines:
        coordinates = line
        while len(coordinates) == 1 and hasattr(coordinates[0], "__len__"):
            coordinates = coordinates[0]

        if len(coordinates) != 4:
            continue

        _x1, y1, _x2, y2 = (int(value) for value in coordinates)
        if abs(y1 - y2) < tolerance:
            return (y1 + y2) // 2

    return None
