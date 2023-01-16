def get_centroid_from_bounding_box(bounding_box):
    start_x, start_y, end_x, end_y = bounding_box.astype("int")
    
    center_x = (start_x + end_x) // 2
    center_y = (start_y + end_y) // 2
    centroid = (center_x, center_y)
    
    return centroid


def get_rectangle_points_indicating_edge_proximity(width, height):
    rectangle_edge_proximity_start_x = int(0.1 * width)
    rectangle_edge_proximity_start_y = int(0.3 * height)
    rectangle_edge_proximity_end_x = int(0.9 * width)
    rectangle_edge_proximity_end_y = int(0.7 * height)
    rectangle_edge_proximity = (
        rectangle_edge_proximity_start_x, rectangle_edge_proximity_start_y,
        rectangle_edge_proximity_end_x, rectangle_edge_proximity_end_y)
    return rectangle_edge_proximity
