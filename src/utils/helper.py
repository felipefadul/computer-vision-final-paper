def get_centroid_from_bounding_box(bounding_box):
    start_x, start_y, end_x, end_y = bounding_box.astype("int")
    
    center_x = (start_x + end_x) // 2
    center_y = (start_y + end_y) // 2
    centroid = (center_x, center_y)
    
    return centroid
