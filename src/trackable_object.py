from count_polylines_intersections import Point


class TrackableObject:
    def __init__(self, object_id, centroid):
        # Store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        self.centroids = [Point(centroid[0], centroid[1])]
