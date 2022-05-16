from count_polylines_intersections import Point


class TrackableObject:
    def __init__(self, object_id, centroid):
        # Store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        self.centroids = [Point(centroid[0], centroid[1])]
        
        # Initialize a boolean used to indicate if the object has
        # already been pre-counted or not
        self.pre_counted = False
        
        self.intersections_count = 0
