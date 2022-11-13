from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist

from count_polylines_intersections import PolylinesIntersectionsCounter
from src.utils.constants import OBJECT_ID_TO_DEBUG


class CentroidTracker:
    def __init__(self, default_line, max_disappeared=50, max_distance=50):
        # Initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object ID
        # to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bounding_box = OrderedDict()
        
        # Store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.max_disappeared = max_disappeared
        
        # Store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.max_distance = max_distance
        
        self.polylines_intersections_counter = PolylinesIntersectionsCounter()
        self.default_line = default_line
    
    def register(self, centroid, input_rect):
        # When registering an object we use the next available object ID
        # to store the centroid
        self.objects[self.next_object_id] = centroid
        self.bounding_box[self.next_object_id] = input_rect
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        # To deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bounding_box[object_id]
    
    def count_intersections(self, trackable_objects, object_id):
        to = trackable_objects.get(object_id, None)
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            self.default_line, to.centroids)
        if OBJECT_ID_TO_DEBUG == object_id:
            print('Test deregister - object_id', object_id)
            print('Test deregister - to.object_id', to.object_id)
            print('Test deregister - to.centroids', to.centroids)
            print('Test deregister - intersections_count', intersections_count)
            print('Test deregister - intersection_points', intersection_points)
        return intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment
    
    def update(self, rects, trackable_objects):
        intersections_list = []
        # Check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # Loop over any existing tracked objects and mark them
            # as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # If we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[object_id] > self.max_disappeared:
                    intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.count_intersections(
                        trackable_objects, object_id)
                    direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                                first_point_of_the_first_segment,
                                                                                                last_point_of_the_last_segment)
                    self.deregister(object_id)
                    intersections_list.append((object_id, intersections_count, direction))
            
            # Return early as there are no centroids or tracking info
            # to update
            return self.bounding_box, intersections_list
        
        # Initialize an array of input centroids for the current frame
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_rects = []
        
        # Loop over the bounding box rectangles
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            # Use the bounding box coordinates to derive the centroid
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y)
            input_rects.append(rects[i])
        
        # If we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], input_rects[i])
        
        # Otherwise, we are currently tracking objects, so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # Grab the set of object IDs and corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            if OBJECT_ID_TO_DEBUG in object_ids:
                print('List of object_ids', object_ids)
                print('List of object_centroids', object_centroids)
            
            # Compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            distance = dist.cdist(np.array(object_centroids), input_centroids)
            
            # In order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = distance.min(axis=1, initial=None).argsort()
            
            # Next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            columns = distance.argmin(axis=1)[rows]
            
            # In order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            used_rows = set()
            used_columns = set()
            
            # Loop over the combination of the (row, column) index
            # tuples
            for (row, column) in zip(rows, columns):
                # If we have already examined either the row or
                # column value before, ignore it
                if row in used_rows or column in used_columns:
                    continue
                
                # If the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if distance[row, column] > self.max_distance:
                    continue
                
                to = trackable_objects.get(object_ids[row], None)
                if to is not None and not to.pre_counted:
                    # Otherwise, grab the object ID for the current row,
                    # set its new centroid, and reset the disappeared
                    # counter
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[column]
                    self.bounding_box[object_id] = input_rects[column]
                    self.disappeared[object_id] = 0
                    
                    # Indicate that we have examined each of the row and
                    # column indexes, respectively
                    used_rows.add(row)
                    used_columns.add(column)
            
            # Compute both the row and column index we have NOT yet
            # examined
            unused_rows = set(range(0, distance.shape[0])).difference(used_rows)
            unused_columns = set(range(0, distance.shape[1])).difference(used_columns)
            
            # In the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if distance.shape[0] >= distance.shape[1]:
                # Loop over the unused row indexes
                for row in unused_rows:
                    # Grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    # Check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[object_id] > self.max_disappeared:
                        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.count_intersections(
                            trackable_objects, object_id)
                        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                                    first_point_of_the_first_segment,
                                                                                                    last_point_of_the_last_segment)
                        self.deregister(object_id)
                        intersections_list.append((object_id, intersections_count, direction))
            
            # Otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for column in unused_columns:
                    self.register(input_centroids[column], input_rects[column])
        
        return self.bounding_box, intersections_list
