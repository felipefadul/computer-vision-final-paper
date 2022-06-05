from enum import Enum

import cv2
import numpy as np

from src.point import Point

# A Python3 program to find if 2 given line segments intersect or not

DirectionOptions = Enum("DirectionOptions", ["UP", "DOWN"])


class PolylinesIntersectionsCounter:
    
    @staticmethod
    def find_line_segments_intersection(s1, s2, t1, t2):
        """Method to check the intersection of two line segments. Returns
        None if no intersection, or a coordinate indicating the intersection.

        An implementation from the NCGIA core curriculum. s1 and s2 are points
        (e.g.: 2-item tuples) marking the beginning and end of segment s. t1
        and t2 are points marking the beginning and end of segment t. Each point
        has an x and y coordinate: (1, 3).
        Variables are named following linear formula: y = a + bx.

        This method was adapted from this Stack Overflow question:
        https://stackoverflow.com/questions/26152787/counting-line-intersections-from-lists-in-python"""
        if s1.x != s2.x:  # If s is not vertical
            b1 = (s2.y - s1.y) / float(s2.x - s1.x)
            if t1.x != t2.x:  # If t is not vertical
                b2 = (t2.y - t1.y) / float(t2.x - t1.x)
                if b1 == b2:  # If lines are parallel (slopes match)
                    return None
                a1 = s1.y - (b1 * s1.x)
                a2 = t1.y - (b2 * t1.x)
                xi = -(a1 - a2) / float(b1 - b2)  # Solve for intersection point
                yi = a1 + (b1 * xi)
            else:
                xi = t1.x
                a1 = s1.y - (b1 * s1.x)
                yi = a1 + (b1 * xi)
        else:
            xi = s1.x
            if t1.x != t2.x:  # If t is not vertical
                b2 = (t2.y - t1.y) / float(t2.x - t1.x)
                a2 = t1.y - (b2 * t1.x)
                yi = a2 + (b2 * xi)
            else:
                return None
        # Here is the actual intersection test!
        if (s1.x - xi) * (xi - s2.x) >= 0 and \
                (s1.y - yi) * (yi - s2.y) >= 0 and \
                (t1.x - xi) * (xi - t2.x) >= 0 and \
                (t1.y - yi) * (yi - t2.y) >= 0:
            return Point(round(xi, 0), round(yi, 0))  # Return the intersection point
        else:
            return None
    
    def count_intersections(self, polyline1, polyline2):
        intersection_points_list = []
        intersections_count = 0
        first_point_of_the_first_segment = Point(0, 0)
        last_point_of_the_last_segment = Point(0, 0)
        
        for i, p1_first_point in enumerate(polyline1[:-1]):
            p1_second_point = polyline1[i + 1]
            
            for j, p2_first_point in enumerate(polyline2[:-1]):
                p2_second_point = polyline2[j + 1]
                if j == 0:
                    first_point_of_the_first_segment = p2_first_point
                if j == len(polyline2) - 2:
                    last_point_of_the_last_segment = p2_second_point
                
                intersection_point = self.find_line_segments_intersection(p1_first_point, p1_second_point,
                                                                          p2_first_point, p2_second_point)
                if intersection_point and intersection_point not in intersection_points_list:
                    intersection_points_list.append(intersection_point)
                    intersections_count += 1
        
        return intersections_count, intersection_points_list, first_point_of_the_first_segment, last_point_of_the_last_segment
    
    @staticmethod
    def get_intersection_direction(intersection_points_list,
                                   first_point_of_the_first_segment,
                                   last_point_of_the_last_segment):
        # Check if the list is empty or if the number of intersection points is even, which means that the polyline is
        # in the same subspace determined by the default polyline and hasn't crossed it.
        if not intersection_points_list or len(intersection_points_list) % 2 == 0:
            return None
        
        # If the first y-point of the first segment is above the y-point of the first intersection AND the last y-point
        # of the last segment is below the y-point of the last intersection, it means the direction is DOWN.
        if (first_point_of_the_first_segment.y < intersection_points_list[0].y and
                last_point_of_the_last_segment.y > intersection_points_list[-1].y):
            return DirectionOptions.DOWN
        
        # If the first y-point of the first segment is below the y-point of the first intersection AND the last y-point
        # of the last segment is above the y-point of the last intersection, it means the direction is UP.
        if (first_point_of_the_first_segment.y > intersection_points_list[0].y and
                last_point_of_the_last_segment.y < intersection_points_list[-1].y):
            return DirectionOptions.UP
        
        return None


def draw_points(points_list, color, img):
    for point in points_list:
        cv2.circle(img, np.int32((point.x, point.y)), 0, color, 10)


def draw_polyline(polyline, other_points_color, img, first_point_color=(255, 255, 0)):
    polyline_aux = []
    first_point = (0, 0)
    for i, point in enumerate(polyline):
        polyline_aux.append((point.x, point.y))
        if i == 0:
            first_point = (point.x, point.y)
    cv2.polylines(img, np.int32([polyline_aux]), False, other_points_color, 10)
    # Painting the first point with a different color help us to check the direction when debugging
    cv2.circle(img, np.int32(first_point), 0, first_point_color, 10)


if __name__ == "__main__":
    default_line = [Point(100, 200), Point(200, 400), Point(500, 400), Point(600, 200)]
    
    # Test 1
    # first_polyline = [Point(100, 50), Point(100, 400)]
    # second_polyline = [Point(0, 0), Point(100, 100), Point(0, 200)]
    
    # Test 2
    # first_polyline = [Point(100, 100), Point(200, 200), Point(200, 300), Point(400, 500), Point(600, 300),
    #                   Point(300, 100)]
    # second_polyline = [Point(100, 400), Point(300, 500), Point(400, 300), Point(700, 400)]
    
    # Test 3
    # first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
    #                   Point(200, 300)]
    # second_polyline = [Point(100, 400), Point(300, 500), Point(400, 300), Point(700, 400)]
    
    # Test 4
    # first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 600), Point(200, 200),
    #                   Point(200, 300)]
    # second_polyline = [Point(100, 400), Point(300, 500), Point(400, 300), Point(700, 400)]
    
    # Test 5
    # first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
    #                   Point(200, 300)]
    # second_polyline = [Point(100, 400), Point(300, 500), Point(500, 300), Point(700, 400)]
    
    # Test 6
    # first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
    #                   Point(200, 300)]
    # second_polyline = [Point(100, 400), Point(300, 500), Point(300, 400), Point(500, 300), Point(700, 400)]
    
    # Test 7
    # first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
    #                   Point(200, 300)]
    # second_polyline = [Point(100, 400), Point(300, 500), Point(400, 500), Point(500, 300), Point(700, 400)]
    
    # Test 8
    # first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
    #                   Point(200, 300)]
    # second_polyline = [Point(300, 500), Point(500, 300)]
    
    # Test 9
    # first_polyline = default_line
    # second_polyline = [Point(300, 500), Point(500, 300)]
    
    # Test 10
    # first_polyline = default_line
    # second_polyline = [Point(200, 300), Point(300, 400), Point(400, 300)]
    
    # Test 11
    # first_polyline = default_line
    # second_polyline = [Point(200, 300), Point(350, 400), Point(550, 300)]
    
    # Test 12
    # first_polyline = default_line
    # second_polyline = [Point(200, 300), Point(350, 400), Point(200, 500)]
    
    # Test 13
    # first_polyline = default_line
    # second_polyline = [Point(200, 500), Point(350, 400), Point(200, 300)]
    
    # Test 14
    # first_polyline = default_line
    # second_polyline = [Point(200, 300), Point(350, 450), Point(200, 500)]
    
    # Test 15
    # first_polyline = default_line
    # second_polyline = [Point(350, 450), Point(200, 500), Point(200, 300)]
    
    # Test 16
    # first_polyline = default_line
    # second_polyline = [Point(200, 300), Point(300, 300)]
    
    # Test 17
    # first_polyline = default_line
    # second_polyline = [Point(250, 300), Point(250, 500), Point(350, 300), Point(350, 500), Point(450, 300),
    #                    Point(450, 500)]
    
    # Test 18
    # first_polyline = default_line
    # second_polyline = [Point(450, 500), Point(450, 300), Point(350, 500), Point(350, 300), Point(250, 500),
    #                    Point(250, 300)]
    
    # Test 19
    # first_polyline = default_line
    # second_polyline = [Point(250, 300), Point(250, 500), Point(350, 300), Point(350, 500), Point(450, 300)]
    
    # Test 20
    first_polyline = default_line
    second_polyline = [Point(450, 500), Point(450, 300), Point(350, 500), Point(350, 300), Point(250, 500)]
    
    
    def main():
        h, w = 700, 700
        img = np.zeros((h, w, 3), np.uint8)
        
        draw_polyline(first_polyline, (255, 0, 0), img)
        print('First polyline: ', first_polyline)
        draw_polyline(second_polyline, (0, 255, 255), img)
        print('Second polyline: ', second_polyline)
        
        polylines_intersections_counter = PolylinesIntersectionsCounter()
        
        intersection_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = polylines_intersections_counter.count_intersections(
            first_polyline,
            second_polyline)
        print('Number of intersections: ', intersection_count)
        print('Intersection points: ', intersection_points)
        
        direction = polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                               first_point_of_the_first_segment,
                                                                               last_point_of_the_last_segment)
        print('Direction: ', direction)
        
        draw_points(intersection_points, (0, 0, 255), img)
        # draw_points(first_polyline, (0, 255, 0), img)
        # draw_points(second_polyline, (255, 255, 0), img)
        
        cv2.imshow("image", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    
    main()
