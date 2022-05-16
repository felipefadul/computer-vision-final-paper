import cv2
import numpy as np


# A Python3 program to find if 2 given line segments intersect or not

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


pt1 = Point(100, 100)
pt2 = Point(500, 100)
pt3 = Point(500, 500)
pt4 = Point(100, 500)
pt5 = Point(200, 200)
pt6 = Point(200, 300)
pt7 = Point(400, 500)
pt8 = Point(600, 300)
pt9 = Point(300, 100)
pt10 = Point(100, 400)
pt11 = Point(300, 500)
pt12 = Point(400, 300)
pt13 = Point(700, 400)
pt14 = Point(100, 200)
pt15 = Point(200, 400)
pt16 = Point(500, 400)
pt17 = Point(600, 200)

default_line = [pt14, pt15, pt16, pt17]

# Test 1
# first_polyline = [Point(100, 50), Point(100, 400)]
# second_polyline = [Point(0, 0), Point(100, 100), Point(0, 200)]

# Test 2
# first_polyline = [pt1, pt5, pt6, pt7, pt8, pt9]
# second_polyline = [pt10, pt11, pt12, pt13]

# Test 3
# first_polyline = [pt1, pt2, pt3, pt4, pt5, pt6]
# second_polyline = [pt10, pt11, pt12, pt13]

# Test 4
# first_polyline = [pt1, pt2, pt3, Point(100, 600), pt5, pt6]
# second_polyline = [pt10, pt11, pt12, pt13]

# Test 5
# first_polyline = [pt1, pt2, pt3, pt4, pt5, pt6]
# second_polyline = [pt10, pt11, Point(500, 300), pt13]

# Test 6
# first_polyline = [pt1, pt2, pt3, pt4, pt5, pt6]
# second_polyline = [pt10, pt11, Point(300, 400), Point(500, 300), pt13]

# Test 7
# first_polyline = [pt1, pt2, pt3, pt4, pt5, pt6]
# second_polyline = [pt10, pt11, Point(400, 500), Point(500, 300), pt13]

# Test 8
# first_polyline = [pt1, pt2, pt3, pt4, pt5, pt6]
# second_polyline = [pt11, Point(500, 300)]

# Test 9
# first_polyline = default_line
# second_polyline = [pt11, Point(500, 300)]

# Test 10
# first_polyline = default_line
# second_polyline = [Point(200, 300), Point(300, 400), Point(400, 300)]

# Test 11
# first_polyline = default_line
# second_polyline = [Point(200, 300), Point(350, 400), Point(550, 300)]

# Test 12
first_polyline = default_line
second_polyline = [Point(200, 300), Point(350, 400), Point(200, 500)]


class PolylinesIntersectionsCounter:
    
    def find_line_segments_intersection(self, s1, s2, t1, t2):
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
            return round(xi, 0), round(yi, 0)  # Return the intersection point
        else:
            return None
    
    def count_intersections(self, polyline1, polyline2):
        intersection_points_list = []
        intersections_count = 0
        
        for i, p1_first_point in enumerate(polyline1[:-1]):
            p1_second_point = polyline1[i + 1]
            
            for j, p2_first_point in enumerate(polyline2[:-1]):
                p2_second_point = polyline2[j + 1]
                
                intersection_point = self.find_line_segments_intersection(p1_first_point, p1_second_point,
                                                                          p2_first_point, p2_second_point)
                if intersection_point and intersection_point not in intersection_points_list:
                    intersection_points_list.append(intersection_point)
                    intersections_count += 1
        
        return intersections_count, intersection_points_list


def draw_points(points_list, color):
    for point in points_list:
        cv2.circle(img, np.int32(point), 0, color, 10)


def draw_polyline(polyline, color):
    polyline_aux = []
    for point in polyline:
        polyline_aux.append((point.x, point.y))
    cv2.polylines(img, np.int32([polyline_aux]), False, color, 10)
    return polyline_aux


if __name__ == "__main__":
    h, w = 700, 700
    img = np.zeros((h, w, 3), np.uint8)
    
    first_polyline_aux = draw_polyline(first_polyline, (255, 0, 0))
    print('first_polyline_aux: ', first_polyline_aux)
    second_polyline_aux = draw_polyline(second_polyline, (0, 255, 255))
    print('second_polyline_aux: ', second_polyline_aux)
    
    polylines_intersections_counter = PolylinesIntersectionsCounter()
    
    intersection_count, intersection_points = polylines_intersections_counter.count_intersections(first_polyline,
                                                                                                  second_polyline)
    print('Number of intersections: ', intersection_count)
    print('Intersection points: ', intersection_points)
    
    draw_points(intersection_points, (0, 0, 255))
    # draw_points(first_polyline_aux, (0, 255, 0))
    # draw_points(second_polyline_aux, (255, 255, 0))
    
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
