import unittest

from src.count_polylines_intersections import DirectionOptions, PolylinesIntersectionsCounter
from src.point import Point


class CountIntersectionTest(unittest.TestCase):
    def setUp(self):
        self.polylines_intersections_counter = PolylinesIntersectionsCounter()
        self.pt1 = Point(100, 100)
        self.pt2 = Point(500, 100)
        self.pt3 = Point(500, 500)
        self.pt4 = Point(100, 500)
        self.pt5 = Point(200, 200)
        self.pt6 = Point(200, 300)
        self.pt7 = Point(400, 500)
        self.pt8 = Point(600, 300)
        self.pt9 = Point(300, 100)
        self.pt10 = Point(100, 400)
        self.pt11 = Point(300, 500)
        self.pt12 = Point(400, 300)
        self.pt13 = Point(700, 400)
        self.pt14 = Point(100, 200)
        self.pt15 = Point(200, 400)
        self.pt16 = Point(500, 400)
        self.pt17 = Point(600, 200)
        self.default_line = [self.pt14, self.pt15, self.pt16, self.pt17]
    
    def test_count_intersection_01(self):
        first_polyline = [Point(100, 50), Point(100, 400)]
        second_polyline = [Point(0, 0), Point(100, 100), Point(0, 200)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(1, intersections_count, 'ERROR!')
    
    def test_count_intersection_02(self):
        first_polyline = [self.pt1, self.pt5, self.pt6, self.pt7, self.pt8, self.pt9]
        second_polyline = [self.pt10, self.pt11, self.pt12, self.pt13]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(2, intersections_count, 'ERROR!')
    
    def test_count_intersection_03(self):
        first_polyline = [self.pt1, self.pt2, self.pt3, self.pt4, self.pt5, self.pt6]
        second_polyline = [self.pt10, self.pt11, self.pt12, self.pt13]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(3, intersections_count, 'ERROR!')
    
    def test_count_intersection_04(self):
        first_polyline = [self.pt1, self.pt2, self.pt3, Point(100, 600), self.pt5, self.pt6]
        second_polyline = [self.pt10, self.pt11, self.pt12, self.pt13]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(2, intersections_count, 'ERROR!')
    
    def test_count_intersection_05(self):
        first_polyline = [self.pt1, self.pt2, self.pt3, self.pt4, self.pt5, self.pt6]
        second_polyline = [self.pt10, self.pt11, Point(500, 300), self.pt13]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(3, intersections_count, 'ERROR!')
    
    def test_count_intersection_06(self):
        first_polyline = [self.pt1, self.pt2, self.pt3, self.pt4, self.pt5, self.pt6]
        second_polyline = [self.pt10, self.pt11, Point(300, 400), Point(500, 300), self.pt13]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(3, intersections_count, 'ERROR!')
    
    def test_count_intersection_07(self):
        first_polyline = [self.pt1, self.pt2, self.pt3, self.pt4, self.pt5, self.pt6]
        second_polyline = [self.pt10, self.pt11, Point(400, 500), Point(500, 300), self.pt13]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(4, intersections_count, 'ERROR!')
    
    def test_count_intersection_08(self):
        first_polyline = [self.pt1, self.pt2, self.pt3, self.pt4, self.pt5, self.pt6]
        second_polyline = [self.pt11, Point(500, 300)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(2, intersections_count, 'ERROR!')
    
    def test_count_intersection_09(self):
        first_polyline = self.default_line
        second_polyline = [self.pt11, Point(500, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(1, intersections_count, 'ERROR!')
        self.assertEqual(DirectionOptions.UP, direction, 'ERROR!')
    
    def test_count_intersection_10(self):
        first_polyline = self.default_line
        second_polyline = [Point(200, 300), Point(300, 400), Point(400, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(1, intersections_count, 'ERROR!')
        self.assertEqual(None, direction, 'ERROR!')
    
    def test_count_intersection_11(self):
        first_polyline = self.default_line
        second_polyline = [Point(200, 300), Point(350, 400), Point(550, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(2, intersections_count, 'ERROR!')
        self.assertEqual(None, direction, 'ERROR!')
    
    def test_count_intersection_12(self):
        first_polyline = self.default_line
        second_polyline = [Point(200, 300), Point(350, 400), Point(200, 500)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(1, intersections_count, 'ERROR!')
        self.assertEqual(DirectionOptions.DOWN, direction, 'ERROR!')
    
    def test_count_intersection_13(self):
        first_polyline = self.default_line
        second_polyline = [Point(200, 300), Point(350, 450), Point(200, 500)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(1, intersections_count, 'ERROR!')
        self.assertEqual(DirectionOptions.DOWN, direction, 'ERROR!')
    
    def test_count_intersection_14(self):
        first_polyline = self.default_line
        second_polyline = [Point(200, 300), Point(300, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(0, intersections_count, 'ERROR!')
        self.assertEqual(None, direction, 'ERROR!')


if __name__ == "__main__":
    unittest.main()
