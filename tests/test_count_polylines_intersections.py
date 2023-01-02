import unittest

from src.count_polylines_intersections import DirectionOptions, PolylinesIntersectionsCounter
from src.point import Point


class CountIntersectionTest(unittest.TestCase):
    def setUp(self):
        self.polylines_intersections_counter = PolylinesIntersectionsCounter()
        self.default_line = [Point(100, 200), Point(200, 400), Point(500, 400), Point(600, 200)]
    
    def test_count_intersection_01(self):
        first_polyline = [Point(100, 50), Point(100, 400)]
        second_polyline = [Point(0, 0), Point(100, 100), Point(0, 200)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(1, intersections_count, 'ERROR!')
    
    def test_count_intersection_02(self):
        first_polyline = [Point(100, 100), Point(200, 200), Point(200, 300), Point(400, 500), Point(600, 300),
                          Point(300, 100)]
        second_polyline = [Point(100, 400), Point(300, 500), Point(400, 300), Point(700, 400)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(2, intersections_count, 'ERROR!')
    
    def test_count_intersection_03(self):
        first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
                          Point(200, 300)]
        second_polyline = [Point(100, 400), Point(300, 500), Point(400, 300), Point(700, 400)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(3, intersections_count, 'ERROR!')
    
    def test_count_intersection_04(self):
        first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 600), Point(200, 200),
                          Point(200, 300)]
        second_polyline = [Point(100, 400), Point(300, 500), Point(400, 300), Point(700, 400)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(2, intersections_count, 'ERROR!')
    
    def test_count_intersection_05(self):
        first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
                          Point(200, 300)]
        second_polyline = [Point(100, 400), Point(300, 500), Point(500, 300), Point(700, 400)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(3, intersections_count, 'ERROR!')
    
    def test_count_intersection_06(self):
        first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
                          Point(200, 300)]
        second_polyline = [Point(100, 400), Point(300, 500), Point(300, 400), Point(500, 300), Point(700, 400)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(3, intersections_count, 'ERROR!')
    
    def test_count_intersection_07(self):
        first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
                          Point(200, 300)]
        second_polyline = [Point(100, 400), Point(300, 500), Point(400, 500), Point(500, 300), Point(700, 400)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(4, intersections_count, 'ERROR!')
    
    def test_count_intersection_08(self):
        first_polyline = [Point(100, 100), Point(500, 100), Point(500, 500), Point(100, 500), Point(200, 200),
                          Point(200, 300)]
        second_polyline = [Point(300, 500), Point(500, 300)]
        intersections_count = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)[0]
        self.assertEqual(2, intersections_count, 'ERROR!')
    
    def test_count_intersection_09(self):
        first_polyline = self.default_line
        second_polyline = [Point(300, 500), Point(500, 300)]
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
        second_polyline = [Point(200, 500), Point(350, 400), Point(200, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(1, intersections_count, 'ERROR!')
        self.assertEqual(DirectionOptions.UP, direction, 'ERROR!')
    
    def test_count_intersection_14(self):
        first_polyline = self.default_line
        second_polyline = [Point(200, 300), Point(350, 450), Point(200, 500)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(1, intersections_count, 'ERROR!')
        self.assertEqual(DirectionOptions.DOWN, direction, 'ERROR!')
    
    def test_count_intersection_15(self):
        first_polyline = self.default_line
        second_polyline = [Point(350, 450), Point(200, 500), Point(200, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(1, intersections_count, 'ERROR!')
        self.assertEqual(DirectionOptions.UP, direction, 'ERROR!')
    
    def test_count_intersection_16(self):
        first_polyline = self.default_line
        second_polyline = [Point(200, 300), Point(300, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(0, intersections_count, 'ERROR!')
        self.assertEqual(None, direction, 'ERROR!')
    
    def test_count_intersection_17(self):
        first_polyline = self.default_line
        second_polyline = [Point(250, 400), Point(350, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(1, intersections_count, 'ERROR!')
        self.assertEqual(None, direction, 'ERROR!')
    
    def test_count_intersection_18(self):
        first_polyline = self.default_line
        second_polyline = [Point(250, 300), Point(250, 500), Point(350, 300), Point(350, 500), Point(450, 300),
                           Point(450, 500)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(5, intersections_count, 'ERROR!')
        self.assertEqual(DirectionOptions.DOWN, direction, 'ERROR!')
    
    def test_count_intersection_19(self):
        first_polyline = self.default_line
        second_polyline = [Point(450, 500), Point(450, 300), Point(350, 500), Point(350, 300), Point(250, 500),
                           Point(250, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(5, intersections_count, 'ERROR!')
        self.assertEqual(DirectionOptions.UP, direction, 'ERROR!')
    
    def test_count_intersection_20(self):
        first_polyline = self.default_line
        second_polyline = [Point(250, 300), Point(250, 500), Point(350, 300), Point(350, 500), Point(450, 300)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(4, intersections_count, 'ERROR!')
        self.assertEqual(None, direction, 'ERROR!')
    
    def test_count_intersection_21(self):
        first_polyline = self.default_line
        second_polyline = [Point(450, 500), Point(450, 300), Point(350, 500), Point(350, 300), Point(250, 500)]
        intersections_count, intersection_points, first_point_of_the_first_segment, last_point_of_the_last_segment = self.polylines_intersections_counter.count_intersections(
            first_polyline, second_polyline)
        direction = self.polylines_intersections_counter.get_intersection_direction(intersection_points,
                                                                                    first_point_of_the_first_segment,
                                                                                    last_point_of_the_last_segment)
        self.assertEqual(4, intersections_count, 'ERROR!')
        self.assertEqual(None, direction, 'ERROR!')


if __name__ == "__main__":
    unittest.main()
