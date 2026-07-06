import unittest

from src.desk_edge_detection import find_horizontal_edge_y


class DeskEdgeDetectionTests(unittest.TestCase):
    def test_accepts_opencv_n_by_1_by_4_shape(self):
        lines = [[[10, 120, 200, 122]]]
        self.assertEqual(find_horizontal_edge_y(lines), 121)

    def test_accepts_opencv_n_by_4_shape(self):
        lines = [[10, 120, 200, 122]]
        self.assertEqual(find_horizontal_edge_y(lines), 121)

    def test_skips_non_horizontal_lines(self):
        lines = [[10, 20, 200, 100], [10, 90, 200, 92]]
        self.assertEqual(find_horizontal_edge_y(lines), 91)

    def test_returns_none_without_lines(self):
        self.assertIsNone(find_horizontal_edge_y(None))


if __name__ == "__main__":
    unittest.main()
