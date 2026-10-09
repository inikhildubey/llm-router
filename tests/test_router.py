import unittest
from router import Triage, Router, RouterConfig

class TestRouter(unittest.TestCase):
    def setUp(self):
        self.config = RouterConfig(
            low_threshold=4,
            high_threshold=8,
            model_mapping={
                "low": "model-low",
                "medium": "model-medium",
                "high": "model-high"
            }
        )
        self.router = Router(self.config)

    def test_route_low(self):
        self.assertEqual(self.router.route(2), "model-low")
        self.assertEqual(self.router.route(3), "model-low")

    def test_route_medium(self):
        self.assertEqual(self.router.route(4), "model-medium")
        self.assertEqual(self.router.route(7), "model-medium")

    def test_route_high(self):
        self.assertEqual(self.router.route(8), "model-high")
        self.assertEqual(self.router.route(10), "model-high")

if __name__ == "__main__":
    unittest.main()
