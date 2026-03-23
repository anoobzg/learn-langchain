import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import get_weather


class WeatherToolTests(unittest.TestCase):
    def test_get_weather_returns_expected_text(self) -> None:
        self.assertEqual(get_weather("SF"), "It's always sunny in SF!")


if __name__ == "__main__":
    unittest.main()
