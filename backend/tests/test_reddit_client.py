import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.reddit_client import (
    _extract_query_locations,
    _food_intent_score,
    _location_relevance_score,
    _subreddit_matches_query_location,
)


class RedditRetrievalRegressionTests(unittest.TestCase):
    def test_extract_query_locations_ignores_food_words(self):
        self.assertEqual(["seattle"], _extract_query_locations("authentic sushi seattle"))
        self.assertEqual(["ann", "arbor"], _extract_query_locations("best sushi ann arbor"))

    def test_location_score_allows_generic_subreddit_when_content_matches(self):
        score = _location_relevance_score(
            "authentic sushi seattle",
            "Seattle/Bellevue Sushi",
            ["Good omakase in Seattle"],
            "sushi",
        )
        self.assertGreaterEqual(score, 2)
        self.assertFalse(_subreddit_matches_query_location("authentic sushi seattle", "sushi"))

    def test_city_specific_subreddit_is_recognized(self):
        self.assertTrue(_subreddit_matches_query_location("best pho houston", "HoustonFood"))
        self.assertFalse(_subreddit_matches_query_location("best pho houston", "austinfood"))

    def test_non_food_threads_score_poorly(self):
        score = _food_intent_score(
            "best ramen ann arbor",
            "City Administrator mandates return to office for Ann Arbor employees.",
            ["More PTO for school closures or emergencies."],
        )
        self.assertLess(score, 2)


if __name__ == "__main__":
    unittest.main()
