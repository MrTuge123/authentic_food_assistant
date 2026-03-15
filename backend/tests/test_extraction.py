import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.extraction import extract_structured


class ExtractionRegressionTests(unittest.TestCase):
    def test_filters_capitalized_sentence_starters(self):
        structured = extract_structured(
            "Best Sushi in Ann Arbor",
            [
                "Tabe is the answer here. Incredible sushi, nice cocktails, great formal vibe.",
                "Hinodae was great. I'd love to find a decent alternative.",
                "What a blast from the past... definitely one of my go to locations in the late 80's.",
                "Every time I go for lunch it takes me back to AA.",
            ],
        )

        names = {candidate["name"] for candidate in structured["restaurant_candidates"]}
        self.assertIn("Hinodae", names)
        self.assertNotIn("Incredible", names)
        self.assertNotIn("I'd", names)
        self.assertNotIn("What", names)
        self.assertNotIn("Every", names)
        self.assertNotIn("80's", names)

    def test_keeps_real_restaurants_and_trims_dish_names(self):
        structured = extract_structured(
            "Best vegetarian ramen?",
            [
                "Slurping Turtle Red Curry Ramen is the best veg option imo!",
                "I love mama satto’s shoyu ramen.",
            ],
        )

        names = {candidate["name"] for candidate in structured["restaurant_candidates"]}
        self.assertIn("Slurping Turtle", names)
        self.assertNotIn("Slurping Turtle Red Curry Ramen", names)
        self.assertIn("red curry ramen", structured["dish"])

    def test_title_noise_is_not_extracted_as_restaurant(self):
        structured = extract_structured(
            "Pho Ben in the Heights Houston TX Vietnamese Refugees Brought The Best From Vietnam To America.",
            [],
        )

        self.assertEqual(["Pho Ben"], structured["restaurant_name"])

    def test_non_restaurant_tooling_sentence_is_ignored(self):
        structured = extract_structured(
            "A year of work mapping U.S. regional food traditions [OC]",
            [
                "Base map + county shapefiles, GIS editing (QGIS), Adobe Illustrator for final layout and labeling."
            ],
        )

        self.assertEqual([], structured["restaurant_candidates"])

    def test_pronoun_contractions_are_not_restaurants(self):
        structured = extract_structured(
            "Best vegetarian ramen?",
            [
                "They’re super allergen friendly they really try to keep everything separated there which I really appreciate about them."
            ],
        )

        names = {candidate["name"] for candidate in structured["restaurant_candidates"]}
        self.assertNotIn("Theyre", names)

    def test_extracts_short_recommendation_names(self):
        structured = extract_structured(
            "Best Sushi in Ann Arbor",
            [
                "Tabe is the answer here.",
                "I like Totoro and mama sato.",
            ],
        )

        names = {candidate["name"] for candidate in structured["restaurant_candidates"]}
        self.assertIn("Tabe", names)
        self.assertIn("Totoro", names)
        self.assertIn("Mama Sato", names)

    def test_extracts_short_lowercase_pho_names(self):
        structured = extract_structured(
            "Best pho in town",
            [
                "pho dan",
                "Pho Van",
            ],
        )

        names = {candidate["name"] for candidate in structured["restaurant_candidates"]}
        self.assertIn("Pho Dan", names)
        self.assertIn("Pho Van", names)

    def test_filters_acronym_locations_from_restaurant_candidates(self):
        structured = extract_structured(
            "Pho in southeast side",
            [
                "While not the best broth, Pho 21 on the backside of NASA JSC has the best soft tendon of anywhere in Houston."
            ],
        )

        names = {candidate["name"] for candidate in structured["restaurant_candidates"]}
        self.assertIn("Pho 21", names)
        self.assertNotIn("NASA JSC", names)
        self.assertNotIn("While", names)


if __name__ == "__main__":
    unittest.main()
