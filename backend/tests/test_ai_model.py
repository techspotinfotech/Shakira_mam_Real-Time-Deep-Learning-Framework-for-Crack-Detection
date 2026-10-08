import unittest

from app.ai_model import analyze_project_idea


class AIModelTests(unittest.TestCase):
    def test_analyze_project_idea_returns_score_and_recommendation(self):
        result = analyze_project_idea(
            title="Smart Project Analyzer",
            domain="AI/ML",
            description="Build a dashboard that evaluates research ideas, scores feasibility, and recommends execution steps for an AI-based project.",
        )

        self.assertIn("project_title", result)
        self.assertIn("feasibility_score", result)
        self.assertIn("risk_level", result)
        self.assertIn("recommendation", result)
        self.assertGreaterEqual(result["feasibility_score"], 70)
        self.assertIn("AI/ML", result["matched_focus_areas"])


if __name__ == "__main__":
    unittest.main()
