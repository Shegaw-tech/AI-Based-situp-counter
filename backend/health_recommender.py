from typing import List, Dict


class HealthRecommender:
    def __init__(self):
        self.recommendation_db = {
            'beginner': [
                "Start with 3 sets of 5-10 sit-ups with 60s rest",
                "Focus on controlled movements rather than speed",
                "Engage your core throughout the entire movement"

            ],
            'intermediate': [
                "Try 4 sets of 12-15 sit-ups with 45s rest",
                "Add a twist at the top to engage obliques",
                "Try pausing for 2 seconds at the top position"
            ],
            'advanced': [
                "Perform 5 sets of 20+ sit-ups with 30s rest",
                "Add weight by holding a plate to your chest",
                "Try decline bench sit-ups for more resistance"
            ],
            'form': {
                'shoulder_symmetry': [
                    "Your shoulders are uneven - focus on symmetry",
                    "Keep both shoulders at the same height",
                    "Avoid leaning to one side"
                ],
                'back_angle': [
                    "Maintain a straight back throughout",
                    "Avoid rounding your back too much",
                    "Engage your core to protect your spine"
                ]
            },
            'full': [
                    "For full sit-ups: Control the descent",
                    "Full sit-ups: Keep your feet anchored",
                    "Full sit-ups: Use your arms for momentum if needed"
             ]
        }

    def get_recommendations(self, count: int, form_metrics: Dict) -> List[str]:
        recommendations = []
        level = self._determine_level(count)

        # Level-based recommendations
        recommendations.extend(self.recommendation_db[level][:2])

        # Form-specific recommendations
        if form_metrics.get('shoulder_symmetry', 0) > 0.1:
            recommendations.append(self.recommendation_db['form']['shoulder_symmetry'][0])
        if form_metrics.get('back_angle', 180) < 150:
            recommendations.append(self.recommendation_db['form']['back_angle'][0])

        # Exercise-type specific tips (handled by pose analyzer)
        return recommendations[:3]  # Return max 3 recommendations

    def _determine_level(self, count: int) -> str:
        if count < 15:
            return 'beginner'
        elif 15 <= count < 30:
            return 'intermediate'
        return 'advanced'