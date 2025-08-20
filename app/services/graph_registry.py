"""
Graph Registry Service for Teaching Analytics Chatbot

Defines available graph types and their metadata for intent analysis
"""

AVAILABLE_GRAPHS = {
    "teaching_area_distribution": {
        "name": "Teaching Area Distribution",
        "description": "Bar chart showing distribution of teaching activities across Singapore Teaching Practice areas (1.1, 3.4, etc.)",
        "type": "grouped_bar_chart",
        "data_requirements": ["teaching_areas", "area_distribution"],
        "best_for": "Comparing teaching focus across different areas, identifying strengths and areas for improvement",
        "supports_lesson_filter": True,
        "supports_area_filter": True
    },
    "total_distribution": {
        "name": "Total Distribution Overview", 
        "description": "Aggregated bar chart showing total teaching area distribution across all selected lessons",
        "type": "bar_chart",
        "data_requirements": ["teaching_areas", "aggregated_stats"],
        "best_for": "Overall teaching pattern analysis, lesson comparison, identifying dominant teaching approaches",
        "supports_lesson_filter": True,
        "supports_area_filter": True
    },
    "utterance_timeline": {
        "name": "Utterance Timeline",
        "description": "Line chart showing teaching area patterns across lesson progression",
        "type": "line_chart", 
        "data_requirements": ["sequence_order", "teaching_areas", "lesson_progression"],
        "best_for": "Understanding how teaching focus changes throughout lessons, identifying lesson structure patterns",
        "supports_lesson_filter": True,
        "supports_area_filter": True
    },
    "area_distribution_time": {
        "name": "Area Distribution Over Time",
        "description": "Bar chart showing teaching area distribution across time intervals (chunks)",
        "type": "bar_chart",
        "data_requirements": ["chunks", "time_intervals", "area_distribution"],
        "best_for": "Analyzing teaching patterns over specific time periods, understanding lesson flow",
        "supports_lesson_filter": True,
        "supports_area_filter": True
    },
    "wpm_trend": {
        "name": "Words Per Minute Trend",
        "description": "Area chart showing speaking pace (words per minute) over lesson duration",
        "type": "area_chart",
        "data_requirements": ["word_count", "duration_seconds", "sequence_order"],
        "best_for": "Analyzing speaking pace, identifying fast/slow sections, understanding lesson rhythm",
        "supports_lesson_filter": True,
        "supports_area_filter": False  # WPM is about speaking pace, not teaching areas
    }
}

# Teaching area mapping for natural language to area codes
TEACHING_AREA_MAPPING = {
    "interaction": "1.1",
    "rapport": "1.1",
    "rules": "1.2",
    "routine": "1.2",
    "prior knowledge": "3.1",
    "motivation": "3.2",
    "engagement": "3.2",
    "questioning": "3.3",
    "questions": "3.3",
    "collaboration": "3.4",
    "collaborative": "3.4",
    "conclusion": "3.5",
    "wrap up": "3.5",
    "understanding": "4.1",
    "feedback": "4.1",
    "checking": "4.1"
}

def get_graph_by_name(graph_name: str):
    """Get graph metadata by name"""
    return AVAILABLE_GRAPHS.get(graph_name)

def get_all_graphs():
    """Get all available graphs"""
    return AVAILABLE_GRAPHS

def get_graphs_for_intent_analysis():
    """Get formatted graph descriptions for intent analysis prompt"""
    descriptions = []
    for key, graph in AVAILABLE_GRAPHS.items():
        lesson_support = "✓" if graph["supports_lesson_filter"] else "✗"
        area_support = "✓" if graph["supports_area_filter"] else "✗"
        descriptions.append(f"- {key}: {graph['description']} (Lesson Filter: {lesson_support}, Area Filter: {area_support})")
    return "\n".join(descriptions)

def validate_graph_type(graph_type: str) -> bool:
    """Validate if a graph type exists"""
    return graph_type in AVAILABLE_GRAPHS

def validate_lesson_filter(graph_type: str, lesson_filter: list) -> bool:
    """Validate if lesson filtering is supported for the given graph type"""
    if not lesson_filter:  # Empty filter is always valid (default behavior)
        return True
    return AVAILABLE_GRAPHS.get(graph_type, {}).get("supports_lesson_filter", False)

def validate_area_filter(graph_type: str, area_filter: list) -> bool:
    """Validate if area filtering is supported for the given graph type"""
    if not area_filter:  # Empty filter is always valid (default behavior)
        return True
    return AVAILABLE_GRAPHS.get(graph_type, {}).get("supports_area_filter", False)

def map_natural_language_to_area_codes(natural_language: str) -> list:
    """Map natural language descriptions to teaching area codes"""
    if not natural_language:
        return []
    
    natural_language = natural_language.lower()
    mapped_codes = []
    
    for key, code in TEACHING_AREA_MAPPING.items():
        if key in natural_language:
            mapped_codes.append(code)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_codes = []
    for code in mapped_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)
    
    return unique_codes

def get_available_area_codes() -> list:
    """Get list of all available teaching area codes"""
    return ["1.1", "1.2", "3.1", "3.2", "3.3", "3.4", "3.5", "4.1"]
