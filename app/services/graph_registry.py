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
        "best_for": "Comparing teaching focus across different areas, identifying strengths and areas for improvement"
    },
    "total_distribution": {
        "name": "Total Distribution Overview", 
        "description": "Aggregated bar chart showing total teaching area distribution across all selected lessons",
        "type": "bar_chart",
        "data_requirements": ["teaching_areas", "aggregated_stats"],
        "best_for": "Overall teaching pattern analysis, lesson comparison, identifying dominant teaching approaches"
    },
    "utterance_timeline": {
        "name": "Utterance Timeline",
        "description": "Line chart showing teaching area patterns across lesson progression",
        "type": "line_chart", 
        "data_requirements": ["sequence_order", "teaching_areas", "lesson_progression"],
        "best_for": "Understanding how teaching focus changes throughout lessons, identifying lesson structure patterns"
    },
    "area_distribution_time": {
        "name": "Area Distribution Over Time",
        "description": "Bar chart showing teaching area distribution across time intervals (chunks)",
        "type": "bar_chart",
        "data_requirements": ["chunks", "time_intervals", "area_distribution"],
        "best_for": "Analyzing teaching patterns over specific time periods, understanding lesson flow"
    },
    "wpm_trend": {
        "name": "Words Per Minute Trend",
        "description": "Area chart showing speaking pace (words per minute) over lesson duration",
        "type": "area_chart",
        "data_requirements": ["word_count", "duration_seconds", "sequence_order"],
        "best_for": "Analyzing speaking pace, identifying fast/slow sections, understanding lesson rhythm"
    }
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
        descriptions.append(f"- {key}: {graph['description']} (Best for: {graph['best_for']})")
    return "\n".join(descriptions)

def validate_graph_type(graph_type: str) -> bool:
    """Validate if a graph type exists"""
    return graph_type in AVAILABLE_GRAPHS
