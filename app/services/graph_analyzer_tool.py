from typing import List, Dict, Any, Optional
from app.services.graph_registry import (
    AVAILABLE_GRAPHS, 
    validate_graph_type,
    map_natural_language_to_area_codes,
    get_available_area_codes
)

class GraphAnalyzerTool:
    """
    Tool for analyzing graph visualization needs and recommending appropriate graphs
    Used by IntentAnalyzer to determine if graphs are needed and which types to recommend
    """
    
    def __init__(self):
        self.name = "graph_analyzer"
        self.description = "Analyzes user queries for visualization needs and recommends graphs"
    
    def analyze_graph_need(self, query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze if query needs graphs and recommend appropriate types
        
        Args:
            query: User's query string
            conversation_history: Previous conversation for context
            
        Returns:
            Dict containing:
            - needs_graph: boolean
            - graph_types: List[Dict] with type and reason
            - suggested_areas: List[str] area codes mentioned in query
            - query_analysis: Dict with detected patterns
        """
        
        # Step 1: Detect if visualization is needed
        needs_graph = self._detect_visualization_need(query)
        
        if not needs_graph:
            return {
                "needs_graph": False,
                "graph_types": [],
                "suggested_areas": [],
                "query_analysis": {"visualization_indicators": [], "focus_areas": []}
            }
        
        # Step 2: Analyze query patterns
        query_analysis = self._analyze_query_patterns(query)
        
        # Step 3: Extract teaching areas mentioned
        suggested_areas = self._extract_teaching_areas(query)
        
        # Step 4: Recommend graph types based on analysis
        graph_types = self._recommend_graph_types(query, query_analysis, suggested_areas)
        
        return {
            "needs_graph": True,
            "graph_types": graph_types,
            "suggested_areas": suggested_areas,
            "query_analysis": query_analysis
        }
    
    def get_graph_descriptions_for_prompt(self) -> str:
        """Get formatted graph descriptions for LLM prompt"""
        descriptions = []
        for key, graph in AVAILABLE_GRAPHS.items():
            lesson_support = "✓" if graph["supports_lesson_filter"] else "✗"
            area_support = "✓" if graph["supports_area_filter"] else "✗"
            descriptions.append(
                f"- {key}: {graph['description']} "
                f"(Lesson Filter: {lesson_support}, Area Filter: {area_support})"
            )
        return "\n".join(descriptions)
    
    def _detect_visualization_need(self, query: str) -> bool:
        """Detect if query needs visualization"""
        query_lower = query.lower()
        
        visualization_indicators = [
            "show me a chart", "visualize", "graph", "plot", "draw",
            "compare", "trend", "pattern", "distribution", "overview",
            "how often", "frequency", "over time", "across lessons",
            "breakdown", "proportion", "percentage", "statistics",
            "see the data", "visual representation", "chart of"
        ]
        
        return any(indicator in query_lower for indicator in visualization_indicators)
    
    def _analyze_query_patterns(self, query: str) -> Dict[str, Any]:
        """Analyze query for specific patterns that suggest graph types"""
        query_lower = query.lower()
        
        patterns = {
            "timeline_focus": any(word in query_lower for word in 
                ["timeline", "over time", "progression", "throughout", "during the lesson"]),
            "comparison_focus": any(word in query_lower for word in 
                ["compare", "comparison", "vs", "versus", "between lessons"]),
            "distribution_focus": any(word in query_lower for word in 
                ["distribution", "breakdown", "proportion", "how much", "frequency"]),
            "pace_focus": any(word in query_lower for word in 
                ["pace", "speed", "words per minute", "wpm", "speaking rate"]),
            "time_intervals": any(word in query_lower for word in 
                ["chunks", "intervals", "periods", "segments"]),
            "comprehensive": any(word in query_lower for word in 
                ["overview", "comprehensive", "everything", "all aspects"])
        }
        
        # Detect visualization indicators found
        visualization_indicators = []
        all_indicators = [
            "show me a chart", "visualize", "graph", "plot", "draw",
            "compare", "trend", "pattern", "distribution", "overview",
            "how often", "frequency", "over time", "across lessons",
            "breakdown", "proportion", "percentage", "statistics"
        ]
        
        for indicator in all_indicators:
            if indicator in query_lower:
                visualization_indicators.append(indicator)
        
        return {
            "patterns": patterns,
            "visualization_indicators": visualization_indicators
        }
    
    def _extract_teaching_areas(self, query: str) -> List[str]:
        """Extract teaching area codes mentioned in query"""
        return map_natural_language_to_area_codes(query)
    
    def _recommend_graph_types(self, query: str, analysis: Dict[str, Any], areas: List[str]) -> List[Dict[str, str]]:
        """Recommend appropriate graph types based on analysis"""
        patterns = analysis.get("patterns", {})
        recommended_graphs = []
        
        # Timeline/progression analysis
        if patterns.get("timeline_focus"):
            recommended_graphs.append({
                "type": "utterance_timeline",
                "reason": "Shows teaching patterns over lesson progression"
            })
        
        # Speaking pace analysis
        if patterns.get("pace_focus"):
            recommended_graphs.append({
                "type": "wpm_trend",
                "reason": "Shows speaking pace trends over time"
            })
        
        # Time-based distribution
        if patterns.get("time_intervals"):
            recommended_graphs.append({
                "type": "area_distribution_time",
                "reason": "Shows teaching area distribution across time periods"
            })
        
        # Comparison focus (lesson comparison)
        if patterns.get("comparison_focus"):
            recommended_graphs.append({
                "type": "teaching_area_distribution",
                "reason": "Shows distribution comparison across lessons"
            })
        
        # Distribution/breakdown focus
        if patterns.get("distribution_focus"):
            if patterns.get("comparison_focus"):
                recommended_graphs.append({
                    "type": "teaching_area_distribution",
                    "reason": "Shows distribution comparison across lessons"
                })
            else:
                recommended_graphs.append({
                    "type": "total_distribution",
                    "reason": "Shows overall teaching area distribution"
                })
        
        # Comprehensive overview
        if patterns.get("comprehensive"):
            recommended_graphs.extend([
                {
                    "type": "total_distribution",
                    "reason": "Shows overall teaching patterns"
                },
                {
                    "type": "utterance_timeline",
                    "reason": "Shows patterns over time"
                }
            ])
        
        # Default fallback if no specific patterns matched but visualization is needed
        if not recommended_graphs:
            recommended_graphs.append({
                "type": "total_distribution",
                "reason": "General teaching area distribution"
            })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_graphs = []
        for graph in recommended_graphs:
            graph_key = (graph["type"], graph["reason"])
            if graph_key not in seen:
                seen.add(graph_key)
                unique_graphs.append(graph)
        
        return unique_graphs

# Global instance
graph_analyzer_tool = GraphAnalyzerTool()
