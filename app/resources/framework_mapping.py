"""
Singapore Teaching Practice Framework Mapping

Comprehensive framework for mapping teaching skills to official categories.
"""

# Comprehensive Singapore Teaching Practice Framework
SINGAPORE_TEACHING_FRAMEWORK = {
    # Domain 1: Nurturing the Whole Child
    "1.1": {
        "name": "Establishing Interaction and Rapport",
        "description": "Building positive relationships and connections with students",
        "keywords": ["rapport", "relationship", "connection", "trust", "respect", "interaction", "positive environment"]
    },
    "1.2": {
        "name": "Setting and Maintaining Rules and Routine",
        "description": "Clear expectations and effective classroom management",
        "keywords": ["classroom management", "rules", "routine", "discipline", "expectations", "behavior", "procedures", "structure"]
    },
    "1.3": {
        "name": "Creating a Positive Learning Environment",
        "description": "Establishing safe and conducive learning spaces",
        "keywords": ["learning environment", "safe space", "classroom atmosphere", "physical setup", "seating arrangement"]
    },
    
    # Domain 2: Knowing the Subject Matter
    "2.1": {
        "name": "Demonstrating Subject Matter Knowledge",
        "description": "Deep understanding of content and pedagogy",
        "keywords": ["subject knowledge", "content mastery", "curriculum", "pedagogy", "teaching methods"]
    },
    "2.2": {
        "name": "Staying Current with Subject Developments",
        "description": "Keeping updated with latest in field",
        "keywords": ["professional development", "current trends", "research", "innovations", "updates"]
    },
    
    # Domain 3: Engaging Learners in Learning
    "3.1": {
        "name": "Activating Prior Knowledge",
        "description": "Connecting to existing knowledge and experiences",
        "keywords": ["prior knowledge", "activation", "schema", "background knowledge", "connections", "prerequisite"]
    },
    "3.2": {
        "name": "Motivating Learners for Learning Engagement",
        "description": "Inspiring active participation and interest",
        "keywords": ["engagement", "motivation", "participation", "interest", "active learning", "student involvement", "enthusiasm"]
    },
    "3.3": {
        "name": "Using Questions to Deepen Learning",
        "description": "Strategic questioning for critical thinking",
        "keywords": ["questioning", "questions", "inquiry", "probing", "higher order thinking", "blooms taxonomy", "critical thinking"]
    },
    "3.4": {
        "name": "Facilitating Collaborative Learning",
        "description": "Effective student-to-student interactions",
        "keywords": ["collaboration", "group work", "cooperative learning", "peer learning", "teamwork", "discussion", "group dynamics"]
    },
    "3.5": {
        "name": "Concluding the Lesson",
        "description": "Effective lesson closure and summary",
        "keywords": ["conclusion", "closure", "summary", "wrap up", "consolidation", "reflection", "ending"]
    },
    "3.6": {
        "name": "Using Resources and Technology",
        "description": "Effective use of teaching resources and EdTech",
        "keywords": ["technology", "edtech", "resources", "digital tools", "multimedia", "ICT", "educational technology"]
    },
    
    # Domain 4: Monitoring and Providing Feedback
    "4.1": {
        "name": "Checking for Understanding and Providing Feedback",
        "description": "Assessment and constructive feedback",
        "keywords": ["assessment", "feedback", "understanding", "formative assessment", "checking", "evaluation", "monitoring"]
    },
    "4.2": {
        "name": "Adjusting Teaching Strategies",
        "description": "Adapting instruction based on student needs",
        "keywords": ["differentiation", "adaptation", "flexibility", "adjusting", "responsive teaching", "personalization"]
    },
    
    # Domain 5: Reflective Practice
    "5.1": {
        "name": "Reflecting on Practice",
        "description": "Critical self-evaluation and improvement",
        "keywords": ["reflection", "self-evaluation", "metacognition", "improvement", "professional growth", "self-assessment"]
    },
    "5.2": {
        "name": "Professional Learning and Development",
        "description": "Continuous learning and skill development",
        "keywords": ["professional development", "learning", "training", "workshops", "courses", "skill development"]
    }
}


def get_framework_description(code: str) -> str:
    """Get full description of a framework code"""
    if code in SINGAPORE_TEACHING_FRAMEWORK:
        return SINGAPORE_TEACHING_FRAMEWORK[code]["name"]
    return "Custom Teaching Skill"


def get_all_framework_codes() -> list[str]:
    """Get all framework codes"""
    return list(SINGAPORE_TEACHING_FRAMEWORK.keys())


def search_framework_by_keywords(query: str) -> tuple[str, str, float]:
    """
    Search framework by keywords in query
    
    Returns:
        (framework_code, framework_name, confidence_score)
    """
    query_lower = query.lower()
    best_match = None
    best_score = 0.0
    
    for code, data in SINGAPORE_TEACHING_FRAMEWORK.items():
        # Check if any keywords match
        matches = sum(1 for keyword in data["keywords"] if keyword in query_lower)
        
        if matches > 0:
            # Calculate confidence based on number of keyword matches
            score = min(matches / len(data["keywords"]), 1.0)
            
            if score > best_score:
                best_score = score
                best_match = (code, data["name"], score)
    
    if best_match:
        return best_match
    
    # No match found
    return ("CUSTOM", "Custom Teaching Skill", 0.0)


def map_skill_to_framework(skill: str) -> dict:
    """
    Map an extracted skill to Singapore Teaching Framework
    
    Args:
        skill: The extracted skill/competency (e.g., "questioning techniques")
    
    Returns:
        {
            "framework_code": "3.3",
            "framework_name": "Using Questions to Deepen Learning",
            "confidence": 0.85,
            "is_custom": False
        }
    """
    code, name, confidence = search_framework_by_keywords(skill)
    
    return {
        "framework_code": code,
        "framework_name": name,
        "confidence": confidence,
        "is_custom": code == "CUSTOM"
    }