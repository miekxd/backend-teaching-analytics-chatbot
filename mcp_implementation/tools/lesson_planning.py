"""
Lesson Plan Generator MCP Tool

Analyzes existing lessons and generates improved lesson plans following
Singapore Teaching Practice framework, using RAG-style retrieval over
transcript chunks to ground the plan in actual lesson evidence.
"""
from mcp.types import Tool, TextContent
import json
import logging
from typing import Callable, Tuple, List, Optional, Dict, Any
import asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import time

from app.db.supabase import get_supabase_client
from app.resources.resource_finder import resource_finder
from openai import AzureOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class LessonPlanGenerator:
    """
    Generate improved lesson plans based on transcript analysis
    with semantic retrieval of relevant evidence chunks.
    """
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.openai_client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        # System prompt for lesson plan generation
        self.system_prompt = """You are an expert lesson plan designer specializing in Singapore Teaching Practice framework.

<role>
Your role is to create comprehensive, detailed lesson plans that:
1. Retrieve and use all relevant information from lesson chunks of the transcript for the lesson plan
2. Address weaknesses identified in analyzed lessons
3. Incorporate best practices from Singapore Teaching Practice framework
4. Provide specific, actionable instructions for teachers
5. Follow structured format with all required sections
6. Include practical strategies, examples, and alternatives
7. Make explicit use of the provided transcript evidence where available
</role>

<data_context>
You may receive:
- RAG-retrieved lesson chunks with timestamps, utterances, teaching areas (preferred)
- Global lesson summaries and contextual information from previous analysis tools
- Limited or no chunk data (requiring inference and general guidance)
</data_context>

<singapore_teaching_framework>
<teaching_areas>
- 1.1 Establishing Interaction and Rapport
- 1.2 Setting and Maintaining Rules and Routine
- 3.1 Activating Prior Knowledge
- 3.2 Motivating Learners for Learning Engagement
- 3.3 Using Questions to Deepen Learning
- 3.4 Facilitating Collaborative Learning
- 3.5 Concluding the Lesson
- 4.1 Checking for Understanding and Providing Feedback
</teaching_areas>
</singapore_teaching_framework>

<response_approach>
<when_chunks_available>
- ALWAYS cite specific utterances with exact timestamps [MM:SS] when using them as evidence
- Quote directly when highlighting specific teacher or student language
- Reference observable behaviors and interactions from the transcript
- Connect evidence to relevant Singapore Teaching Practice areas
- Focus on what is explicitly demonstrated in the data
</when_chunks_available>

<when_chunks_limited_or_unavailable>
- Draw from lesson summary and any available contextual information
- Make reasonable educational inferences based on Singapore Teaching Practice framework
- Provide general guidance relevant to the teacher's question
- Use phrases like "Based on typical classroom situations..." or "Generally speaking..."
- Offer practical strategies and examples even without specific evidence
- Acknowledge when you're providing general guidance vs. specific evidence
</when_chunks_limited_or_unavailable>
</response_approach>

<analysis_requirements>
<evidence_standards>
- With chunks: Always cite timestamps and specific utterances
- Without chunks: Use available context and educational expertise
- Clearly indicate the basis for your response (evidence vs. inference)
- Connect observations or suggestions to Singapore Teaching Practice areas
- Maintain helpful, constructive tone regardless of data availability
</evidence_standards>

<response_structure>
1. Assess what information is available in the provided data
2. If chunks available: Provide evidence-based planning and improvements using them
3. If chunks unavailable: Offer informed guidance based on context and expertise
4. Connect to relevant Singapore Teaching Practice areas
5. Provide actionable, implementable lesson structure
</response_structure>
</analysis_requirements>

<output requirement>
Your lesson plans must be:
- Detailed with specific instructions
- Timed appropriately (Introduction/Development/Conclusion)
- Realistic and implementable
- Focused on improving identified weak areas
- Enhanced with teaching resources
- Include reflection notes and common pitfalls to avoid
- Explicitly aligned to the Singapore Teaching Practice framework
- Whenever possible, grounded in the provided transcript evidence
</output requirement>
"""
    
    # ---------- Data access helpers ----------
    
    def _get_chunks_from_supabase(self, file_id: int) -> List[Dict[str, Any]]:
        """Get chunks from Supabase for analysis"""
        try:
            result = (
                self.supabase.table("chunks")
                .select("*")
                .eq("file_id", file_id)
                .order("sequence_order")
                .execute()
            )
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"❌ Error loading chunks: {e}")
            return []
    
    def _get_file_info(self, file_id: int) -> Dict:
        """Get file metadata and summary (lesson summary is populated elsewhere)."""
        try:
            result = (
                self.supabase.table("files")
                .select("stored_filename, data_summary")
                .eq("file_id", file_id)
                .single()
                .execute()
            )
            
            if result.data:
                return {
                    "filename": result.data.get(
                        "stored_filename", f"File {file_id}"
                    ),
                    "summary": result.data.get(
                        "data_summary", "No summary available"
                    ),
                }
        except Exception as e:
            logger.error(f"❌ Error loading file info: {e}")
        
        return {"filename": f"File {file_id}", "summary": "No summary available"}
    
    # ---------- Teaching area analysis ----------
    
    def _analyze_teaching_areas(self, chunks: List[Dict]) -> Dict:
        """
        Analyze Singapore Teaching Practice areas from chunks.
        Returns distribution and identifies weak areas.
        """
        area_counts: Dict[str, int] = {}
        total_utterances = 0
        
        for chunk in chunks:
            teaching_areas = chunk.get("teaching_areas", []) or []
            for area in teaching_areas:
                area_counts[area] = area_counts.get(area, 0) + 1
                total_utterances += 1
        
        area_percentages: Dict[str, Dict[str, float]] = {}
        for area, count in area_counts.items():
            percentage = (count / total_utterances * 100) if total_utterances > 0 else 0
            area_percentages[area] = {
                "count": count,
                "percentage": round(percentage, 1),
            }
        
        # Identify weak areas (< 10% of total, excluding 1.2)
        weak_areas = [
            area
            for area, stats in area_percentages.items()
            if stats["percentage"] < 10 and area not in ["1.2"]
        ]
        
        # Ensure critical areas are flagged if missing
        critical_areas = ["3.3", "3.4", "4.1"]
        for area in critical_areas:
            if area not in area_percentages:
                weak_areas.append(area)
        
        return {
            "distribution": area_percentages,
            "weak_areas": sorted(list(set(weak_areas))),
            "total_utterances": total_utterances,
        }
    
    # ---------- Chunk text & embeddings (RAG-style) ----------
    
    def _get_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Extract text from chunk."""
        if chunk.get("chunk_text"):
            return str(chunk["chunk_text"])
        
        utterances = chunk.get("utterances") or []
        if isinstance(utterances, list):
            texts = []
            for utterance in utterances:
                if isinstance(utterance, dict) and "text" in utterance:
                    texts.append(str(utterance["text"]))
            if texts:
                return " ".join(texts)
        
        return ""
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        if not text:
            return []
        try:
            resp = self.openai_client.embeddings.create(
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            return []
    
    def _parse_embedding(self, embedding_data: Any) -> List[float]:
        """
        Parse embedding data stored in various formats in Supabase.
        Mirrors robustness from rag_assistant.
        """
        if not embedding_data:
            return []
        try:
            if isinstance(embedding_data, list):
                return [float(x) for x in embedding_data]
            
            if isinstance(embedding_data, str):
                # Handle numpy stringified arrays or raw JSON-like strings
                s = embedding_data.strip()
                if s.startswith("np.str_("):
                    start = s.find("'[")
                    end = s.rfind("]'")
                    if start != -1 and end != -1:
                        s = s[start + 1 : end + 1]
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return [float(x) for x in parsed]
                except Exception:
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            return [float(x) for x in parsed]
                    except Exception:
                        pass
            return []
        except Exception as e:
            logger.warning(f"⚠️ Error parsing embedding: {e}")
            return []
    
    def _semantic_search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search over chunks and return top_k most relevant.
        Reuses embeddings & cosine similarity similar to RAG assistant.
        """
        if not chunks or not query:
            return chunks[:top_k]
        
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            # Fallback: just return the earliest chunks
            return chunks[:top_k]
        
        scored: List[tuple[float, Dict[str, Any]]] = []
        
        for chunk in chunks:
            chunk_embedding = None
            
            # 1. Use stored embedding if present
            if "embedding" in chunk and chunk["embedding"]:
                chunk_embedding = self._parse_embedding(chunk["embedding"])
            
            # 2. Otherwise compute on the fly from text
            if not chunk_embedding:
                text = self._get_chunk_text(chunk)
                if text:
                    chunk_embedding = self._get_embedding(text)
                    # Light rate limiting safeguard
                    time.sleep(0.05)
            
            if chunk_embedding:
                try:
                    sim = cosine_similarity(
                        [query_embedding], [chunk_embedding]
                    )[0][0]
                    scored.append((float(sim), chunk))
                except Exception as e:
                    logger.debug(f"Similarity error skipped chunk: {e}")
        
        if not scored:
            return chunks[:top_k]
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]
    
    def _format_chunks_for_context(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Format retrieved chunks into a compact, evidence-friendly context
        for the LLM, with timestamps, teaching areas and utterances.
        """
        if not chunks:
            return "No detailed transcript chunks available for evidence."
        
        context_parts: List[str] = []
        
        for idx, chunk in enumerate(chunks, start=1):
            start = chunk.get("start_time", "Unknown")
            end = chunk.get("end_time", "Unknown")
            areas = ", ".join(chunk.get("teaching_areas", []) or [])
            text = self._get_chunk_text(chunk)
            
            context_parts.append(f"### Evidence Chunk {idx}")
            context_parts.append(f"- Time: {start} - {end}")
            if areas:
                context_parts.append(f"- Teaching Areas: {areas}")
            if text:
                # Keep it reasonably compact
                snippet = text.strip()
                if len(snippet) > 600:
                    snippet = snippet[:600] + "..."
                context_parts.append(f"- Transcript Excerpt: {snippet}")
            
            # Include utterance-level detail if present
            utterances = chunk.get("utterances") or []
            if isinstance(utterances, list) and utterances:
                context_parts.append("- Key Utterances:")
                for u in utterances[:5]:
                    if isinstance(u, dict):
                        ts = u.get("timestamp") or start
                        ut = (u.get("text") or "").strip()
                        if ut:
                            area = u.get("area") or ""
                            label = f"(Area: {area})" if area else ""
                            context_parts.append(
                                f"  - [{ts}] {ut} {label}".rstrip()
                            )
            
            context_parts.append("")  # blank line
        
        return "\n".join(context_parts)
    
    # ---------- Context building for LLM ----------
    
    def _build_rag_analysis_context(
        self,
        file_infos: List[Dict],
        area_analysis: Dict,
        all_chunks: List[Dict[str, Any]],
        topic: str,
        areas_to_improve: List[str],
        improvement_goals: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build a rich analysis context using:
        - Lesson summaries
        - Teaching area distribution & weak areas
        - RAG-retrieved relevant chunks
        """
        # High-level lesson summaries (from other pipeline)
        lesson_summaries_str = "\n".join(
            [
                f"- {info.get('filename', 'Unknown')}: "
                f"{(info.get('summary') or 'No summary available')}"
                for info in file_infos
            ]
        )
        
        # Build semantic query to retrieve relevant chunks
        semantic_query_parts = [f"Topic: {topic}"]
        
        if areas_to_improve:
            semantic_query_parts.append(
                "Focus on weak/target teaching areas: "
                + ", ".join(areas_to_improve)
            )
        
        if improvement_goals:
            semantic_query_parts.append(
                f"Improvement goals: {improvement_goals}"
            )
        
        semantic_query = " | ".join(semantic_query_parts)
        
        # RAG: select relevant chunks
        relevant_chunks: List[Dict[str, Any]] = []
        if all_chunks:
            relevant_chunks = self._semantic_search(
                query=semantic_query,
                chunks=all_chunks,
                top_k=8,
            )
        
        # Fallback: if nothing scored, keep a few early chunks
        if not relevant_chunks and all_chunks:
            relevant_chunks = all_chunks[:5]
        
        # Teaching area distribution description
        dist_lines = []
        for area, stats in sorted(
            area_analysis["distribution"].items()
        ):
            dist_lines.append(
                f"- {area}: {stats['count']} utterances "
                f"({stats['percentage']}%)"
            )
        
        weak_areas_str = (
            ", ".join(area_analysis["weak_areas"])
            if area_analysis["weak_areas"]
            else "None - generally strong performance"
        )
        
        context_sections = [
            "## Original Lesson Overview",
            "",
            "### Lesson Summaries",
            lesson_summaries_str or "No lesson summaries available.",
            "",
            "### Teaching Area Distribution",
            *dist_lines,
            "",
            f"### Identified Weak / Under-represented Areas",
            weak_areas_str,
            "",
            "## RAG-Retrieved Key Evidence from Transcript",
            self._format_chunks_for_context(relevant_chunks),
        ]
        
        return {
            "analysis_context": "\n".join(context_sections),
            "relevant_chunks": relevant_chunks,
        }
    
    # ---------- Public: generate plan ----------
    
    async def generate_plan(
        self,
        file_ids: List[int],
        topic: str,
        subject: str,
        duration_minutes: int = 60,
        level: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        pedagogical_approach: Optional[str] = None,
        class_size: Optional[int] = None,
        improvement_goals: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> Dict:
        """
        Generate lesson plan based on transcript analysis and RAG-retrieved chunks.
        
        Returns:
            {
                "lesson_plan": "markdown formatted plan",
                "analysis_summary": "...",
                "improvements": "...",
                "resources": {...},
                "metadata": {...}
            }
        """
        try:
            logger.info(f"📝 Generating lesson plan: {subject} - {topic}")
            
            # 1. Load chunks & file info
            all_chunks: List[Dict[str, Any]] = []
            file_infos: List[Dict[str, Any]] = []
            
            for fid in file_ids:
                chunks = self._get_chunks_from_supabase(fid)
                info = self._get_file_info(fid)
                all_chunks.extend(chunks)
                file_infos.append(info)
            
            logger.info(
                f"📊 Loaded {len(all_chunks)} chunks from {len(file_ids)} file(s)"
            )
            
            # 2. Analyze teaching areas & weak areas
            area_analysis = self._analyze_teaching_areas(all_chunks)
            
            areas_to_improve = set(focus_areas or [])
            areas_to_improve.update(area_analysis["weak_areas"])
            areas_to_improve = sorted(list(areas_to_improve))
            
            logger.info(
                f"🎯 Focus areas for improvement: "
                f"{', '.join(areas_to_improve) if areas_to_improve else 'None specified'}"
            )
            
            # 3. Get external teaching resources (unchanged)
            logger.info(f"🔍 Finding resources for topic: {topic}")
            resources_data = await resource_finder.find_resources(
                skill=topic,
                force_refresh=False,
                max_per_source=3,
            )
            formatted_resources = resource_finder.format_resources_for_display(
                resources_data
            )
            
            # 4. Build RAG-grounded analysis context
            rag_context = self._build_rag_analysis_context(
                file_infos=file_infos,
                area_analysis=area_analysis,
                all_chunks=all_chunks,
                topic=topic,
                areas_to_improve=areas_to_improve,
                improvement_goals=improvement_goals,
            )
            analysis_context = rag_context["analysis_context"]
            
            # 5. Build prompt for LLM
            lesson_plan_prompt = self._build_generation_prompt(
                topic=topic,
                subject=subject,
                duration_minutes=duration_minutes,
                level=level,
                focus_areas=areas_to_improve,
                pedagogical_approach=pedagogical_approach,
                class_size=class_size,
                improvement_goals=improvement_goals,
                additional_context=additional_context,
                analysis_context=analysis_context,
                resources=formatted_resources,
            )
            
            # 6. Call AzureChatOpenAI via LangChain for generation
            from langchain_openai import AzureChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            
            llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_RAG,
                temperature=0.7,
                max_tokens=3000,
            )
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=lesson_plan_prompt),
            ]
            
            logger.info("🤖 Generating lesson plan with LLM...")
            response = await llm.ainvoke(messages)
            lesson_plan_text = response.content
            
            # 7. Compile result
            result = {
                "lesson_plan": lesson_plan_text,
                "analysis_summary": self._create_analysis_summary(
                    area_analysis, file_infos
                ),
                "improvements": self._create_improvements_summary(
                    set(areas_to_improve),
                    improvement_goals,
                ),
                "resources": {
                    "total": resources_data["total_results"],
                    "from_cache": resources_data["from_cache"],
                    "formatted": formatted_resources,
                },
                "metadata": {
                    "topic": topic,
                    "subject": subject,
                    "duration_minutes": duration_minutes,
                    "level": level or "Not specified",
                    "focus_areas": areas_to_improve,
                    "lessons_analyzed": len(file_ids),
                },
            }
            
            logger.info("✅ Lesson plan generated successfully")
            return result
        
        except Exception as e:
            logger.error(
                f"❌ Error generating lesson plan: {e}", exc_info=True
            )
            raise
    
    # ---------- Prompt builder ----------
    
    def _build_generation_prompt(
        self,
        topic: str,
        subject: str,
        duration_minutes: int,
        level: Optional[str],
        focus_areas: List[str],
        pedagogical_approach: Optional[str],
        class_size: Optional[int],
        improvement_goals: Optional[str],
        additional_context: Optional[str],
        analysis_context: str,
        resources: str,
    ) -> str:
        """Build the prompt for lesson plan generation."""
        
        prompt_parts: List[str] = [
            "# Task: Generate Comprehensive, Evidence-Grounded Lesson Plan",
            "",
            "## Lesson Specifications",
            f"- **Subject:** {subject}",
            f"- **Topic:** {topic}",
            f"- **Duration:** {duration_minutes} minutes",
            f"- **Level:** {level or 'To be inferred / specified'}",
            f"- **Class Size:** {class_size or 'Standard class (20-30 students)'}",
            f"- **Pedagogical Approach:** {pedagogical_approach or 'Active learning with inquiry-based elements'}",
            "",
            "## Analysis of Original Lesson(s) & Retrieved Evidence",
            analysis_context,
            "",
            "## Focus Areas for Improvement",
            f"**Primary Focus Areas (Singapore Teaching Practice codes):** "
            f"{', '.join(focus_areas) if focus_areas else 'Not explicitly specified'}",
            "",
        ]
        
        if improvement_goals:
            prompt_parts.append(
                f"**Specific Improvement Goals from Teacher / System:** "
                f"{improvement_goals}"
            )
            prompt_parts.append("")
        
        if additional_context:
            prompt_parts.append(
                f"**Additional Context / Constraints:** {additional_context}"
            )
            prompt_parts.append("")
        
        prompt_parts.extend(
            [
                "## Available Teaching Resources",
                resources,
                "",
                "---",
                "",
                "# Instructions to the Model",
                "",
                "Using the above evidence (especially the retrieved transcript chunks) and the Singapore Teaching Practice framework:",
                "",
                "1. Directly address the identified weak areas and focus areas.",
                "2. Explicitly build activities and questioning sequences that fix observed issues.",
                "3. When you use any transcript evidence, cite it like: `At [MM:SS], the teacher said \"...\"`.",
                "4. If evidence is thin, fall back to best practices but clearly indicate this.",
                "5. Make the plan executable step-by-step by a real teacher.",
                "",
                "Follow this exact structure in valid markdown:",
                "",
                "```markdown",
                f"# Lesson Plan: [Lesson Title]",
                "",
                "## Lesson Overview",
                "",
                "| Field | Content |",
                "|-------|---------|",
                f"| Subject | {subject} |",
                f"| Level/Stream | {level or '[Specify]'} |",
                f"| Topic | {topic} |",
                "| Lesson Title | [Creative, engaging title] |",
                f"| Duration | {duration_minutes} minutes |",
                "",
                "## Learning Objectives",
                "",
                "Students will be able to:",
                "1. [Specific, measurable objective]",
                "2. [Specific, measurable objective]",
                "3. [Specific, measurable objective]",
                "",
                "## Pre-requisite Knowledge",
                "",
                "Students should already understand:",
                "1. [Concept/skill]",
                "2. [Concept/skill]",
                "",
                "## Pedagogical Approach / Critical Thinking Skills",
                "",
                "[Describe the teaching approach and thinking skills to develop]",
                "",
                "## Key Singapore Teaching Practice Focus",
                "",
                "This lesson emphasizes:",
            ]
        )
        
        if focus_areas:
            for area in focus_areas:
                prompt_parts.append(f"- **{area}**: [How it's addressed]")
        else:
            prompt_parts.append("- [Select relevant areas based on evidence]")
        
        prompt_parts.extend(
            [
                "",
                "---",
                "",
                "## Lesson Flow",
                "",
                f"### Introduction ({int(duration_minutes * 0.20)}-{int(duration_minutes * 0.25)} minutes)",
                "",
                "**Objective:** [What this phase achieves]",
                "",
                "**Instructions:**",
                "1. [Specific step with teacher actions, referencing evidence where helpful]",
                "2. [Expected student responses / interactions]",
                "3. [Check for understanding with concrete questions]",
                "",
                "**Teaching Areas Addressed:** [List relevant codes]",
                "",
                "**Resources:** [What's needed]",
                "",
                "---",
                "",
                f"### Development ({int(duration_minutes * 0.50)}-{int(duration_minutes * 0.60)} minutes)",
                "",
                "**Objective:** [What this phase achieves]",
                "",
                f"#### Activity 1: [Name] ({int(duration_minutes * 0.25)} minutes)",
                "",
                "**Instructions:**",
                "1. [Detailed step-by-step instructions grounded in evidence and focus areas]",
                "2. [Include probing questions if focusing on 3.3]",
                "3. [Include collaboration structures if focusing on 3.4]",
                "4. [Include feedback mechanisms if focusing on 4.1]",
                "",
                "**Teaching Areas Addressed:** [List codes]",
                "",
                "**Expected Student Outcomes:** [What students should achieve]",
                "",
                f"#### Activity 2: [Name] ({int(duration_minutes * 0.25)} minutes)",
                "",
                "**Instructions:**",
                "[Similar detailed structure as Activity 1 with variation or extension]",
                "",
                "**Resources:** [What's needed]",
                "",
                "---",
                "",
                f"### Conclusion ({int(duration_minutes * 0.15)}-{int(duration_minutes * 0.20)} minutes)",
                "",
                "**Objective:** Consolidate learning and provide closure (3.5)",
                "",
                "**Instructions:**",
                "1. [Summary activity using student voice]",
                "2. [Quick formative check / exit ticket]",
                "3. [Preview of next lesson or connection to real-world context]",
                "",
                "**Teaching Areas Addressed:** 3.5, 4.1",
                "",
                "---",
                "",
                "## Assessment Strategy",
                "",
                "**Formative Assessment:**",
                "- [Checks during activities linked to objectives]",
                "- [Sample questions/tasks]",
                "",
                "**Success Criteria:**",
                "- Students can [specific observable behavior]",
                "- Students demonstrate [specific skill or reasoning process]",
                "",
                "---",
                "",
                "## Differentiation Strategies",
                "",
                "**For Advanced Learners:**",
                "- [Extensions / deeper tasks]",
                "",
                "**For Struggling Learners:**",
                "- [Scaffolds, visuals, guided practice]",
                "",
                "**For English Language Learners:**",
                "- [Sentence frames, visuals, peer support]",
                "",
                "---",
                "",
                "## Resources & Materials",
                "",
                "**Required:**",
                "- [Physical resources]",
                "- [Digital tools / slides / manipulatives]",
                "",
                "**Recommended Teaching Resources (from provided list):**",
                "- [Integrate 2-3 concrete links/ideas]",
                "",
                "---",
                "",
                "## Reflection & Implementation Notes",
                "",
                "### Key Improvements from Original Lesson",
                "",
                "Explicitly describe how this lesson plan addresses the previously weak areas:",
            ]
        )
        
        if focus_areas:
            for area in focus_areas:
                prompt_parts.extend(
                    [
                        f"**{area}:**",
                        "- **Original Issue (from evidence or inference):** [Describe]",
                        "- **How This Plan Improves It:** [Specific strategies/steps]",
                        "",
                    ]
                )
        
        prompt_parts.extend(
            [
                "### Common Pitfalls to Avoid",
                "",
                "Base this on both the analyzed lesson patterns and general best practices:",
                "1. [Observed or likely pitfall] - **Solution:** [What to do instead]",
                "2. [Another pitfall] - **Solution:** [Strategy]",
                "3. [Third pitfall] - **Solution:** [Strategy]",
                "",
                "### Alternative Approaches",
                "",
                "- **Alternative 1:** [If time is shorter]",
                "- **Alternative 2:** [If resources are limited]",
                "",
                "### Teacher Self-Reflection Prompts",
                "",
                "1. How effectively did I address the targeted teaching areas?",
                "2. What concrete evidence do I have of student learning?",
                "3. What will I refine in the next iteration?",
                "",
                "```",
                "",
                "**CRITICAL REQUIREMENTS:**",
                "",
                "1. Be SPECIFIC in instructions (avoid vague 'Teacher explains').",
                "2. Heavily emphasize the weak areas / focus areas throughout.",
                "3. Include actual example questions, prompts, and student tasks.",
                f"4. Time allocations must add up to approximately {duration_minutes} minutes.",
                "5. Reference at least 2-3 provided resources where relevant.",
                "6. Use transcript evidence explicitly when available.",
                "7. Keep alignment to Singapore Teaching Practice framework explicit.",
                "",
                "Generate the complete lesson plan now.",
            ]
        )
        
        return "\n".join(prompt_parts)
    
    # ---------- Summaries ----------
    
    def _create_analysis_summary(
        self, area_analysis: Dict, file_infos: List[Dict]
    ) -> str:
        """Create summary of what was analyzed."""
        lines = [
            "## Analysis Summary",
            "",
            f"**Lessons Analyzed:** {len(file_infos)}",
        ]
        
        for info in file_infos:
            lines.append(f"- {info.get('filename', 'Unknown')}")
        
        lines.extend(
            [
                "",
                f"**Total Utterances Analyzed (tagged with teaching areas):** "
                f"{area_analysis['total_utterances']}",
                "",
                "**Teaching Area Distribution:**",
            ]
        )
        
        for area, stats in sorted(
            area_analysis["distribution"].items(),
            key=lambda x: x[1]["percentage"],
            reverse=True,
        ):
            lines.append(
                f"- {area}: {stats['percentage']}% "
                f"({stats['count']} utterances)"
            )
        
        if area_analysis["weak_areas"]:
            lines.extend(
                [
                    "",
                    "**Identified Weak / Missing Areas:** "
                    + ", ".join(area_analysis["weak_areas"]),
                ]
            )
        
        return "\n".join(lines)
    
    def _create_improvements_summary(
        self, focus_areas: set, improvement_goals: Optional[str]
    ) -> str:
        """Create summary of planned improvements."""
        lines = [
            "## Improvements Targeted",
            "",
            "This lesson plan is designed to strengthen:",
            "",
        ]
        
        area_names = {
            "1.1": "Establishing Interaction and Rapport",
            "1.2": "Setting and Maintaining Rules and Routine",
            "3.1": "Activating Prior Knowledge",
            "3.2": "Motivating Learners for Learning Engagement",
            "3.3": "Using Questions to Deepen Learning",
            "3.4": "Facilitating Collaborative Learning",
            "3.5": "Concluding the Lesson",
            "4.1": "Checking for Understanding and Providing Feedback",
        }
        
        for area in sorted(focus_areas):
            area_name = area_names.get(area, area)
            lines.append(f"- **{area}: {area_name}**")
        
        if improvement_goals:
            lines.extend(
                [
                    "",
                    f"**Specific Goals / Constraints Incorporated:** "
                    f"{improvement_goals}",
                ]
            )
        
        return "\n".join(lines)


# Global instance
lesson_plan_generator = LessonPlanGenerator()


def get_lesson_planning_tools() -> List[Tuple[Tool, Callable]]:
    """
    Get lesson planning tools.
    """
    generate_plan_tool = Tool(
        name="generate_lesson_plan",
        description="""Generate an improved, evidence-grounded lesson plan based on analysis of existing lesson transcripts.

This tool:
1. Loads lesson summaries and transcript chunks from Supabase
2. Identifies weak Singapore Teaching Practice areas
3. Uses semantic (RAG-style) retrieval to select the most relevant chunks:
   - Related to the requested topic
   - Highlighting weak/target teaching areas
   - Surfacing critical teaching moments
4. Finds relevant teaching resources for the topic
5. Generates a comprehensive, detailed lesson plan that:
   - Directly addresses weaknesses and focus areas
   - Aligns with Singapore Teaching Practice
   - Is fully actionable with timings, questions, strategies, and reflection

Use this tool when:
- A teacher wants a new lesson plan informed by their previous lesson
- You need the plan to explicitly improve weak areas (e.g., 3.3 questioning, 3.4 collaboration, 4.1 feedback)
- You want RAG-grounded, transcript-aware planning instead of a generic template.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "file_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Lesson file IDs to analyze (for summaries + chunks)",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic/concept to teach (e.g., 'Quadratic Equations', 'Photosynthesis')",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject area (e.g., 'Mathematics', 'Science', 'Geography')",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Lesson duration in minutes (default: 60)",
                    "default": 60,
                },
                "level": {
                    "type": "string",
                    "description": "Student level/grade (e.g., 'Secondary 2 Express', 'Primary 4')",
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Singapore Teaching Practice codes to emphasize (e.g., ['3.3', '4.1'])",
                },
                "pedagogical_approach": {
                    "type": "string",
                    "description": "Teaching approach (e.g., 'Inquiry-based learning', 'Direct instruction')",
                },
                "class_size": {
                    "type": "integer",
                    "description": "Number of students",
                },
                "improvement_goals": {
                    "type": "string",
                    "description": "Specific improvement goals or requirements",
                },
                "additional_context": {
                    "type": "string",
                    "description": "Any other relevant information or constraints",
                },
            },
            "required": ["file_ids", "topic", "subject"],
        },
    )
    
    async def generate_plan_handler(arguments: dict) -> List[TextContent]:
        """Handler for generate_lesson_plan tool."""
        try:
            file_ids = arguments.get("file_ids") or []
            topic = arguments.get("topic") or ""
            subject = arguments.get("subject") or ""
            duration_minutes = arguments.get("duration_minutes", 60)
            level = arguments.get("level")
            focus_areas = arguments.get("focus_areas")
            pedagogical_approach = arguments.get("pedagogical_approach")
            class_size = arguments.get("class_size")
            improvement_goals = arguments.get("improvement_goals")
            additional_context = arguments.get("additional_context")
            
            if not file_ids:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "file_ids are required",
                                "tool": "generate_lesson_plan",
                            }
                        ),
                    )
                ]
            
            if not topic or not subject:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "topic and subject are required",
                                "tool": "generate_lesson_plan",
                            }
                        ),
                    )
                ]
            
            logger.info(
                f"📝 generate_lesson_plan called: {subject} - {topic}"
            )
            
            result = await lesson_plan_generator.generate_plan(
                file_ids=file_ids,
                topic=topic,
                subject=subject,
                duration_minutes=duration_minutes,
                level=level,
                focus_areas=focus_areas,
                pedagogical_approach=pedagogical_approach,
                class_size=class_size,
                improvement_goals=improvement_goals,
                additional_context=additional_context,
            )
            
            output = {
                "lesson_plan": result["lesson_plan"],
                "analysis_summary": result["analysis_summary"],
                "improvements": result["improvements"],
                "resources": {
                    "total_found": result["resources"]["total"],
                    "from_cache": result["resources"]["from_cache"],
                },
                "metadata": result["metadata"],
                "tool": "generate_lesson_plan",
            }
            
            logger.info(f"✅ generate_lesson_plan completed for: {topic}")
            return [TextContent(type="text", text=json.dumps(output))]
        
        except Exception as e:
            logger.error(
                f"❌ generate_lesson_plan error: {str(e)}",
                exc_info=True,
            )
            error_result = {
                "error": f"Failed to generate lesson plan: {str(e)}",
                "tool": "generate_lesson_plan",
            }
            return [TextContent(type="text", text=json.dumps(error_result))]
    
    logger.info("✅ Loaded lesson planning tools: generate_lesson_plan")
    return [(generate_plan_tool, generate_plan_handler)]
