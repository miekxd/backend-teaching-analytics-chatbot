"""
Resource Finder Service

Orchestrates resource search across multiple APIs, handles caching,
and formats results for display.
"""
import asyncio
import logging
from typing import List, Dict, Optional
import json

from app.resources.api_clients import (
    ArxivClient,
    SemanticScholarClient,
    YouTubeClient,
    GoogleSearchClient
)
from app.resources.csv_manager import (
    search_resources_by_skill,
    add_resource,
    increment_usage_count
)
from app.resources.framework_mapping import map_skill_to_framework

logger = logging.getLogger(__name__)


class ResourceFinder:
    """
    Main service for finding and managing teaching resources
    """
    
    def __init__(self):
        self.arxiv_client = ArxivClient()
        self.semantic_scholar_client = SemanticScholarClient()
        self.youtube_client = YouTubeClient()
        self.google_client = GoogleSearchClient()
    
    async def find_resources(
        self,
        skill: str,
        force_refresh: bool = False,
        max_per_source: int = 3
    ) -> Dict:
        """
        Find teaching resources for a skill/competency
        
        Args:
            skill: The skill/competency to search for (e.g., "questioning techniques")
            force_refresh: Force new API calls even if cached results exist
            max_per_source: Maximum results per API source
        
        Returns:
            {
                "skill": "questioning techniques",
                "framework_mapping": {...},
                "cached_results": [...],
                "new_results": [...],
                "sources_used": ["arxiv", "youtube"],
                "sources_failed": ["google"],
                "total_results": 10
            }
        """
        logger.info(f"🔍 Finding resources for skill: {skill}")
        
        # Step 1: Map skill to Singapore Teaching Framework
        framework_mapping = map_skill_to_framework(skill)
        logger.info(f"📊 Mapped to framework: {framework_mapping['framework_code']} - {framework_mapping['framework_name']}")
        
        # Step 2: Check cache (unless force_refresh)
        cached_resources = []
        if not force_refresh:
            cached_resources = search_resources_by_skill(
                skill=skill,
                framework_code=framework_mapping['framework_code']
            )
            
            if cached_resources:
                logger.info(f"✅ Found {len(cached_resources)} cached resources")
                
                # Increment usage counts
                for resource in cached_resources:
                    increment_usage_count(resource['url'])
                
                return {
                    "skill": skill,
                    "framework_mapping": framework_mapping,
                    "cached_results": cached_resources,
                    "new_results": [],
                    "sources_used": [],
                    "sources_failed": [],
                    "total_results": len(cached_resources),
                    "from_cache": True
                }
        
        # Step 3: Call all APIs in parallel
        logger.info(f"🌐 Calling external APIs for: {skill}")
        
        results = await self._call_all_apis(skill, max_per_source)
        
        # Step 4: Rank and filter results
        ranked_results = self._rank_results(results, skill)
        
        # Step 5: Store new results in CSV
        new_resources = []
        for result in ranked_results:
            resource_entry = {
                "skill_category": framework_mapping['framework_name'],
                "framework_code": framework_mapping['framework_code'],
                "resource_type": result['type'],
                "title": result['title'],
                "url": result['url'],
                "description": result['description'],
                "source_api": result['source'],
                "citations_or_views": result.get('citations', result.get('views', 'N/A')),
                "relevance_score": result['relevance_score'],
                "metadata_json": json.dumps({
                    "authors": result.get('authors', []),
                    "published": result.get('published', 'N/A'),
                    "channel": result.get('channel', 'N/A'),
                    "duration": result.get('duration', 'N/A')
                })
            }
            
            if add_resource(resource_entry):
                new_resources.append(resource_entry)
        
        logger.info(f"✅ Added {len(new_resources)} new resources to cache")
        
        # Step 6: Compile response
        return {
            "skill": skill,
            "framework_mapping": framework_mapping,
            "cached_results": cached_resources,
            "new_results": new_resources,
            "sources_used": results['sources_used'],
            "sources_failed": results['sources_failed'],
            "total_results": len(cached_resources) + len(new_resources),
            "from_cache": False
        }
    
    async def _call_all_apis(self, skill: str, max_per_source: int) -> Dict:
        """Call all APIs in parallel"""
        
        # Create tasks for all APIs
        tasks = {
            'arxiv': self._safe_api_call(self.arxiv_client.search(skill, max_per_source), 'arxiv'),
            'semantic_scholar': self._safe_api_call(self.semantic_scholar_client.search(skill, max_per_source), 'semantic_scholar'),
            'youtube': self._safe_api_call(self.youtube_client.search(skill, max_per_source), 'youtube'),
            'google': self._safe_api_call(self.google_client.search(skill, max_per_source), 'google')
        }
        
        # Execute all in parallel
        results = await asyncio.gather(*tasks.values())
        
        # Combine results
        all_results = []
        sources_used = []
        sources_failed = []
        
        for api_name, result in zip(tasks.keys(), results):
            if result['success']:
                all_results.extend(result['data'])
                if result['data']:  # Only count as "used" if it returned results
                    sources_used.append(api_name)
            else:
                sources_failed.append(api_name)
                logger.warning(f"⚠️  {api_name} API failed: {result.get('error', 'Unknown error')}")
        
        return {
            'resources': all_results,
            'sources_used': sources_used,
            'sources_failed': sources_failed
        }
    
    async def _safe_api_call(self, coro, api_name: str) -> Dict:
        """Safely execute an API call with error handling"""
        try:
            data = await coro
            return {
                'success': True,
                'data': data,
                'api': api_name
            }
        except Exception as e:
            logger.error(f"❌ {api_name} API call failed: {e}")
            return {
                'success': False,
                'data': [],
                'api': api_name,
                'error': str(e)
            }
    
    def _rank_results(self, results: Dict, skill: str) -> List[Dict]:
        """
        Rank and filter results by relevance
        
        Assigns relevance scores based on:
        - Position in API results (first = best)
        - Source credibility
        """
        resources = results['resources']
        
        # Group by source
        by_source = {}
        for resource in resources:
            source = resource['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(resource)
        
        # Assign relevance scores
        ranked = []
        for source, source_resources in by_source.items():
            for i, resource in enumerate(source_resources):
                # Score based on position (first result = higher score)
                position_score = 1.0 - (i / len(source_resources)) * 0.3
                
                # Boost academic sources slightly
                source_boost = 1.0
                if source in ['arxiv', 'semantic_scholar']:
                    source_boost = 1.1
                
                resource['relevance_score'] = round(min(position_score * source_boost, 1.0), 2)
                ranked.append(resource)
        
        # Sort by relevance score
        ranked.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Return top results (max 15)
        return ranked[:15]
    
    def format_resources_for_display(self, resources_data: Dict) -> str:
        """
        Format resources as markdown for chatbot display
        
        Args:
            resources_data: Output from find_resources()
        
        Returns:
            Formatted markdown string
        """
        skill = resources_data['skill']
        framework = resources_data['framework_mapping']
        cached = resources_data['cached_results']
        new = resources_data['new_results']
        from_cache = resources_data['from_cache']
        
        # Build header
        output = [f"## 📚 Resources for: **{skill}**\n"]
        
        if framework['framework_code'] != 'CUSTOM':
            output.append(f"*Singapore Teaching Practice: {framework['framework_code']} - {framework['framework_name']}*\n")
        
        output.append(f"*Found {resources_data['total_results']} resources*\n")
        
        # Group resources by type
        all_resources = cached + new
        
        papers = [r for r in all_resources if r.get('resource_type') == 'paper']
        videos = [r for r in all_resources if r.get('resource_type') == 'video']
        articles = [r for r in all_resources if r.get('resource_type') == 'article']
        
        # Display papers
        if papers:
            output.append("\n### 📄 Academic Papers\n")
            for i, paper in enumerate(papers[:5], 1):
                title = paper.get('title', 'Untitled')
                url = paper.get('url', '#')
                desc = paper.get('description', '')[:200]
                citations = paper.get('citations_or_views', 'N/A')
                
                # Parse metadata
                try:
                    metadata = json.loads(paper.get('metadata_json', '{}'))
                    authors = metadata.get('authors', [])
                    author_str = ', '.join(authors[:2]) if authors else 'Unknown'
                    if len(authors) > 2:
                        author_str += ' et al.'
                    published = metadata.get('published', 'N/A')
                except:
                    author_str = 'Unknown'
                    published = 'N/A'
                
                output.append(f"{i}. **[{title}]({url})**")
                output.append(f"   - Authors: {author_str}")
                output.append(f"   - Published: {published} | Citations: {citations}")
                output.append(f"   - *{desc}...*\n")
        
        # Display videos
        if videos:
            output.append("\n### 📺 Educational Videos\n")
            for i, video in enumerate(videos[:5], 1):
                title = video.get('title', 'Untitled')
                url = video.get('url', '#')
                desc = video.get('description', '')[:150]
                views = video.get('citations_or_views', 'N/A')
                
                # Parse metadata
                try:
                    metadata = json.loads(video.get('metadata_json', '{}'))
                    channel = metadata.get('channel', 'Unknown')
                    duration = metadata.get('duration', 'N/A')
                except:
                    channel = 'Unknown'
                    duration = 'N/A'
                
                output.append(f"{i}. **[{title}]({url})**")
                output.append(f"   - Channel: {channel} | Duration: {duration}")
                output.append(f"   - Views: {views}")
                output.append(f"   - *{desc}...*\n")
        
        # Display articles
        if articles:
            output.append("\n### 🌐 Articles & Resources\n")
            for i, article in enumerate(articles[:5], 1):
                title = article.get('title', 'Untitled')
                url = article.get('url', '#')
                desc = article.get('description', '')[:150]
                
                output.append(f"{i}. **[{title}]({url})**")
                output.append(f"   - *{desc}...*\n")
        
        # Add footer
        if from_cache:
            output.append("\n---")
            output.append("\n💡 *These resources were retrieved from cache. Would you like me to search for more recent resources?*")
        else:
            if resources_data['sources_failed']:
                output.append("\n---")
                output.append(f"\n⚠️  *Note: Some sources were unavailable: {', '.join(resources_data['sources_failed'])}*")
        
        return '\n'.join(output)
    
    async def close(self):
        """Close all API clients"""
        await self.arxiv_client.close()
        await self.semantic_scholar_client.close()
        await self.youtube_client.close()
        await self.google_client.close()


# Global instance
resource_finder = ResourceFinder()