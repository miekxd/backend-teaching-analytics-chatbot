"""
API Clients for External Teaching Resources

Clients for:
- arXiv API
- Semantic Scholar API
- YouTube Data API v3
- Google Custom Search API
"""
import aiohttp
import logging
from typing import List, Dict, Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET

from app.core.config import settings

logger = logging.getLogger(__name__)


class ResourceAPIClient:
    """Base class for resource API clients"""
    
    def __init__(self):
        self.session = None
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()


class ArxivClient(ResourceAPIClient):
    """arXiv API Client - Academic papers"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search arXiv for academic papers
        
        Args:
            query: Search query (e.g., "questioning techniques in education")
            max_results: Maximum number of results
        
        Returns:
            List of paper dictionaries
        """
        try:
            session = await self.get_session()
            
            # Build search query - add "education" or "teaching" context
            search_query = f"all:{query} AND (cat:cs.CY OR cat:physics.ed-ph OR all:education OR all:teaching)"
            
            params = {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"arXiv API error: {response.status}")
                    return []
                
                xml_data = await response.text()
                return self._parse_arxiv_response(xml_data)
        
        except Exception as e:
            logger.error(f"❌ arXiv API error: {e}", exc_info=True)
            return []
    
    def _parse_arxiv_response(self, xml_data: str) -> List[Dict]:
        """Parse arXiv XML response"""
        try:
            root = ET.fromstring(xml_data)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            results = []
            for entry in root.findall('atom:entry', namespace):
                # Extract data
                title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')[:300]
                link = entry.find('atom:id', namespace).text
                published = entry.find('atom:published', namespace).text[:10]
                
                # Extract authors
                authors = [
                    author.find('atom:name', namespace).text
                    for author in entry.findall('atom:author', namespace)
                ]
                
                results.append({
                    "type": "paper",
                    "title": title,
                    "url": link,
                    "description": summary,
                    "source": "arxiv",
                    "authors": authors,
                    "published": published,
                    "citations": "N/A"
                })
            
            logger.info(f"✅ arXiv: Found {len(results)} papers")
            return results
        
        except Exception as e:
            logger.error(f"Error parsing arXiv response: {e}")
            return []


class SemanticScholarClient(ResourceAPIClient):
    """Semantic Scholar API Client"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Semantic Scholar for academic papers"""
        try:
            session = await self.get_session()
            
            # Add education context
            search_query = f"{query} education teaching"
            
            params = {
                "query": search_query,
                "limit": max_results,
                "fields": "title,abstract,url,citationCount,year,authors,publicationDate"
            }
            
            headers = {}
            if hasattr(settings, 'SEMANTIC_SCHOLAR_API_KEY') and settings.SEMANTIC_SCHOLAR_API_KEY:
                headers['x-api-key'] = settings.SEMANTIC_SCHOLAR_API_KEY
            
            async with session.get(self.BASE_URL, params=params, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Semantic Scholar API error: {response.status}")
                    return []
                
                data = await response.json()
                return self._parse_semantic_scholar_response(data)
        
        except Exception as e:
            logger.error(f"❌ Semantic Scholar API error: {e}", exc_info=True)
            return []
    
    def _parse_semantic_scholar_response(self, data: dict) -> List[Dict]:
        """Parse Semantic Scholar response"""
        try:
            results = []
            
            for paper in data.get('data', []):
                results.append({
                    "type": "paper",
                    "title": paper.get('title', 'Untitled'),
                    "url": paper.get('url', ''),
                    "description": (paper.get('abstract', 'No abstract available.') or 'No abstract available.')[:300],
                    "source": "semantic_scholar",
                    "authors": [author.get('name', '') for author in paper.get('authors', [])],
                    "published": paper.get('publicationDate', 'Unknown')[:10],
                    "citations": f"{paper.get('citationCount', 0)} citations"
                })
            
            logger.info(f"✅ Semantic Scholar: Found {len(results)} papers")
            return results
        
        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar response: {e}")
            return []


class YouTubeClient(ResourceAPIClient):
    """YouTube Data API v3 Client"""
    
    BASE_URL = "https://www.googleapis.com/youtube/v3/search"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search YouTube for educational videos"""
        try:
            if not hasattr(settings, 'YOUTUBE_API_KEY') or not settings.YOUTUBE_API_KEY:
                logger.warning("⚠️  YouTube API key not configured")
                return []
            
            session = await self.get_session()
            
            # Add education context
            search_query = f"{query} teaching education lesson"
            
            params = {
                "part": "snippet",
                "q": search_query,
                "type": "video",
                "maxResults": max_results,
                "order": "relevance",
                "videoCategoryId": "27",  # Education category
                "key": settings.YOUTUBE_API_KEY
            }
            
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"YouTube API error: {response.status}")
                    return []
                
                data = await response.json()
                return await self._parse_youtube_response(data)
        
        except Exception as e:
            logger.error(f"❌ YouTube API error: {e}", exc_info=True)
            return []
    
    async def _parse_youtube_response(self, data: dict) -> List[Dict]:
        """Parse YouTube response"""
        try:
            results = []
            
            for item in data.get('items', []):
                video_id = item['id']['videoId']
                snippet = item['snippet']
                
                # Get video statistics
                stats = await self._get_video_stats(video_id)
                
                results.append({
                    "type": "video",
                    "title": snippet.get('title', 'Untitled'),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "description": snippet.get('description', 'No description')[:300],
                    "source": "youtube",
                    "channel": snippet.get('channelTitle', 'Unknown'),
                    "published": snippet.get('publishedAt', '')[:10],
                    "views": stats.get('views', 'N/A'),
                    "duration": stats.get('duration', 'N/A')
                })
            
            logger.info(f"✅ YouTube: Found {len(results)} videos")
            return results
        
        except Exception as e:
            logger.error(f"Error parsing YouTube response: {e}")
            return []
    
    async def _get_video_stats(self, video_id: str) -> dict:
        """Get video statistics (views, duration)"""
        try:
            session = await self.get_session()
            
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "statistics,contentDetails",
                "id": video_id,
                "key": settings.YOUTUBE_API_KEY
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('items'):
                        item = data['items'][0]
                        views = item['statistics'].get('viewCount', 'N/A')
                        duration = item['contentDetails'].get('duration', 'N/A')
                        
                        # Format duration (PT15M32S → 15:32)
                        if duration != 'N/A':
                            duration = self._format_duration(duration)
                        
                        return {
                            "views": f"{int(views):,} views" if views != 'N/A' else 'N/A',
                            "duration": duration
                        }
        except:
            pass
        
        return {"views": "N/A", "duration": "N/A"}
    
    def _format_duration(self, duration: str) -> str:
        """Format ISO 8601 duration to MM:SS"""
        import re
        
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if match:
            hours, minutes, seconds = match.groups()
            hours = int(hours) if hours else 0
            minutes = int(minutes) if minutes else 0
            seconds = int(seconds) if seconds else 0
            
            if hours:
                return f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                return f"{minutes}:{seconds:02d}"
        
        return duration


class GoogleSearchClient(ResourceAPIClient):
    """Google Custom Search API Client"""
    
    BASE_URL = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Google for educational articles/websites"""
        try:
            if not hasattr(settings, 'GOOGLE_CUSTOM_SEARCH_API_KEY') or not settings.GOOGLE_CUSTOM_SEARCH_API_KEY:
                logger.warning("⚠️  Google Custom Search API key not configured")
                return []
            
            if not hasattr(settings, 'GOOGLE_CUSTOM_SEARCH_ENGINE_ID') or not settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID:
                logger.warning("⚠️  Google Custom Search Engine ID not configured")
                return []
            
            session = await self.get_session()
            
            # Add education context
            search_query = f"{query} teaching education"
            
            params = {
                "key": settings.GOOGLE_CUSTOM_SEARCH_API_KEY,
                "cx": settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
                "q": search_query,
                "num": min(max_results, 10)  # Max 10 per request
            }
            
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"Google Search API error: {response.status}")
                    return []
                
                data = await response.json()
                return self._parse_google_response(data)
        
        except Exception as e:
            logger.error(f"❌ Google Search API error: {e}", exc_info=True)
            return []
    
    def _parse_google_response(self, data: dict) -> List[Dict]:
        """Parse Google Search response"""
        try:
            results = []
            
            for item in data.get('items', []):
                results.append({
                    "type": "article",
                    "title": item.get('title', 'Untitled'),
                    "url": item.get('link', ''),
                    "description": item.get('snippet', 'No description')[:300],
                    "source": "google_search",
                    "published": "N/A",
                    "citations": "N/A"
                })
            
            logger.info(f"✅ Google Search: Found {len(results)} articles")
            return results
        
        except Exception as e:
            logger.error(f"Error parsing Google response: {e}")
            return []