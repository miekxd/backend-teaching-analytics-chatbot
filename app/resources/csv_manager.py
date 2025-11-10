"""
CSV Manager for Teaching Resources

Handles reading, writing, and searching the teaching resources cache.
"""
import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# CSV file location
CSV_FILE_PATH = r"C:\Users\Samyak\Documents\URECA Project\backend-teaching-analytics-chatbot\app\resources\teaching_resources.csv"
CSV_HEADERS = [
    "resource_id",
    "skill_category",
    "framework_code",
    "resource_type",
    "title",
    "url",
    "description",
    "source_api",
    "date_added",
    "citations_or_views",
    "relevance_score",
    "usage_count",
    "metadata_json"
]


def ensure_csv_exists():
    """Ensure CSV file and directory exist"""
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
    
    if not os.path.exists(CSV_FILE_PATH):
        # Create CSV with headers
        with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        logger.info(f"✅ Created new CSV file: {CSV_FILE_PATH}")


def get_next_resource_id() -> int:
    """Get next available resource ID"""
    ensure_csv_exists()
    
    try:
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                max_id = max(int(row['resource_id']) for row in rows if row['resource_id'])
                return max_id + 1
    except Exception as e:
        logger.error(f"Error getting next ID: {e}")
    
    return 1


def search_resources_by_skill(skill: str, framework_code: str = None) -> List[Dict]:
    """
    Search for cached resources by skill
    
    Args:
        skill: The skill/competency to search for
        framework_code: Optional framework code to filter by
    
    Returns:
        List of matching resources
    """
    ensure_csv_exists()
    
    try:
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = []
            
            for row in reader:
                # Match by framework code if provided
                if framework_code and row['framework_code'] == framework_code:
                    results.append(row)
                # Or match by skill keywords in title/description
                elif skill.lower() in row['title'].lower() or skill.lower() in row['description'].lower():
                    results.append(row)
            
            logger.info(f"📋 Found {len(results)} cached resources for skill: {skill}")
            return results
    
    except Exception as e:
        logger.error(f"❌ Error searching resources: {e}")
        return []


def check_url_exists(url: str) -> bool:
    """Check if a URL already exists in the CSV (deduplication)"""
    ensure_csv_exists()
    
    try:
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['url'] == url:
                    return True
    except Exception as e:
        logger.error(f"Error checking URL: {e}")
    
    return False


def add_resource(resource: Dict) -> bool:
    """
    Add a new resource to the CSV
    
    Args:
        resource: Dict with keys matching CSV_HEADERS
    
    Returns:
        True if added successfully
    """
    ensure_csv_exists()
    
    # Check for duplicates
    if check_url_exists(resource['url']):
        logger.info(f"⚠️  Resource already exists: {resource['url']}")
        return False
    
    try:
        # Assign new ID
        resource['resource_id'] = get_next_resource_id()
        resource['date_added'] = datetime.now().strftime('%Y-%m-%d')
        resource['usage_count'] = 0
        
        # Write to CSV
        with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(resource)
        
        logger.info(f"✅ Added resource: {resource['title']}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error adding resource: {e}")
        return False


def increment_usage_count(url: str):
    """Increment usage count for a resource"""
    ensure_csv_exists()
    
    try:
        # Read all rows
        rows = []
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['url'] == url:
                    row['usage_count'] = str(int(row['usage_count']) + 1)
                rows.append(row)
        
        # Write back
        with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"✅ Incremented usage count for: {url}")
    
    except Exception as e:
        logger.error(f"❌ Error incrementing usage: {e}")


def get_all_resources() -> List[Dict]:
    """Get all resources from CSV"""
    ensure_csv_exists()
    
    try:
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        logger.error(f"Error reading all resources: {e}")
        return []