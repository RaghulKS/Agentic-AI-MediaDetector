import requests
from duckduckgo_search import DDGS
import json
import time
import hashlib
import os
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus, urlparse
import base64
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

class ImageSearchEngine:
    """Reverse image search and context retrieval engine."""
    
    def __init__(self):
        self.ddgs = DDGS()
        self.search_results_cache = {}
        
    def perform_reverse_search(self, image_path: str) -> Dict:
        """
        Perform comprehensive reverse image search.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing search results and analysis
        """
        try:
            results = {
                'search_results': [],
                'source_analysis': {},
                'credibility_assessment': {},
                'context_information': {},
                'authenticity_indicators': {}
            }
            
            # Perform text-based reverse search
            text_results = self._text_based_search(image_path)
            results['search_results'].extend(text_results)
            
            # Analyze sources
            source_analysis = self._analyze_sources(text_results)
            results['source_analysis'] = source_analysis
            
            # Assess credibility
            credibility = self._assess_source_credibility(text_results)
            results['credibility_assessment'] = credibility
            
            # Extract context information
            context = self._extract_context_information(text_results)
            results['context_information'] = context
            
            # Determine authenticity indicators from search
            auth_indicators = self._determine_authenticity_indicators(results)
            results['authenticity_indicators'] = auth_indicators
            
            return results
            
        except Exception as e:
            return {
                'error': f"Reverse search failed: {str(e)}",
                'search_results': [],
                'source_analysis': {},
                'credibility_assessment': {},
                'context_information': {},
                'authenticity_indicators': {}
            }
    
    def _text_based_search(self, image_path: str) -> List[Dict]:
        """Perform text-based search using image analysis."""
        try:
            # For this implementation, we'll use a simplified approach
            # In production, you might want to integrate with Google Vision API
            # or other image recognition services for better text extraction
            
            results = []
            
            # Generate search terms based on image analysis
            search_terms = self._generate_search_terms(image_path)
            
            for term in search_terms[:3]:  # Limit to top 3 terms
                try:
                    # Perform web search
                    search_results = list(self.ddgs.text(
                        keywords=term,
                        region='wt-wt',
                        safesearch='moderate',
                        max_results=5
                    ))
                    
                    for result in search_results:
                        results.append({
                            'title': result.get('title', ''),
                            'url': result.get('href', ''),
                            'snippet': result.get('body', ''),
                            'search_term': term,
                            'source': 'duckduckgo_text'
                        })
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    continue
            
            return results
            
        except Exception as e:
            return []
    
    def _generate_search_terms(self, image_path: str) -> List[str]:
        """Generate search terms based on image content analysis."""
        try:
            # This is a simplified implementation
            # In production, you'd use computer vision models to extract objects,
            # text, and other features from the image
            
            search_terms = []
            
            # Get basic image info
            with Image.open(image_path) as img:
                # Basic terms based on image properties
                width, height = img.size
                aspect_ratio = width / height
                
                if aspect_ratio > 1.5:
                    search_terms.append("landscape photograph")
                elif aspect_ratio < 0.7:
                    search_terms.append("portrait photograph")
                else:
                    search_terms.append("square image")
                
                # Add generic terms that might help identify the image
                search_terms.extend([
                    "authentic photograph verification",
                    "original image source",
                    "reverse image search",
                    "image authenticity check"
                ])
            
            return search_terms
            
        except Exception:
            return ["image authenticity verification"]
    
    def _analyze_sources(self, search_results: List[Dict]) -> Dict:
        """Analyze the sources found in search results."""
        try:
            analysis = {
                'total_sources': len(search_results),
                'unique_domains': set(),
                'domain_frequency': {},
                'source_types': {},
                'credible_sources': 0,
                'suspicious_sources': 0
            }
            
            for result in search_results:
                url = result.get('url', '')
                if url:
                    try:
                        domain = urlparse(url).netloc.lower()
                        analysis['unique_domains'].add(domain)
                        
                        # Count domain frequency
                        analysis['domain_frequency'][domain] = analysis['domain_frequency'].get(domain, 0) + 1
                        
                        # Categorize source type
                        source_type = self._categorize_source(domain)
                        analysis['source_types'][source_type] = analysis['source_types'].get(source_type, 0) + 1
                        
                        # Check credibility
                        if self._is_credible_source(domain):
                            analysis['credible_sources'] += 1
                        elif self._is_suspicious_source(domain):
                            analysis['suspicious_sources'] += 1
                            
                    except Exception:
                        continue
            
            analysis['unique_domains'] = list(analysis['unique_domains'])
            
            return analysis
            
        except Exception as e:
            return {'analysis_error': str(e)}
    
    def _categorize_source(self, domain: str) -> str:
        """Categorize the source type based on domain."""
        news_sites = ['cnn.com', 'bbc.com', 'reuters.com', 'ap.org', 'nytimes.com']
        social_media = ['twitter.com', 'facebook.com', 'instagram.com', 'reddit.com']
        image_hosts = ['imgur.com', 'flickr.com', 'getty.com', 'shutterstock.com']
        ai_sites = ['openai.com', 'midjourney.com', 'stability.ai', 'huggingface.co']
        
        domain_lower = domain.lower()
        
        if any(news in domain_lower for news in news_sites):
            return 'news_media'
        elif any(social in domain_lower for social in social_media):
            return 'social_media'
        elif any(img in domain_lower for img in image_hosts):
            return 'image_hosting'
        elif any(ai in domain_lower for ai in ai_sites):
            return 'ai_related'
        else:
            return 'other'
    
    def _is_credible_source(self, domain: str) -> bool:
        """Check if domain is from a credible source."""
        credible_domains = [
            'reuters.com', 'ap.org', 'bbc.com', 'cnn.com', 'nytimes.com',
            'washingtonpost.com', 'theguardian.com', 'npr.org', 'pbs.org',
            'getty.com', 'shutterstock.com', 'nationalgeographic.com'
        ]
        
        return any(credible in domain.lower() for credible in credible_domains)
    
    def _is_suspicious_source(self, domain: str) -> bool:
        """Check if domain might be suspicious for authenticity."""
        suspicious_patterns = [
            'ai-generated', 'fake', 'generated', 'artificial', 'synthetic',
            'deepfake', 'bot', 'spam'
        ]
        
        return any(pattern in domain.lower() for pattern in suspicious_patterns)
    
    def _assess_source_credibility(self, search_results: List[Dict]) -> Dict:
        """Assess overall credibility of sources."""
        try:
            assessment = {
                'overall_score': 0.0,
                'credibility_factors': {},
                'red_flags': [],
                'positive_indicators': []
            }
            
            if not search_results:
                assessment['overall_score'] = 0.5
                return assessment
            
            credible_count = 0
            suspicious_count = 0
            total_count = len(search_results)
            
            for result in search_results:
                url = result.get('url', '')
                title = result.get('title', '').lower()
                snippet = result.get('snippet', '').lower()
                
                if url:
                    domain = urlparse(url).netloc.lower()
                    
                    if self._is_credible_source(domain):
                        credible_count += 1
                        assessment['positive_indicators'].append(f"Found on credible source: {domain}")
                    elif self._is_suspicious_source(domain):
                        suspicious_count += 1
                        assessment['red_flags'].append(f"Found on suspicious source: {domain}")
                
                # Check content for authenticity-related keywords
                authenticity_keywords = ['authentic', 'original', 'real', 'verified']
                fake_keywords = ['fake', 'generated', 'artificial', 'ai-made', 'synthetic']
                
                auth_score = sum(1 for keyword in authenticity_keywords if keyword in title or keyword in snippet)
                fake_score = sum(1 for keyword in fake_keywords if keyword in title or keyword in snippet)
                
                if fake_score > auth_score:
                    assessment['red_flags'].append("Content suggests artificial generation")
                elif auth_score > fake_score:
                    assessment['positive_indicators'].append("Content suggests authenticity")
            
            # Calculate overall credibility score
            if total_count > 0:
                credibility_ratio = credible_count / total_count
                suspicion_ratio = suspicious_count / total_count
                
                base_score = 0.5
                base_score += credibility_ratio * 0.3
                base_score -= suspicion_ratio * 0.3
                
                assessment['overall_score'] = max(0.0, min(1.0, base_score))
            
            assessment['credibility_factors'] = {
                'credible_sources': credible_count,
                'suspicious_sources': suspicious_count,
                'total_sources': total_count,
                'credibility_ratio': credible_count / total_count if total_count > 0 else 0
            }
            
            return assessment
            
        except Exception as e:
            return {'assessment_error': str(e), 'overall_score': 0.5}
    
    def _extract_context_information(self, search_results: List[Dict]) -> Dict:
        """Extract contextual information from search results."""
        try:
            context = {
                'topics': [],
                'locations': [],
                'dates': [],
                'entities': [],
                'related_content': []
            }
            
            for result in search_results:
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                text = f"{title} {snippet}".lower()
                
                # Simple keyword extraction (in production, use NLP)
                # Extract potential topics
                topic_keywords = ['sports', 'politics', 'nature', 'technology', 'art', 'science']
                for keyword in topic_keywords:
                    if keyword in text and keyword not in context['topics']:
                        context['topics'].append(keyword)
                
                # Extract potential locations (simplified)
                location_indicators = ['in ', 'at ', 'from ']
                for indicator in location_indicators:
                    if indicator in text:
                        # This is very simplified - in production, use proper NER
                        parts = text.split(indicator)
                        if len(parts) > 1:
                            potential_location = parts[1].split()[0]
                            if len(potential_location) > 2:
                                context['locations'].append(potential_location)
                
                # Store related content
                context['related_content'].append({
                    'title': title,
                    'snippet': snippet[:200],
                    'url': result.get('url', '')
                })
            
            # Remove duplicates
            context['topics'] = list(set(context['topics']))
            context['locations'] = list(set(context['locations']))
            
            return context
            
        except Exception as e:
            return {'context_error': str(e)}
    
    def _determine_authenticity_indicators(self, search_results: Dict) -> Dict:
        """Determine authenticity indicators from search results."""
        try:
            indicators = {
                'source_authenticity_score': 0.0,
                'search_based_indicators': [],
                'confidence_level': 'low'
            }
            
            # Get credibility assessment
            credibility = search_results.get('credibility_assessment', {})
            credibility_score = credibility.get('overall_score', 0.5)
            
            indicators['source_authenticity_score'] = credibility_score
            
            # Analyze search patterns
            total_results = len(search_results.get('search_results', []))
            
            if total_results == 0:
                indicators['search_based_indicators'].append("No reverse search matches found")
                indicators['confidence_level'] = 'low'
            elif total_results < 5:
                indicators['search_based_indicators'].append("Limited reverse search matches")
                indicators['confidence_level'] = 'medium'
            else:
                indicators['search_based_indicators'].append("Multiple reverse search matches found")
                indicators['confidence_level'] = 'high'
            
            # Check for positive indicators
            positive_indicators = credibility.get('positive_indicators', [])
            if positive_indicators:
                indicators['search_based_indicators'].extend(positive_indicators)
            
            # Check for red flags
            red_flags = credibility.get('red_flags', [])
            if red_flags:
                indicators['search_based_indicators'].extend([f"RED FLAG: {flag}" for flag in red_flags])
            
            return indicators
            
        except Exception as e:
            return {
                'indicators_error': str(e),
                'source_authenticity_score': 0.5,
                'search_based_indicators': [],
                'confidence_level': 'low'
            }


class ContextRetriever:
    """Retrieve additional context about images."""
    
    def __init__(self):
        self.ddgs = DDGS()
    
    def get_image_context(self, image_path: str, search_terms: List[str] = None) -> Dict:
        """Get contextual information about an image."""
        try:
            if not search_terms:
                search_terms = ["image authenticity", "photo verification"]
            
            context = {
                'search_terms_used': search_terms,
                'contextual_results': [],
                'relevance_score': 0.0
            }
            
            for term in search_terms[:2]:  # Limit searches
                try:
                    results = list(self.ddgs.text(
                        keywords=term,
                        region='wt-wt',
                        safesearch='moderate',
                        max_results=3
                    ))
                    
                    for result in results:
                        context['contextual_results'].append({
                            'title': result.get('title', ''),
                            'url': result.get('href', ''),
                            'snippet': result.get('body', ''),
                            'search_term': term
                        })
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception:
                    continue
            
            # Calculate relevance score based on results
            if context['contextual_results']:
                # Simple relevance scoring based on number of results
                context['relevance_score'] = min(len(context['contextual_results']) / 10.0, 1.0)
            
            return context
            
        except Exception as e:
            return {
                'error': str(e),
                'contextual_results': [],
                'relevance_score': 0.0
            }
    
    def verify_image_source(self, image_url: str) -> Dict:
        """Verify the source of an image given its URL."""
        try:
            verification = {
                'url_analysis': {},
                'domain_reputation': {},
                'accessibility': {},
                'verification_score': 0.0
            }
            
            if not image_url:
                return verification
            
            # Analyze URL
            parsed_url = urlparse(image_url)
            verification['url_analysis'] = {
                'domain': parsed_url.netloc,
                'path': parsed_url.path,
                'scheme': parsed_url.scheme,
                'is_secure': parsed_url.scheme == 'https'
            }
            
            # Check domain reputation
            domain = parsed_url.netloc.lower()
            verification['domain_reputation'] = {
                'domain': domain,
                'is_credible': self._is_credible_domain(domain),
                'is_suspicious': self._is_suspicious_domain(domain)
            }
            
            # Test accessibility
            try:
                response = requests.head(image_url, timeout=10)
                verification['accessibility'] = {
                    'accessible': response.status_code == 200,
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', '')
                }
            except:
                verification['accessibility'] = {
                    'accessible': False,
                    'status_code': None,
                    'content_type': None
                }
            
            # Calculate verification score
            score = 0.5
            if verification['url_analysis']['is_secure']:
                score += 0.1
            if verification['domain_reputation']['is_credible']:
                score += 0.2
            if verification['domain_reputation']['is_suspicious']:
                score -= 0.3
            if verification['accessibility']['accessible']:
                score += 0.1
            
            verification['verification_score'] = max(0.0, min(1.0, score))
            
            return verification
            
        except Exception as e:
            return {'verification_error': str(e), 'verification_score': 0.0}
    
    def _is_credible_domain(self, domain: str) -> bool:
        """Check if domain is credible."""
        credible_domains = [
            'reuters.com', 'ap.org', 'bbc.com', 'cnn.com',
            'getty.com', 'shutterstock.com', 'unsplash.com',
            'nationalgeographic.com', 'smithsonianmag.com'
        ]
        return any(credible in domain for credible in credible_domains)
    
    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check if domain is suspicious."""
        suspicious_patterns = [
            'fake', 'generated', 'artificial', 'ai-made',
            'deepfake', 'synthetic', 'bot'
        ]
        return any(pattern in domain for pattern in suspicious_patterns)
