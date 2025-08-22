from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import exif
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import warnings
warnings.filterwarnings('ignore')

class MetadataForensicsAnalyzer:
    
    def __init__(self):
        self.suspicious_patterns = {
            'software': [
                'midjourney', 'dall-e', 'stable diffusion', 'gpt',
                'ai', 'generated', 'artificial', 'synthetic',
                'deepfake', 'gan', 'diffusion'
            ],
            'missing_camera_data': [
                'Make', 'Model', 'DateTime', 'ExifVersion'
            ]
        }
    
    def analyze_metadata(self, image_path: str) -> Dict:
        """
        Comprehensive metadata analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing metadata analysis results
        """
        try:
            results = {
                'exif_data': {},
                'authenticity_indicators': {},
                'suspicious_patterns': [],
                'metadata_completeness': {},
                'forensic_analysis': {},
                'authenticity_score': 0.0
            }
            
            # Extract EXIF data
            exif_data = self._extract_exif_data(image_path)
            results['exif_data'] = exif_data
            
            # Analyze authenticity indicators
            auth_indicators = self._analyze_authenticity_indicators(exif_data)
            results['authenticity_indicators'] = auth_indicators
            
            # Check for suspicious patterns
            suspicious = self._detect_suspicious_patterns(exif_data)
            results['suspicious_patterns'] = suspicious
            
            # Analyze metadata completeness
            completeness = self._analyze_metadata_completeness(exif_data)
            results['metadata_completeness'] = completeness
            
            # Forensic analysis
            forensics = self._perform_forensic_analysis(image_path, exif_data)
            results['forensic_analysis'] = forensics
            
            # Calculate overall authenticity score
            auth_score = self._calculate_metadata_authenticity_score(results)
            results['authenticity_score'] = auth_score
            
            return results
            
        except Exception as e:
            return {
                'error': f"Metadata analysis failed: {str(e)}",
                'authenticity_score': 0.5,
                'exif_data': {},
                'authenticity_indicators': {},
                'suspicious_patterns': [],
                'metadata_completeness': {},
                'forensic_analysis': {}
            }
    
    def _extract_exif_data(self, image_path: str) -> Dict:
        """Extract EXIF data from image."""
        try:
            exif_data = {}
            
            # Method 1: Using PIL
            with Image.open(image_path) as image:
                if hasattr(image, '_getexif') and image._getexif() is not None:
                    exif_dict = image._getexif()
                    if exif_dict:
                        for tag_id, value in exif_dict.items():
                            tag = TAGS.get(tag_id, tag_id)
                            exif_data[tag] = str(value)
            
            # Method 2: Using exif library for more detailed data
            try:
                with open(image_path, 'rb') as f:
                    image_exif = exif.Image(f)
                    
                    if image_exif.has_exif:
                        for attr in dir(image_exif):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(image_exif, attr)
                                    if value is not None:
                                        exif_data[attr] = str(value)
                                except:
                                    continue
                                    
            except Exception:
                pass  # Fallback to PIL method only
            
            return exif_data
            
        except Exception as e:
            return {'extraction_error': str(e)}
    
    def _analyze_authenticity_indicators(self, exif_data: Dict) -> Dict:
        """Analyze EXIF data for authenticity indicators."""
        indicators = {
            'has_camera_info': False,
            'has_timestamp': False,
            'has_gps_data': False,
            'has_software_info': False,
            'camera_make': None,
            'camera_model': None,
            'software_used': None,
            'creation_time': None
        }
        
        try:
            # Check for camera information
            if 'Make' in exif_data or 'camera_make' in exif_data:
                indicators['has_camera_info'] = True
                indicators['camera_make'] = exif_data.get('Make', exif_data.get('camera_make'))
            
            if 'Model' in exif_data or 'camera_model' in exif_data:
                indicators['camera_model'] = exif_data.get('Model', exif_data.get('camera_model'))
            
            # Check for timestamp
            timestamp_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized', 'datetime', 'datetime_original']
            for field in timestamp_fields:
                if field in exif_data:
                    indicators['has_timestamp'] = True
                    indicators['creation_time'] = exif_data[field]
                    break
            
            # Check for GPS data
            gps_fields = ['GPSInfo', 'gps_latitude', 'gps_longitude']
            for field in gps_fields:
                if field in exif_data:
                    indicators['has_gps_data'] = True
                    break
            
            # Check for software information
            software_fields = ['Software', 'software']
            for field in software_fields:
                if field in exif_data:
                    indicators['has_software_info'] = True
                    indicators['software_used'] = exif_data[field]
                    break
            
            return indicators
            
        except Exception as e:
            return {**indicators, 'analysis_error': str(e)}
    
    def _detect_suspicious_patterns(self, exif_data: Dict) -> List[Dict]:
        """Detect suspicious patterns in metadata that might indicate AI generation."""
        suspicious = []
        
        try:
            # Check software field for AI-related terms
            software_field = exif_data.get('Software', exif_data.get('software', '')).lower()
            if software_field:
                for pattern in self.suspicious_patterns['software']:
                    if pattern in software_field:
                        suspicious.append({
                            'type': 'ai_software_detected',
                            'field': 'Software',
                            'value': software_field,
                            'pattern': pattern,
                            'severity': 'high'
                        })
            
            # Check for missing essential camera data
            missing_fields = []
            for field in self.suspicious_patterns['missing_camera_data']:
                if field not in exif_data:
                    missing_fields.append(field)
            
            if len(missing_fields) >= 3:  # Missing most essential fields
                suspicious.append({
                    'type': 'missing_camera_metadata',
                    'missing_fields': missing_fields,
                    'severity': 'medium'
                })
            
            # Check for unusual timestamp patterns
            if 'DateTime' in exif_data:
                try:
                    dt_str = exif_data['DateTime']
                    # Look for suspicious timestamp patterns
                    if '2023:01:01 00:00:00' in dt_str or '1970:01:01' in dt_str:
                        suspicious.append({
                            'type': 'suspicious_timestamp',
                            'value': dt_str,
                            'reason': 'Default or epoch timestamp',
                            'severity': 'medium'
                        })
                except:
                    pass
            
            # Check for unusual image dimensions that are common in AI generation
            width = exif_data.get('ExifImageWidth', exif_data.get('pixel_x_dimension'))
            height = exif_data.get('ExifImageHeight', exif_data.get('pixel_y_dimension'))
            
            if width and height:
                try:
                    w, h = int(width), int(height)
                    # Common AI generation sizes
                    ai_common_sizes = [512, 768, 1024, 1536, 2048]
                    if w in ai_common_sizes and h in ai_common_sizes:
                        suspicious.append({
                            'type': 'ai_common_dimensions',
                            'dimensions': f"{w}x{h}",
                            'severity': 'low'
                        })
                except:
                    pass
            
            return suspicious
            
        except Exception as e:
            return [{'type': 'analysis_error', 'error': str(e)}]
    
    def _analyze_metadata_completeness(self, exif_data: Dict) -> Dict:
        """Analyze how complete the metadata is."""
        expected_fields = {
            'essential': ['Make', 'Model', 'DateTime'],
            'camera_settings': ['FNumber', 'ExposureTime', 'ISO', 'FocalLength'],
            'technical': ['ColorSpace', 'ExifVersion', 'FlashPixVersion'],
            'optional': ['GPS', 'UserComment', 'ImageDescription']
        }
        
        completeness = {}
        
        for category, fields in expected_fields.items():
            present = 0
            for field in fields:
                # Check both exact match and common variations
                field_variants = [field, field.lower(), field.replace('_', '')]
                if any(variant in exif_data for variant in field_variants):
                    present += 1
            
            completeness[category] = {
                'present': present,
                'total': len(fields),
                'percentage': (present / len(fields)) * 100
            }
        
        # Overall completeness score
        total_present = sum(cat['present'] for cat in completeness.values())
        total_possible = sum(cat['total'] for cat in completeness.values())
        
        completeness['overall'] = {
            'present': total_present,
            'total': total_possible,
            'percentage': (total_present / total_possible) * 100 if total_possible > 0 else 0
        }
        
        return completeness
    
    def _perform_forensic_analysis(self, image_path: str, exif_data: Dict) -> Dict:
        """Perform additional forensic analysis."""
        forensics = {}
        
        try:
            # File size analysis
            file_size = os.path.getsize(image_path)
            forensics['file_size_bytes'] = file_size
            forensics['file_size_mb'] = round(file_size / (1024 * 1024), 2)
            
            # File hash for integrity
            with open(image_path, 'rb') as f:
                content = f.read()
                forensics['md5_hash'] = hashlib.md5(content).hexdigest()
                forensics['sha256_hash'] = hashlib.sha256(content).hexdigest()
            
            # Timestamp consistency analysis
            timestamps = self._extract_all_timestamps(exif_data)
            forensics['timestamps'] = timestamps
            forensics['timestamp_consistency'] = self._analyze_timestamp_consistency(timestamps)
            
            # Modification history indicators
            forensics['modification_indicators'] = self._detect_modification_indicators(exif_data)
            
            return forensics
            
        except Exception as e:
            return {'forensic_error': str(e)}
    
    def _extract_all_timestamps(self, exif_data: Dict) -> Dict:
        """Extract all timestamp-related data from EXIF."""
        timestamps = {}
        
        timestamp_fields = [
            'DateTime', 'DateTimeOriginal', 'DateTimeDigitized',
            'datetime', 'datetime_original', 'datetime_digitized'
        ]
        
        for field in timestamp_fields:
            if field in exif_data:
                timestamps[field] = exif_data[field]
        
        return timestamps
    
    def _analyze_timestamp_consistency(self, timestamps: Dict) -> Dict:
        """Analyze consistency between different timestamps."""
        analysis = {
            'consistent': True,
            'discrepancies': [],
            'time_gaps': {}
        }
        
        try:
            if len(timestamps) < 2:
                return analysis
            
            # Parse timestamps
            parsed_times = {}
            for field, timestamp in timestamps.items():
                try:
                    # Try different timestamp formats
                    for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                        try:
                            parsed_times[field] = datetime.strptime(timestamp, fmt)
                            break
                        except ValueError:
                            continue
                except:
                    continue
            
            if len(parsed_times) >= 2:
                # Check for unrealistic time gaps
                time_values = list(parsed_times.values())
                for i in range(len(time_values) - 1):
                    gap = abs((time_values[i+1] - time_values[i]).total_seconds())
                    if gap > 86400:  # More than 1 day difference
                        analysis['consistent'] = False
                        analysis['discrepancies'].append(f"Large time gap: {gap/3600:.1f} hours")
            
            return analysis
            
        except Exception as e:
            return {**analysis, 'analysis_error': str(e)}
    
    def _detect_modification_indicators(self, exif_data: Dict) -> List[str]:
        """Detect indicators that the image might have been modified."""
        indicators = []
        
        try:
            # Check for editing software in the software field
            software = exif_data.get('Software', '').lower()
            editing_software = ['photoshop', 'gimp', 'lightroom', 'pixelmator', 'snapseed']
            
            for editor in editing_software:
                if editor in software:
                    indicators.append(f"Edited with {editor}")
            
            # Check for multiple timestamps (might indicate processing)
            timestamp_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
            present_timestamps = [field for field in timestamp_fields if field in exif_data]
            
            if len(present_timestamps) >= 3:
                indicators.append("Multiple timestamps present")
            
            # Check for thumbnail data (might be inconsistent if modified)
            if 'thumbnail' in str(exif_data).lower():
                indicators.append("Thumbnail data present")
            
            return indicators
            
        except Exception as e:
            return [f"Detection error: {str(e)}"]
    
    def _calculate_metadata_authenticity_score(self, results: Dict) -> float:
        """Calculate overall metadata-based authenticity score."""
        try:
            score = 0.5  # Start with neutral
            
            # Positive indicators (increase authenticity score)
            auth_indicators = results.get('authenticity_indicators', {})
            
            if auth_indicators.get('has_camera_info'):
                score += 0.15
            if auth_indicators.get('has_timestamp'):
                score += 0.1
            if auth_indicators.get('has_gps_data'):
                score += 0.05
            
            # Metadata completeness bonus
            completeness = results.get('metadata_completeness', {})
            overall_completeness = completeness.get('overall', {}).get('percentage', 0)
            score += (overall_completeness / 100) * 0.2
            
            # Negative indicators (decrease authenticity score)
            suspicious_patterns = results.get('suspicious_patterns', [])
            
            for pattern in suspicious_patterns:
                severity = pattern.get('severity', 'low')
                if severity == 'high':
                    score -= 0.3
                elif severity == 'medium':
                    score -= 0.15
                else:  # low
                    score -= 0.05
            
            # Timestamp consistency
            forensics = results.get('forensic_analysis', {})
            timestamp_consistency = forensics.get('timestamp_consistency', {})
            if not timestamp_consistency.get('consistent', True):
                score -= 0.1
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            return 0.5  # Neutral score on error


class MetadataExtractor:
    """Utility class for extracting specific metadata components."""
    
    @staticmethod
    def get_camera_info(image_path: str) -> Dict:
        """Extract camera-specific information."""
        try:
            with Image.open(image_path) as image:
                exif_data = image._getexif() or {}
                
                camera_info = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in ['Make', 'Model', 'LensMake', 'LensModel']:
                        camera_info[tag] = str(value)
                
                return camera_info
                
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def get_creation_time(image_path: str) -> Optional[datetime]:
        """Extract image creation time."""
        try:
            with Image.open(image_path) as image:
                exif_data = image._getexif() or {}
                
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        return datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                
                return None
                
        except Exception:
            return None
    
    @staticmethod
    def get_gps_coordinates(image_path: str) -> Optional[Dict]:
        """Extract GPS coordinates if available."""
        try:
            with Image.open(image_path) as image:
                exif_data = image._getexif() or {}
                
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'GPSInfo':
                        gps_info = {}
                        for gps_tag_id, gps_value in value.items():
                            gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            gps_info[gps_tag] = gps_value
                        return gps_info
                
                return None
                
        except Exception:
            return None
