"""
Mapping utilities for converting class predictions to correction codes and parts.
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from omegaconf import OmegaConf
from loguru import logger


class PredictionMapper:
    """Maps model predictions to correction codes and vehicle parts."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the prediction mapper."""
        self.cfg = OmegaConf.load(config_path)
        self.correction_codes = self._load_correction_codes()
        self.parts_mapping = self._load_parts_mapping()
        
        logger.info("Prediction mapper initialized")
    
    def _load_correction_codes(self) -> Dict:
        """Load correction codes mapping from YAML file."""
        codes_file = Path("configs/correction_codes.yaml")
        
        if codes_file.exists():
            codes_data = OmegaConf.load(codes_file)
            logger.info(f"Loaded correction codes for {len(codes_data.damage_types)} damage types")
            return codes_data
        
        logger.warning("Correction codes file not found, using empty mapping")
        return {"damage_types": {}, "severity_mapping": {}, "part_categories": {}}
    
    def _load_parts_mapping(self) -> Dict:
        """Load vehicle parts mapping."""
        # This could be loaded from a separate file or embedded here
        parts_mapping = {
            # Exterior parts
            "front_bumper": {
                "category": "exterior",
                "part_code": "FB001",
                "description": "Front bumper assembly",
                "labor_hours": 2.5,
                "common_damage": ["scratches", "dents", "cracks", "missing"]
            },
            "rear_bumper": {
                "category": "exterior",
                "part_code": "RB001", 
                "description": "Rear bumper assembly",
                "labor_hours": 2.0,
                "common_damage": ["scratches", "dents", "cracks", "missing"]
            },
            "hood": {
                "category": "exterior",
                "part_code": "HD001",
                "description": "Hood panel",
                "labor_hours": 3.0,
                "common_damage": ["dents", "scratches", "misaligned"]
            },
            "front_door": {
                "category": "exterior",
                "part_code": "FD001",
                "description": "Front door panel",
                "labor_hours": 4.0,
                "common_damage": ["dents", "scratches", "misaligned"]
            },
            "rear_door": {
                "category": "exterior",
                "part_code": "RD001",
                "description": "Rear door panel", 
                "labor_hours": 4.0,
                "common_damage": ["dents", "scratches", "misaligned"]
            },
            "front_fender": {
                "category": "exterior",
                "part_code": "FF001",
                "description": "Front fender panel",
                "labor_hours": 3.5,
                "common_damage": ["dents", "scratches", "rust"]
            },
            "rear_fender": {
                "category": "exterior", 
                "part_code": "RF001",
                "description": "Rear fender panel",
                "labor_hours": 3.5,
                "common_damage": ["dents", "scratches", "rust"]
            },
            "headlight": {
                "category": "lighting",
                "part_code": "HL001",
                "description": "Headlight assembly",
                "labor_hours": 1.5,
                "common_damage": ["cracked", "missing", "foggy"]
            },
            "taillight": {
                "category": "lighting",
                "part_code": "TL001", 
                "description": "Taillight assembly",
                "labor_hours": 1.0,
                "common_damage": ["cracked", "missing", "broken"]
            },
            "side_mirror": {
                "category": "exterior",
                "part_code": "SM001",
                "description": "Side mirror assembly",
                "labor_hours": 1.0,
                "common_damage": ["cracked", "missing", "loose"]
            },
            "windshield": {
                "category": "glass",
                "part_code": "WS001",
                "description": "Windshield",
                "labor_hours": 2.0,
                "common_damage": ["cracked", "chipped", "shattered"]
            },
            "roof": {
                "category": "exterior",
                "part_code": "RF001",
                "description": "Roof panel",
                "labor_hours": 5.0,
                "common_damage": ["dents", "scratches", "hail_damage"]
            }
        }
        
        logger.info(f"Loaded parts mapping for {len(parts_mapping)} vehicle parts")
        return parts_mapping
    
    def map_predictions_to_codes(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Map model predictions to correction codes and parts information."""
        
        predicted_classes = []
        correction_codes = []
        parts_info = []
        severity_analysis = {}
        
        # Extract predicted classes from fused predictions
        for class_name, pred_data in predictions.items():
            if pred_data.get('predicted', False):
                predicted_classes.append(class_name)
                
                # Get correction code
                code_info = self._get_correction_code(class_name, pred_data)
                if code_info:
                    correction_codes.append(code_info)
                
                # Get parts information
                part_info = self._get_part_info(class_name, pred_data)
                if part_info:
                    parts_info.append(part_info)
                
                # Analyze severity
                severity = self._analyze_severity(class_name, pred_data)
                if severity:
                    severity_analysis[class_name] = severity
        
        # Calculate overall damage assessment
        damage_assessment = self._calculate_damage_assessment(
            predicted_classes, severity_analysis
        )
        
        result = {
            'predicted_classes': predicted_classes,
            'correction_codes': correction_codes,
            'parts_information': parts_info,
            'severity_analysis': severity_analysis,
            'damage_assessment': damage_assessment,
            'repair_recommendations': self._generate_repair_recommendations(
                predicted_classes, severity_analysis
            )
        }
        
        return result
    
    def _get_correction_code(self, class_name: str, pred_data: Dict) -> Optional[Dict]:
        """Get correction code for a predicted damage class."""
        
        damage_types = self.correction_codes.get('damage_types', {})
        
        if class_name in damage_types:
            code_data = damage_types[class_name]
            
            return {
                'class_name': class_name,
                'correction_code': code_data.get('code', 'UNKNOWN'),
                'description': code_data.get('description', 'No description'),
                'category': code_data.get('category', 'general'),
                'confidence': pred_data.get('fused_score', 0.0),
                'labor_type': code_data.get('labor_type', 'repair'),
                'estimated_cost_range': code_data.get('cost_range', [0, 1000])
            }
        
        return None
    
    def _get_part_info(self, class_name: str, pred_data: Dict) -> Optional[Dict]:
        """Get vehicle part information for a predicted damage class."""
        
        # Map class name to part (you might need more sophisticated mapping)
        part_name = self._class_to_part_mapping(class_name)
        
        if part_name and part_name in self.parts_mapping:
            part_data = self.parts_mapping[part_name]
            
            return {
                'part_name': part_name,
                'part_code': part_data.get('part_code', 'UNKNOWN'),
                'description': part_data.get('description', ''),
                'category': part_data.get('category', 'unknown'),
                'estimated_labor_hours': part_data.get('labor_hours', 0.0),
                'common_damage_types': part_data.get('common_damage', []),
                'detected_damage': class_name,
                'confidence': pred_data.get('fused_score', 0.0)
            }
        
        return None
    
    def _class_to_part_mapping(self, class_name: str) -> Optional[str]:
        """Map damage class to vehicle part."""
        
        # Define mapping from damage classes to parts
        class_to_part = {
            'front_bumper_damage': 'front_bumper',
            'rear_bumper_damage': 'rear_bumper', 
            'hood_damage': 'hood',
            'door_damage': 'front_door',  # Could be front or rear
            'fender_damage': 'front_fender',
            'headlight_damage': 'headlight',
            'taillight_damage': 'taillight',
            'mirror_damage': 'side_mirror',
            'windshield_damage': 'windshield',
            'roof_damage': 'roof',
            'scratch': 'front_bumper',  # Default to most common
            'dent': 'front_door',       # Default to most common
            'crack': 'windshield',      # Default for cracks
            'rust': 'front_fender',     # Default for rust
        }
        
        return class_to_part.get(class_name)
    
    def _analyze_severity(self, class_name: str, pred_data: Dict) -> Optional[Dict]:
        """Analyze damage severity based on confidence and class type."""
        
        confidence = pred_data.get('fused_score', 0.0)
        
        # Define severity thresholds
        if confidence >= 0.9:
            severity = 'severe'
        elif confidence >= 0.7:
            severity = 'moderate'
        elif confidence >= 0.5:
            severity = 'minor'
        else:
            severity = 'minimal'
        
        # Map severity to repair actions
        severity_mapping = self.correction_codes.get('severity_mapping', {})
        
        return {
            'severity_level': severity,
            'confidence': confidence,
            'repair_action': severity_mapping.get(severity, {}).get('action', 'inspect'),
            'urgency': severity_mapping.get(severity, {}).get('urgency', 'low'),
            'estimated_cost_multiplier': severity_mapping.get(severity, {}).get('cost_multiplier', 1.0)
        }
    
    def _calculate_damage_assessment(self, predicted_classes: List[str], 
                                   severity_analysis: Dict) -> Dict:
        """Calculate overall damage assessment."""
        
        if not predicted_classes:
            return {
                'overall_severity': 'none',
                'total_estimated_cost': 0,
                'repair_priority': 'none',
                'total_labor_hours': 0.0,
                'safety_impact': 'none'
            }
        
        # Calculate overall severity
        severity_scores = {'minimal': 1, 'minor': 2, 'moderate': 3, 'severe': 4}
        max_severity_score = max([
            severity_scores.get(analysis.get('severity_level', 'minimal'), 1)
            for analysis in severity_analysis.values()
        ])
        
        overall_severity = {v: k for k, v in severity_scores.items()}[max_severity_score]
        
        # Estimate total cost and labor
        total_cost = 0
        total_labor = 0.0
        
        for class_name in predicted_classes:
            part_name = self._class_to_part_mapping(class_name)
            if part_name and part_name in self.parts_mapping:
                part_data = self.parts_mapping[part_name]
                
                # Base cost estimation (simplified)
                base_cost = 500  # Base repair cost
                labor_hours = part_data.get('labor_hours', 2.0)
                labor_rate = 100  # $ per hour
                
                severity_mult = severity_analysis.get(class_name, {}).get(
                    'estimated_cost_multiplier', 1.0
                )
                
                repair_cost = (base_cost + labor_hours * labor_rate) * severity_mult
                total_cost += repair_cost
                total_labor += labor_hours
        
        # Determine repair priority
        if max_severity_score >= 4:
            priority = 'immediate'
        elif max_severity_score >= 3:
            priority = 'high'
        elif max_severity_score >= 2:
            priority = 'medium'
        else:
            priority = 'low'
        
        # Assess safety impact
        safety_critical_parts = ['headlight', 'taillight', 'windshield', 'side_mirror']
        safety_impact = 'high' if any(
            self._class_to_part_mapping(cls) in safety_critical_parts 
            for cls in predicted_classes
        ) else 'low'
        
        return {
            'overall_severity': overall_severity,
            'total_estimated_cost': int(total_cost),
            'repair_priority': priority,
            'total_labor_hours': round(total_labor, 1),
            'safety_impact': safety_impact,
            'damage_count': len(predicted_classes)
        }
    
    def _generate_repair_recommendations(self, predicted_classes: List[str], 
                                       severity_analysis: Dict) -> List[Dict]:
        """Generate repair recommendations based on damage analysis."""
        
        recommendations = []
        
        if not predicted_classes:
            return [{
                'type': 'no_damage',
                'description': 'No damage detected, vehicle appears to be in good condition',
                'priority': 'none'
            }]
        
        # Sort by severity for prioritized recommendations
        sorted_classes = sorted(
            predicted_classes,
            key=lambda x: {'severe': 4, 'moderate': 3, 'minor': 2, 'minimal': 1}.get(
                severity_analysis.get(x, {}).get('severity_level', 'minimal'), 1
            ),
            reverse=True
        )
        
        for class_name in sorted_classes:
            severity_info = severity_analysis.get(class_name, {})
            part_name = self._class_to_part_mapping(class_name)
            
            recommendation = {
                'damage_type': class_name,
                'affected_part': part_name,
                'severity': severity_info.get('severity_level', 'minimal'),
                'recommended_action': severity_info.get('repair_action', 'inspect'),
                'urgency': severity_info.get('urgency', 'low'),
                'description': self._get_repair_description(class_name, severity_info),
                'estimated_cost': self._estimate_repair_cost(class_name, severity_info)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_repair_description(self, class_name: str, severity_info: Dict) -> str:
        """Generate repair description for damage type."""
        
        severity = severity_info.get('severity_level', 'minimal')
        part_name = self._class_to_part_mapping(class_name)
        
        if severity == 'severe':
            return f"Immediate replacement of {part_name} required due to severe {class_name}"
        elif severity == 'moderate':
            return f"Professional repair of {part_name} recommended for {class_name}"
        elif severity == 'minor':
            return f"Minor repair needed for {class_name} on {part_name}"
        else:
            return f"Inspect {part_name} for {class_name}, may not require immediate action"
    
    def _estimate_repair_cost(self, class_name: str, severity_info: Dict) -> Dict:
        """Estimate repair cost for damage type."""
        
        part_name = self._class_to_part_mapping(class_name)
        base_cost = 300  # Base cost
        
        if part_name and part_name in self.parts_mapping:
            labor_hours = self.parts_mapping[part_name].get('labor_hours', 2.0)
            base_cost = 200 + (labor_hours * 100)  # Parts + labor
        
        multiplier = severity_info.get('estimated_cost_multiplier', 1.0)
        estimated_cost = int(base_cost * multiplier)
        
        return {
            'estimated_cost': estimated_cost,
            'cost_range': [
                int(estimated_cost * 0.8), 
                int(estimated_cost * 1.3)
            ],
            'currency': 'USD'
        }


def create_mapper(config_path: str = "configs/config.yaml") -> PredictionMapper:
    """Factory function to create prediction mapper."""
    return PredictionMapper(config_path)


if __name__ == "__main__":
    # Test the mapper
    mapper = create_mapper()
    
    # Test with dummy predictions
    test_predictions = {
        'front_bumper_damage': {
            'fused_score': 0.85,
            'predicted': True,
            'threshold': 0.5
        },
        'headlight_damage': {
            'fused_score': 0.72,
            'predicted': True, 
            'threshold': 0.5
        }
    }
    
    result = mapper.map_predictions_to_codes(test_predictions)
    
    print("Mapping Test Results:")
    print(f"Predicted classes: {result['predicted_classes']}")
    print(f"Number of correction codes: {len(result['correction_codes'])}")
    print(f"Overall damage assessment: {result['damage_assessment']['overall_severity']}")
    print(f"Estimated cost: ${result['damage_assessment']['total_estimated_cost']}")
    print(f"Number of recommendations: {len(result['repair_recommendations'])}")