"""
Pydantic models for request and response schemas.
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import base64

from pydantic import BaseModel, Field, validator
from PIL import Image
import io


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ImageInput(BaseModel):
    """Input image for prediction."""
    image_data: str = Field(..., description="Base64 encoded image data")
    image_format: str = Field(default="JPEG", description="Image format (JPEG, PNG, etc.)")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        """Validate that image data is valid base64."""
        try:
            # Try to decode the base64 data
            image_bytes = base64.b64decode(v)
            # Try to open as PIL Image
            Image.open(io.BytesIO(image_bytes))
            return v
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def to_pil_image(self) -> Image.Image:
        """Convert base64 data to PIL Image."""
        image_bytes = base64.b64decode(self.image_data)
        return Image.open(io.BytesIO(image_bytes))


class PredictionRequest(BaseModel):
    """Prediction request model."""
    image: ImageInput = Field(..., description="Input image for prediction")
    include_details: bool = Field(default=True, description="Include detailed analysis")
    include_timing: bool = Field(default=False, description="Include timing information")
    confidence_threshold: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0,
        description="Override confidence threshold"
    )


class ClassPrediction(BaseModel):
    """Individual class prediction."""
    class_name: str = Field(..., description="Damage class name")
    fused_score: float = Field(..., ge=0.0, le=1.0, description="Fused prediction score")
    resnet_score: float = Field(..., ge=0.0, le=1.0, description="ResNet prediction score")
    yolo_score: float = Field(..., ge=0.0, le=1.0, description="YOLO prediction score")
    predicted: bool = Field(..., description="Whether class is predicted as present")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold used for prediction")
    fusion_method: str = Field(..., description="Method used for score fusion")


class CorrectionCode(BaseModel):
    """Correction code information."""
    class_name: str = Field(..., description="Associated damage class")
    correction_code: str = Field(..., description="Correction code identifier")
    description: str = Field(..., description="Code description")
    category: str = Field(..., description="Damage category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    labor_type: str = Field(..., description="Type of labor required")
    estimated_cost_range: List[int] = Field(..., description="Estimated cost range [min, max]")


class PartInformation(BaseModel):
    """Vehicle part information."""
    part_name: str = Field(..., description="Vehicle part name")
    part_code: str = Field(..., description="Part identification code")
    description: str = Field(..., description="Part description")
    category: str = Field(..., description="Part category")
    estimated_labor_hours: float = Field(..., ge=0.0, description="Estimated labor hours")
    common_damage_types: List[str] = Field(..., description="Common damage types for this part")
    detected_damage: str = Field(..., description="Detected damage type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class SeverityAnalysis(BaseModel):
    """Damage severity analysis."""
    severity_level: str = Field(..., description="Severity level (minimal, minor, moderate, severe)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    repair_action: str = Field(..., description="Recommended repair action")
    urgency: str = Field(..., description="Repair urgency level")
    estimated_cost_multiplier: float = Field(..., ge=0.0, description="Cost multiplier based on severity")


class DamageAssessment(BaseModel):
    """Overall damage assessment."""
    overall_severity: str = Field(..., description="Overall damage severity")
    total_estimated_cost: int = Field(..., ge=0, description="Total estimated repair cost")
    repair_priority: str = Field(..., description="Repair priority level")
    total_labor_hours: float = Field(..., ge=0.0, description="Total estimated labor hours")
    safety_impact: str = Field(..., description="Safety impact assessment")
    damage_count: int = Field(..., ge=0, description="Number of damage types detected")


class RepairRecommendation(BaseModel):
    """Repair recommendation."""
    damage_type: str = Field(..., description="Type of damage")
    affected_part: Optional[str] = Field(..., description="Affected vehicle part")
    severity: str = Field(..., description="Damage severity")
    recommended_action: str = Field(..., description="Recommended action")
    urgency: str = Field(..., description="Urgency level")
    description: str = Field(..., description="Detailed description")
    estimated_cost: Dict[str, Any] = Field(..., description="Cost estimation details")


class InferenceTiming(BaseModel):
    """Inference timing information."""
    resnet_time: float = Field(..., ge=0.0, description="ResNet inference time (seconds)")
    yolo_time: float = Field(..., ge=0.0, description="YOLO inference time (seconds)")
    fusion_time: float = Field(..., ge=0.0, description="Prediction fusion time (seconds)")
    total_time: float = Field(..., ge=0.0, description="Total inference time (seconds)")


class ModelVersions(BaseModel):
    """Model version information."""
    resnet: Union[str, int] = Field(..., description="ResNet model version/epoch")
    yolo: Union[str, int] = Field(..., description="YOLO model version/epoch")


class ImageInfo(BaseModel):
    """Image metadata."""
    size: List[int] = Field(..., description="Image dimensions [width, height]")
    mode: str = Field(..., description="Image mode (RGB, RGBA, etc.)")


class PredictionResponse(BaseModel):
    """Prediction response model."""
    # Core predictions
    predictions: Dict[str, ClassPrediction] = Field(..., description="Individual class predictions")
    predicted_classes: List[str] = Field(..., description="List of predicted damage classes")
    
    # Detailed analysis (included if include_details=True)
    correction_codes: Optional[List[CorrectionCode]] = Field(None, description="Correction codes")
    parts_information: Optional[List[PartInformation]] = Field(None, description="Parts information")
    severity_analysis: Optional[Dict[str, SeverityAnalysis]] = Field(None, description="Severity analysis")
    damage_assessment: Optional[DamageAssessment] = Field(None, description="Overall damage assessment")
    repair_recommendations: Optional[List[RepairRecommendation]] = Field(None, description="Repair recommendations")
    
    # Model metadata
    model_versions: ModelVersions = Field(..., description="Model version information")
    pipeline_version: str = Field(..., description="Inference pipeline version")
    
    # Timing information (included if include_timing=True)
    inference_times: Optional[InferenceTiming] = Field(None, description="Timing information")
    
    # Image metadata
    image_info: ImageInfo = Field(..., description="Input image information")
    
    # Response metadata
    timestamp: float = Field(..., description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    images: List[ImageInput] = Field(..., min_items=1, max_items=10, description="List of input images")
    include_details: bool = Field(default=True, description="Include detailed analysis")
    include_timing: bool = Field(default=False, description="Include timing information")
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override confidence threshold"
    )
    
    @validator('images')
    def validate_batch_size(cls, v):
        """Validate batch size limits."""
        if len(v) > 10:
            raise ValueError("Maximum batch size is 10 images")
        return v


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse] = Field(..., description="Individual predictions")
    batch_size: int = Field(..., description="Number of images processed")
    total_processing_time: float = Field(..., description="Total batch processing time")
    average_time_per_image: float = Field(..., description="Average processing time per image")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    timestamp: float = Field(..., description="Batch response timestamp")


class ServiceStats(BaseModel):
    """Service statistics model."""
    total_inferences: int = Field(..., description="Total number of inferences performed")
    avg_resnet_time: float = Field(..., description="Average ResNet inference time")
    avg_yolo_time: float = Field(..., description="Average YOLO inference time")
    avg_fusion_time: float = Field(..., description="Average fusion time")
    avg_total_time: float = Field(..., description="Average total inference time")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    device: str = Field(..., description="Inference device")
    model_versions: ModelVersions = Field(..., description="Model versions")
    uptime_seconds: float = Field(..., description="Service uptime")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    gpu_usage_percent: Optional[float] = Field(None, description="GPU usage percentage")


# Example usage and validation
if __name__ == "__main__":
    # Test the models with sample data
    
    # Test image input
    sample_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    try:
        image_input = ImageInput(image_data=sample_image, image_format="PNG")
        print("✓ ImageInput validation passed")
    except Exception as e:
        print(f"✗ ImageInput validation failed: {e}")
    
    # Test prediction request
    try:
        request = PredictionRequest(
            image=ImageInput(image_data=sample_image, image_format="PNG"),
            include_details=True,
            confidence_threshold=0.7
        )
        print("✓ PredictionRequest validation passed")
    except Exception as e:
        print(f"✗ PredictionRequest validation failed: {e}")
    
    # Test class prediction
    try:
        class_pred = ClassPrediction(
            class_name="front_bumper_damage",
            fused_score=0.85,
            resnet_score=0.82,
            yolo_score=0.85,
            predicted=True,
            threshold=0.5,
            fusion_method="max_rule"
        )
        print("✓ ClassPrediction validation passed")
    except Exception as e:
        print(f"✗ ClassPrediction validation failed: {e}")
    
    print("Schema validation tests completed!")