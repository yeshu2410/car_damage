"""
FastAPI service for collision parts prediction inference.
Provides REST API endpoints for damage detection and classification.
"""

import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
import os
import psutil

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from loguru import logger

from .pipeline import InferencePipeline, create_pipeline
from .mapping import PredictionMapper, create_mapper
from .schemas import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, HealthCheckResponse, ServiceStats, 
    ErrorResponse, ClassPrediction, ModelVersions, ImageInfo,
    InferenceTiming, CorrectionCode, PartInformation, SeverityAnalysis,
    DamageAssessment, RepairRecommendation
)


class CollisionPartsService:
    """Collision parts prediction service."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the service."""
        self.config_path = config_path
        self.start_time = time.time()
        self.pipeline: Optional[InferencePipeline] = None
        self.mapper: Optional[PredictionMapper] = None
        self.service_stats = {
            "requests_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_request_time": 0.0
        }
        
        logger.info("Collision Parts Service initialized")
    
    async def startup(self):
        """Initialize models and components on startup."""
        try:
            logger.info("Starting service initialization...")
            
            # Initialize inference pipeline
            self.pipeline = create_pipeline(self.config_path)
            logger.info("Inference pipeline loaded")
            
            # Initialize prediction mapper
            self.mapper = create_mapper(self.config_path)
            logger.info("Prediction mapper loaded")
            
            logger.info("Service startup completed successfully")
            
        except Exception as e:
            logger.error(f"Service startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup on shutdown."""
        logger.info("Service shutting down...")
        # Add any cleanup logic here
        logger.info("Service shutdown completed")
    
    def get_health_status(self) -> HealthCheckResponse:
        """Get service health status."""
        models_loaded = {
            "resnet": self.pipeline.resnet_model is not None if self.pipeline else False,
            "yolo": self.pipeline.yolo_model is not None if self.pipeline else False
        }
        
        return HealthCheckResponse(
            status="healthy" if all(models_loaded.values()) else "unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            models_loaded=models_loaded,
            uptime_seconds=time.time() - self.start_time
        )
    
    def get_service_stats(self) -> ServiceStats:
        """Get detailed service statistics."""
        pipeline_stats = self.pipeline.get_stats() if self.pipeline else {}
        
        # Get system resource usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        return ServiceStats(
            total_inferences=pipeline_stats.get("total_inferences", 0),
            avg_resnet_time=pipeline_stats.get("avg_resnet_time", 0.0),
            avg_yolo_time=pipeline_stats.get("avg_yolo_time", 0.0),
            avg_fusion_time=pipeline_stats.get("avg_fusion_time", 0.0),
            avg_total_time=pipeline_stats.get("avg_total_time", 0.0),
            models_loaded=pipeline_stats.get("models_loaded", {}),
            device=pipeline_stats.get("device", "unknown"),
            model_versions=ModelVersions(
                resnet=pipeline_stats.get("model_versions", {}).get("resnet", "unknown"),
                yolo=pipeline_stats.get("model_versions", {}).get("yolo", "unknown")
            ),
            uptime_seconds=time.time() - self.start_time,
            memory_usage_mb=memory_mb
        )
    
    async def predict_single(self, request: PredictionRequest, 
                           request_id: str) -> PredictionResponse:
        """Process single prediction request."""
        start_time = time.time()
        
        try:
            # Convert image input to PIL Image
            image = request.image.to_pil_image()
            
            # Run inference pipeline
            pipeline_result = self.pipeline.predict(image)
            
            # Convert pipeline predictions to schema format
            predictions = {}
            for class_name, pred_data in pipeline_result["predictions"].items():
                predictions[class_name] = ClassPrediction(
                    class_name=class_name,
                    fused_score=pred_data["fused_score"],
                    resnet_score=pred_data["resnet_score"],
                    yolo_score=pred_data["yolo_score"],
                    predicted=pred_data["predicted"],
                    threshold=pred_data["threshold"],
                    fusion_method=pred_data["fusion_method"]
                )
            
            # Prepare base response
            response_data = {
                "predictions": predictions,
                "predicted_classes": pipeline_result["predicted_classes"],
                "model_versions": ModelVersions(
                    resnet=pipeline_result["model_versions"]["resnet"],
                    yolo=pipeline_result["model_versions"]["yolo"]
                ),
                "pipeline_version": pipeline_result["pipeline_version"],
                "image_info": ImageInfo(
                    size=list(pipeline_result["image_info"]["size"]),
                    mode=pipeline_result["image_info"]["mode"]
                ),
                "timestamp": pipeline_result["timestamp"],
                "request_id": request_id
            }
            
            # Add timing information if requested
            if request.include_timing:
                response_data["inference_times"] = InferenceTiming(
                    resnet_time=pipeline_result["inference_times"]["resnet_time"],
                    yolo_time=pipeline_result["inference_times"]["yolo_time"],
                    fusion_time=pipeline_result["inference_times"]["fusion_time"],
                    total_time=pipeline_result["inference_times"]["total_time"]
                )
            
            # Add detailed analysis if requested
            if request.include_details:
                # Map predictions to correction codes and parts
                mapping_result = self.mapper.map_predictions_to_codes(
                    pipeline_result["predictions"]
                )
                
                # Convert mapping results to schema format
                correction_codes = [
                    CorrectionCode(**code_data) 
                    for code_data in mapping_result["correction_codes"]
                ]
                
                parts_information = [
                    PartInformation(**part_data)
                    for part_data in mapping_result["parts_information"]
                ]
                
                severity_analysis = {
                    class_name: SeverityAnalysis(**severity_data)
                    for class_name, severity_data in mapping_result["severity_analysis"].items()
                }
                
                damage_assessment = DamageAssessment(**mapping_result["damage_assessment"])
                
                repair_recommendations = [
                    RepairRecommendation(**rec_data)
                    for rec_data in mapping_result["repair_recommendations"]
                ]
                
                response_data.update({
                    "correction_codes": correction_codes,
                    "parts_information": parts_information,
                    "severity_analysis": severity_analysis,
                    "damage_assessment": damage_assessment,
                    "repair_recommendations": repair_recommendations
                })
            
            # Update service statistics
            self._update_service_stats(time.time() - start_time, success=True)
            
            return PredictionResponse(**response_data)
            
        except Exception as e:
            self._update_service_stats(time.time() - start_time, success=False)
            logger.error(f"Prediction failed for request {request_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    async def predict_batch(self, request: BatchPredictionRequest,
                          request_id: str) -> BatchPredictionResponse:
        """Process batch prediction request."""
        start_time = time.time()
        
        predictions = []
        successful_count = 0
        failed_count = 0
        
        for i, image_input in enumerate(request.images):
            try:
                # Create individual prediction request
                individual_request = PredictionRequest(
                    image=image_input,
                    include_details=request.include_details,
                    include_timing=request.include_timing,
                    confidence_threshold=request.confidence_threshold
                )
                
                # Process prediction
                result = await self.predict_single(
                    individual_request, 
                    f"{request_id}_image_{i}"
                )
                predictions.append(result)
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Batch prediction failed for image {i}: {e}")
                failed_count += 1
                # Could add failed predictions to response if needed
        
        total_time = time.time() - start_time
        avg_time = total_time / len(request.images) if request.images else 0
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(request.images),
            total_processing_time=total_time,
            average_time_per_image=avg_time,
            successful_predictions=successful_count,
            failed_predictions=failed_count,
            timestamp=time.time()
        )
    
    def _update_service_stats(self, request_time: float, success: bool):
        """Update service statistics."""
        self.service_stats["requests_processed"] += 1
        
        if success:
            self.service_stats["successful_requests"] += 1
        else:
            self.service_stats["failed_requests"] += 1
        
        # Update average request time
        n = self.service_stats["requests_processed"]
        current_avg = self.service_stats["avg_request_time"]
        self.service_stats["avg_request_time"] = (
            (current_avg * (n - 1) + request_time) / n
        )


# Initialize service
service = CollisionPartsService()

# Create FastAPI app
app = FastAPI(
    title="Collision Parts Prediction API",
    description="AI-powered vehicle collision damage detection and parts prediction service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    await service.startup()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await service.shutdown()


# API Routes
@app.get("/healthz", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return service.get_health_status()


@app.get("/stats", response_model=ServiceStats)
async def get_stats():
    """Get service statistics."""
    return service.get_service_stats()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single image prediction endpoint."""
    request_id = str(uuid.uuid4())
    
    try:
        result = await service.predict_single(request, request_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    request_id = str(uuid.uuid4())
    
    try:
        result = await service.predict_batch(request, request_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in batch predict endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "Collision Parts Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/healthz",
        "stats": "/stats"
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            timestamp=datetime.now(),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"exception_type": type(exc).__name__},
            timestamp=datetime.now(),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the FastAPI server."""
    logger.info(f"Starting Collision Parts Prediction Service on {host}:{port}")
    
    uvicorn.run(
        "src.infer.service:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )


if __name__ == "__main__":
    # Development server
    run_server(host="127.0.0.1", port=8000, workers=1)