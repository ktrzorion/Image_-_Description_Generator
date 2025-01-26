import uvicorn
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from PIL import Image
from fastapi.responses import JSONResponse
from image_generator import generate_description, generate_dalle_image, download_image

# Initialize FastAPI app
app = FastAPI(
    title="Image & Description Generator",
    description="Generate appealing food images & their descriptions using CLIP and OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Model for Response
class ImageGenerationRequest(BaseModel):
    description: str
    num_images: int = 1

# Request model for food image generation
class FoodImageRequest(BaseModel):
    food_description: str
    num_images: int = 1

# Response model for food image generation
class FoodImageResponse(BaseModel):
    original_description: str
    generated_images: List[Dict[str, str]]

@app.post("/generate-description")
async def create_description(file: UploadFile = File(...)):
    """
    Generate descriptions for uploaded images
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Generate description
    descriptions = generate_description(image)
    
    return descriptions

@app.post("/generate-image")
async def create_image(request: ImageGenerationRequest):
    """
    Generate images based on text description
    
    - Accepts a description and optional number of images
    - Returns URLs of generated images
    """
    try:
        # Generate images
        image_urls = generate_dalle_image(
            description=request.description, 
            num_images=request.num_images
        )
        
        # Return response
        return {
            "description": request.description,
            "image_urls": image_urls
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={
                "error": str(e),
                "description": request.description
            }
        )

@app.post("/generate-food-images", response_model=FoodImageResponse)
async def generate_food_images(request: FoodImageRequest):
    """
    Generate food images and create detailed descriptions
    
    - **food_description**: Short description of the food item
    - **num_images**: Number of images to generate (default 1)
    
    Returns generated images with detailed descriptions
    """
    try:
        # Generate images using DALL-E
        image_urls = generate_dalle_image(
            description=request.food_description, 
            num_images=request.num_images
        )
        
        # Process each generated image
        generated_images = []
        for image_url in image_urls:
            # Download the image
            image = download_image(image_url)
            
            # Generate description for the image
            description = generate_description(image)
            
            # Prepare image details
            generated_images.append({
                "image_url": image_url,
                "base_description": description['base_description'],
                "enhanced_description": description['enhanced_description']
            })

        # Return response
        return FoodImageResponse(
            original_description=request.food_description,
            generated_images=generated_images
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)