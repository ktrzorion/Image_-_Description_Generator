import io
import os
import torch
import requests
import openai
from dotenv import load_dotenv
import base64
from fastapi import HTTPException
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from langchain_openai import OpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# API Key and Model Initialization
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image Preprocessing
def preprocess_image(image):
    """Preprocess image for CLIP model"""
    inputs = clip_processor(images=image, return_tensors="pt")
    return inputs

def image_to_base64(image):
    """Convert PIL Image to base64 encoded string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Extract Image Features
def extract_image_features(image):
    """Extract features from image using CLIP"""
    inputs = preprocess_image(image)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features

# Generate Base Description
def generate_base_description(image_features):
    """Generate initial description using OpenAI"""
    try:
        prompt = "Name and describe this food item in an attractive and detailed way."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful food description assistant."},
                {"role": "user", "content": f"{prompt} Based on the image's key features: {image_features.tolist()[:10]}"}
            ],
            max_tokens=50,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# Enhance Description
def enhance_description(base_description):
    """Enhance the base description"""
    try:
        prompt = f"Make this description more appealing: '{base_description}'"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative food description enhancer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


def generate_description(image):
    """Generate concise description using OpenAI's vision capabilities"""
    try:
        # Convert image to base64
        base64_image = image_to_base64(image)
        
        # Use GPT-4 Vision for more accurate image understanding
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at creating concise, engaging two-line descriptions of food and beverages."
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": "Create a precise two-line description that fully completes each sentence. Focus on the most striking visual and sensory aspects of the item."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        # First description (more objective)
        base_description = response.choices[0].message.content.strip()
        
        # Enhance description
        enhanced_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a creative copywriter who refines descriptions to be more appealing and complete."
                },
                {
                    "role": "user", 
                    "content": f"Refine this description to be more engaging while ensuring each sentence is complete and impactful: '{base_description}'"
                }
            ],
            max_tokens=100,
            temperature=0.8
        )
        
        enhanced_description = enhanced_response.choices[0].message.content.strip()
        
        return {
            "base_description": base_description,
            "enhanced_description": enhanced_description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    
def generate_dalle_image(description: str, num_images: int = 1):
    """
    Generate images using DALL-E via LangChain
    
    Args:
        description (str): Detailed description for image generation
        num_images (int): Number of images to generate
    
    Returns:
        list: List of generated image URLs
    """
    try:
        # Initialize LLM for prompt enhancement
        llm = OpenAI(temperature=0.9)
        
        # Create prompt template to enhance image description
        prompt = PromptTemplate(
            input_variables=["image_desc"],
            template="Concisely generate a creative DALL-E prompt based on: {image_desc}. Keep it under 900 characters.",
        )
        
        # Create chain to enhance description
        chain = (prompt | llm)
        
        # Enhance the original description and truncate
        enhanced_prompt = chain.invoke({"image_desc": description})[:900]
        
        # Generate images using DALL-E
        dalle = DallEAPIWrapper()
        image_urls = []
        
        for _ in range(num_images):
            image_url = dalle.run(enhanced_prompt)
            image_urls.append(image_url)
        
        return image_urls
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating image: {str(e)}"
        )

def download_image(image_url: str) -> Image.Image:
    """
    Download image from URL and convert to PIL Image
    
    Args:
        image_url (str): URL of the image to download
    
    Returns:
        PIL.Image: Downloaded image
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")
    