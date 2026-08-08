import json
import logging
import sys
from app import app

# Set up logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAPI")

def test_prediction():
    with app.test_client() as client:
        # Load one of the synthetic images we generated
        image_path = 'data/processed/IAM/words/test/word_0.png'
        
        try:
            with open(image_path, 'rb') as img:
                logger.info(f"Sending POST /predict with image {image_path}")
                response = client.post(
                    '/predict',
                    data={'file': (img, 'test_image.png')}
                )
                
                logger.info(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = json.loads(response.data)
                    logger.info("Prediction JSON Response:")
                    logger.info(json.dumps(data, indent=2))
                    
                    if data.get('confidence', 0) > 0 and len(data.get('recognized_text', '')) > 0:
                        logger.info("SUCCESS: Meaningful prediction and confidence > 0%")
                    else:
                        logger.error("FAILED: Prediction is empty or confidence is 0")
                else:
                    logger.error(f"FAILED: API returned {response.status_code} - {response.data}")
        except FileNotFoundError:
            logger.error(f"Image {image_path} not found.")

if __name__ == '__main__':
    test_prediction()
