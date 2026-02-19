import boto3
import json
import base64
import time
import uuid
import os
import logging
from random import randint
from datetime import datetime

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
# Bedrock client (SAME pattern as your working Nova Lambda)
# -------------------------------------------------------------------
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_cors_headers():
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Access-Control-Allow-Headers,Access-Control-Allow-Origin,Content-Type,Authorization',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
    }

def calculate_base64_size(base64_string):
    try:
        if ',' in base64_string:
            base64_data = base64_string.split(',')[1]
        else:
            base64_data = base64_string

        padding = base64_data.count('=')
        size = (len(base64_data) * 3 // 4) - padding
        return size
    except Exception:
        return 0

def calculate_output_images_size(images):
    total_size = 0
    if images:
        for image in images:
            total_size += calculate_base64_size(image)
    return total_size

def log_to_dynamodb(
    request_id,
    model_id,
    prompt,
    mode,
    image_size,
    mask_size,
    output_size,
    generation_time_ms,
    success=True,
    error_message=None
):
    try:
        dynamodb = boto3.resource('dynamodb')
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'ImageGenerationTable')
        table = dynamodb.Table(table_name)

        item = {
            'id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'model_id': model_id,
            'prompt': prompt[:1000],
            'mode': mode,
            'image_base64_size_bytes': image_size,
            'mask_base64_size_bytes': mask_size,
            'output_images_size_bytes': output_size,
            'generation_time_ms': generation_time_ms,
            'success': success
        }

        if error_message:
            item['error_message'] = error_message[:500]

        table.put_item(Item=item)
    except Exception as e:
        logger.error(f"Failed to log to DynamoDB: {str(e)}")

def prepare_titan_request(prompt_content, painting_mode, mask_base64, image_base64):
    image_generation_config = {
        "taskType": painting_mode,
        "imageGenerationConfig": {
            "numberOfImages": 2,
            "quality": "premium",
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": randint(0, 100000)
        }
    }

    params = {
        "image": image_base64,
        "text": prompt_content,
        "maskImage": mask_base64
    }

    if painting_mode == 'OUTPAINTING':
        params['outPaintingMode'] = 'DEFAULT'
        image_generation_config['outPaintingParams'] = params
    elif painting_mode == 'precise-outpaint':
        params['outPaintingMode'] = 'PRECISE'
        image_generation_config['outPaintingParams'] = params
    else:
        image_generation_config['inPaintingParams'] = params

    return json.dumps(image_generation_config)

# -------------------------------------------------------------------
# Lambda Handler
# -------------------------------------------------------------------
def lambda_handler(event, context):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"Received event: {json.dumps(event)}")

    # -------------------------------
    # Parse request body
    # -------------------------------
    try:
        if 'body' not in event:
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': 'Missing body in request'})
            }

        if isinstance(event['body'], dict):
            body = event['body']
        else:
            body = json.loads(event['body'])

    except Exception as e:
        return {
            'statusCode': 400,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': f'Invalid request body: {str(e)}'})
        }

    # -------------------------------
    # Extract parameters
    # -------------------------------
    try:
        prompt_content = body['prompt']['text']
        painting_mode = body['prompt']['mode']

        mask_base64 = body['mask'].split(",")[1] if "," in body['mask'] else body['mask']
        image_base64 = body['base_image'].split(",")[1] if "," in body['base_image'] else body['base_image']

    except Exception as e:
        return {
            'statusCode': 400,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': f'Missing or invalid parameters: {str(e)}'})
        }

    model = body.get('model', 'titan').lower()

    image_size = calculate_base64_size(body['base_image'])
    mask_size = calculate_base64_size(body['mask'])

    if model != 'titan':
        error_msg = f'Unsupported model: {model}'
        log_to_dynamodb(
            request_id,
            'unknown',
            prompt_content,
            painting_mode,
            image_size,
            mask_size,
            0,
            int((time.time() - start_time) * 1000),
            success=False,
            error_message=error_msg
        )
        return {
            'statusCode': 400,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': error_msg})
        }

    # -------------------------------
    # Prepare Titan request
    # -------------------------------
    try:
        request_body = prepare_titan_request(
            prompt_content,
            painting_mode,
            mask_base64,
            image_base64
        )
        model_id = "amazon.titan-image-generator-v2:0"
    except Exception as e:
        return {
            'statusCode': 400,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': f'Error preparing request: {str(e)}'})
        }

    # -------------------------------
    # Invoke Bedrock
    # -------------------------------
    try:
        bedrock_start = time.time()

        response = bedrock.invoke_model(
            modelId=model_id,
            body=request_body,
            contentType="application/json",
            accept="application/json"
        )

        generation_time_ms = int((time.time() - bedrock_start) * 1000)

        response_body = json.loads(response['body'].read())
        images = response_body.get('images')

        output_size = calculate_output_images_size(images)

        log_to_dynamodb(
            request_id,
            model_id,
            prompt_content,
            painting_mode,
            image_size,
            mask_size,
            output_size,
            generation_time_ms,
            success=True
        )

        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'images': images,
                'model_used': model,
                'request_id': request_id,
                'generation_time_ms': generation_time_ms
            })
        }

    except Exception as e:
        error_message = str(e)
        generation_time_ms = int((time.time() - start_time) * 1000)

        log_to_dynamodb(
            request_id,
            model_id,
            prompt_content,
            painting_mode,
            image_size,
            mask_size,
            0,
            generation_time_ms,
            success=False,
            error_message=error_message
        )

        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': error_message,
                'request_id': request_id
            })
        }
