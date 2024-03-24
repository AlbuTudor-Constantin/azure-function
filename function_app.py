# Source code for the function app. This function runs a custom vectorizer for text-to-image queries. 
# It's also a custom skill that vectorizes images from a blob indexer.
import azure.functions as func
import json
import logging
import os
import requests
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
app = func.FunctionApp()

# Doubles as a custom skill and a custom vectorizer
@app.function_name(name="GetImageEmbedding")
@app.route(route="GetImageEmbedding")
def GetImageEmbedding(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request")

    req_body = req.get_body().decode('uft-8')
    logging.info(f"Request body: {req_body}")
    request = json.loads(req_body)
    values = request['values']

    response_values = []
    for value in values:
        imageUrl = value['data']['imageUrl']
        recordId = value['recordId']
        logging.info(f"Input imageUrl: {imageUrl}")
        logging.info(f"Input recordId: {recordId}")

        vector = get_image_embeddings(imageUrl)

        response_values.append({
            "recordId": recordId,
            "data": {
                "vector": vector
            },
            "errors": None,
            "warnings": None
        })

    response_body = {
        "values": response_values
    }
    logging.info(f"Response body: {response_body}")

    return func.HttpResponse(json.dumps(response_body), mimetypes="application/json")


def get_image_embeddings(imageUrl):
    cogSvcsEndpoint = "https://lctwhite.cognitiveservices.azure.com/"
    cogSvcsKey = "cd25aa7cd1d14f02ad7403140b2c38ab"

    url = f"{cogSvcsEndpoint}/computervision/retrieval:vectorizeImage"

    params = {
        "api-version": "2023-02-01-preview"
    }

    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": cogSvcsKey
    }

    data = {
        "url": imageUrl
    }

    response = requests.post(url, params=params, headers=headers, json=data)

    if response.status_code != 200:
        logging.error(f"Error {response.status_code}, {response.text}")
        response.raise_for_status()

    embeddings = response.json()["vector"]

    return embeddings
