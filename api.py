from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
import uuid
from datetime import datetime
from models import image_classify

app = FastAPI(
    title="Image Classification API",
    description="This API allows users to generate credentials and classify images. "
                "Credentials are required for accessing the classification endpoint and are valid for a limited time.",
    version="1.0.0"
)

UPLOAD_DIRECTORY = "./uploads"
DB = 'access.txt'
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


class CredentialResponse(BaseModel):
    credential: str
    timestamp: str
    message: str


class ClassificationResponse(BaseModel):
    filename: str
    classification: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    message: str


@app.get("/credentials/", response_model=CredentialResponse, summary="Generate Credentials",
         description="Generates a unique credential for accessing the classify endpoint. The credential is valid for a limited time.")
async def get_credentials():
    """
    Generates a unique credential and saves it with a timestamp.
    """
    credential = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    with open(DB, 'a') as f:
        f.write(f"{credential} {timestamp}\n")
    return JSONResponse(
        status_code=201,
        content={
            "credential": credential,
            "timestamp": timestamp,
            "message": "Credential created successfully. Use this credential to access the /classify/ endpoint."
        }
    )


@app.post("/classify/", response_model=ClassificationResponse, summary="Classify Image",
          description="Classifies an uploaded image. Requires a valid credential in the request header.",
          responses={
              200: {
                  "description": "Image classified successfully.",
                  "content": {
                      "application/json": {
                          "example": {
                              "filename": "example.jpg",
                              "classification": "cat",
                              "message": "Image classified successfully."
                          }
                      }
                  }
              },
              400: {
                  "description": "No image file sent.",
                  "content": {
                      "application/json": {
                          "example": {
                              "error": "Bad Request",
                              "message": "No image file sent. Please upload an image file."
                          }
                      }
                  }
              },
              401: {
                  "description": "Invalid credential.",
                  "content": {
                      "application/json": {
                          "example": {
                              "error": "Invalid credential",
                              "message": "The provided credential is not valid. Please obtain a valid credential from the /credentials/ endpoint."
                          }
                      }
                  }
              },
              500: {
                  "description": "Classification error.",
                  "content": {
                      "application/json": {
                          "example": {
                              "error": "Classification error",
                              "message": "An error occurred during image classification: <error details>"
                          }
                      }
                  }
              }
          })
async def classify_image(file: UploadFile = File(...), credential: str = Header(...)):
    """
    Classifies the uploaded image using a machine learning model.
    Requires a valid credential in the header.
    """
    if not is_valid_credential(credential):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid credential",
                "message": "The provided credential is not valid. Please obtain a valid credential from the /credentials/ endpoint."
            }
        )

    if not file:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Bad Request",
                "message": "No image file sent. Please upload an image file."
            }
        )

    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        classification = image_classify(file_location)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Classification error",
                "message": f"An error occurred during image classification: {str(e)}"
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "filename": file.filename,
            "classification": classification,
            "message": "Image classified successfully."
        }
    )


def is_valid_credential(credential: str) -> bool:
    """
    Checks if the provided credential is valid.
    """
    if not os.path.exists(DB):
        return False
    with open(DB, 'r') as f:
        credentials = f.read().splitlines()
    for line in credentials:
        parts = line.split()
        if len(parts) != 2:
            continue
        saved_credential, timestamp = parts
        if saved_credential == credential:
            return True
    return False
