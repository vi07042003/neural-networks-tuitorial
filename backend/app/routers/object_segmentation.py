from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
import uuid
import traceback

router = APIRouter(prefix="/vision", tags=["Object Detection"])

model = YOLO("yolo11n-seg.pt")


@router.post("/detect-and-segment")
async def detect_and_segment(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg", "image/png", "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WebP image"
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty"
            )

        image = Image.open(BytesIO(contents)).convert("RGB")
        image_array = np.array(image)

        result = model.predict(
            source=image_array,
            conf=0.25,
            verbose=False
        )[0]

        if result.masks is None or len(result.boxes) == 0:
            raise HTTPException(
                status_code=404,
                detail="No objects detected"
            )

        mask = result.masks.data[0].cpu().numpy()

        mask_image = Image.fromarray(
            (mask * 255).astype(np.uint8)
        ).resize(image.size, Image.Resampling.NEAREST)

        mask_array = np.array(mask_image) > 127

        rgba = np.dstack([
            image_array,
            mask_array.astype(np.uint8) * 255
        ])

        output = BytesIO()
        Image.fromarray(rgba, "RGBA").save(output, format="PNG")

        filename = f"{uuid.uuid4()}_segmented.png"

        return Response(
            content=output.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition":
                    f'attachment; filename="{filename}"'
            }
        )

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )