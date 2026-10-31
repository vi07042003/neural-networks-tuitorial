from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from rembg import remove, new_session
import asyncio
import uuid
import traceback

router = APIRouter(prefix="/preprocess", tags=["Preprocess"])

cpu_session = new_session("u2netp")

@router.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    try:
        input_image_bytes = await file.read()
        output_image_bytes = await asyncio.to_thread(remove, input_image_bytes, session=cpu_session)

        unique_filename = f"{uuid.uuid4()}_nobg.png"
        
        return Response(
            content=output_image_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{unique_filename}"'
            }
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))     