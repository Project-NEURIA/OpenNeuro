import traceback
from http import HTTPStatus

from kui.asgi import HTTPException, JSONResponse, request
import ormsgpack

class ExceptionHandler:

    async def http_exception_handler(self, exc: HTTPException):
        accept = request.headers.get("accept", "")
        content_type = request.headers.get("content-type", "")
        
        err_data = dict(
            statusCode=exc.status_code,
            message=str(exc.content),
            error=HTTPStatus(exc.status_code).phrase,
        )

        if "application/msgpack" in accept or "application/msgpack" in content_type:
            from kui.asgi import Response
            return Response(
                ormsgpack.packb(err_data),
                status_code=exc.status_code,
                headers={"content-type": "application/msgpack"},
            )

        return JSONResponse(err_data, exc.status_code, exc.headers)

    async def other_exception_handler(self, exc: Exception):
        traceback.print_exc()

        status = HTTPStatus.INTERNAL_SERVER_ERROR
        return JSONResponse(
            dict(statusCode=status, message=str(exc), error=status.phrase),
            status,
        )
