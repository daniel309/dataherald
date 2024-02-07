from fastapi import APIRouter, Depends, Security, status
from fastapi.security import HTTPBearer

from modules.finetuning.models.requests import FinetuningRequest
from modules.finetuning.models.responses import (
    ACFinetuningResponse,
    FinetuningResponse,
)
from modules.finetuning.service import FinetuningService
from modules.organization.invoice.models.entities import UsageType
from modules.organization.invoice.service import InvoiceService
from utils.auth import Authorize, VerifyToken, get_api_key
from utils.validation import ObjectIdString

router = APIRouter(
    prefix="/api/finetunings",
    responses={404: {"description": "Not found"}},
)

ac_router = APIRouter(
    prefix="/finetunings",
    responses={404: {"description": "Not found"}},
)

token_auth_scheme = HTTPBearer()
authorize = Authorize()
finetuning_service = FinetuningService()
invoice_service = InvoiceService()


@router.get("")
async def get_finetuning_jobs(
    db_connection_id: str = None,
    api_key: str = Security(get_api_key),
) -> list[FinetuningResponse]:
    return await finetuning_service.get_finetuning_jobs(
        api_key.organization_id, db_connection_id
    )


@router.get("/{id}")
async def get_finetuning_job(
    id: ObjectIdString,
    api_key: str = Security(get_api_key),
) -> FinetuningResponse:
    return await finetuning_service.get_finetuning_job(id, api_key.organization_id)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_finetuning_job(
    finetuning_request: FinetuningRequest,
    api_key: str = Security(get_api_key),
) -> FinetuningResponse:
    usage_type = (
        UsageType.FINETUNING_GPT_4
        if finetuning_service.is_gpt_4_model(finetuning_request.base_llm.model_name)
        else UsageType.FINETUNING_GPT_35
    )
    invoice_service.check_usage(
        api_key.organization_id,
        usage_type,
        len(finetuning_request.golden_records),
    )
    response = await finetuning_service.create_finetuning_job(
        finetuning_request, api_key.organization_id
    )
    invoice_service.record_usage(
        api_key.organization_id,
        usage_type,
        len(finetuning_request.golden_records),
        description=f"finetuning: {response.id}",
    )
    return response


@router.post("/{id}/cancel")
async def cancel_finetuning_job(
    id: ObjectIdString,
    api_key: str = Security(get_api_key),
) -> FinetuningResponse:
    return await finetuning_service.cancel_finetuning_job(id, api_key.organization_id)


@ac_router.get("")
async def ac_get_finetuning_jobs(
    db_connection_id: str = None,
    token: str = Depends(token_auth_scheme),
) -> list[ACFinetuningResponse]:
    user = authorize.user(VerifyToken(token.credentials).verify())
    return await finetuning_service.get_finetuning_jobs(
        user.organization_id, db_connection_id
    )


@ac_router.get("/{id}")
async def ac_get_finetuning_job(
    id: ObjectIdString, token: str = Depends(token_auth_scheme)
) -> ACFinetuningResponse:
    user = authorize.user(VerifyToken(token.credentials).verify())
    return await finetuning_service.get_finetuning_job(id, user.organization_id)


@ac_router.post("", status_code=status.HTTP_201_CREATED)
async def ac_create_finetuning_job(
    finetuning_request: FinetuningRequest,
    token: str = Depends(token_auth_scheme),
) -> ACFinetuningResponse:
    user = authorize.user(VerifyToken(token.credentials).verify())
    usage_type = (
        UsageType.FINETUNING_GPT_4
        if finetuning_service.is_gpt_4_model(finetuning_request.base_llm.model_name)
        else UsageType.FINETUNING_GPT_35
    )
    invoice_service.check_usage(
        user.organization_id,
        usage_type,
        len(finetuning_request.golden_records),
    )
    response = await finetuning_service.create_finetuning_job(
        finetuning_request, user.organization_id
    )
    invoice_service.record_usage(
        user.organization_id,
        usage_type,
        len(finetuning_request.golden_records),
        description=f"finetuning: {response.id}",
    )
    return response


@ac_router.post("/{id}/cancel")
async def ac_cancel_finetuning_job(
    id: ObjectIdString,
    token: str = Depends(token_auth_scheme),
) -> ACFinetuningResponse:
    user = authorize.user(VerifyToken(token.credentials).verify())
    return await finetuning_service.cancel_finetuning_job(id, user.organization_id)
