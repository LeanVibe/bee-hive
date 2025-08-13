from fastapi import APIRouter
from fastapi import Depends

from ..core.auth import RefreshRequest
from ..core.auth import TokenResponse
from ..core.auth import User
from ..core.auth import UserLogin
from ..core.auth import UserResponse
from ..core.auth import get_current_user
from ..core.auth import get_current_user_info as _get_current_user_info
from ..core.auth import login_user as _login_user
from ..core.auth import logout_user as _logout_user
from ..core.auth import refresh_access_token as _refresh_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin) -> TokenResponse:
    return await _login_user(login_data)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return await _get_current_user_info(current_user)  # type: ignore[arg-type]


@router.post("/refresh")
async def refresh(payload: RefreshRequest) -> dict:
    return await _refresh_access_token(payload)


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)) -> dict:
    return await _logout_user(current_user)  # type: ignore[arg-type]
