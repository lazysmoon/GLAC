from typing import Optional

from .base import MultiAgentEnv
from .dubins_car import DubinsCar

ENV = {
    'DubinsCar': DubinsCar,
}


DEFAULT_MAX_STEP = 256


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
        dt: Optional[float] = None,
        n_substeps: Optional[int] = None,
        r_c_params: Optional[dict] = None,
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = ENV[env_id].PARAMS
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs
    if n_rays is not None:
        params['n_rays'] = n_rays
    kwargs = dict(
        num_agents=num_agents,
        area_size=area_size,
        max_step=max_step,
        max_travel=max_travel,
        dt=dt if dt is not None else 0.02,
        params=params,
        r_c_params=r_c_params,
    )
    if n_substeps is not None:
        kwargs['n_substeps'] = n_substeps
    return ENV[env_id](**kwargs)
