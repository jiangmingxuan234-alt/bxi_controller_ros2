from __future__ import annotations

from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
    StateBuildContext,
)

from .reference import LiveReferenceReceiver
from .rgmt_policy import RgmtExternalReferencePolicy
from .state import PicoGmrMotionParams, PicoGmrMotionState


MODEL_ASSET = "assets/rgmt.onnx"


def _build_state(context: ModLoadContext, state: StateBuildContext) -> PicoGmrMotionState:
    params = state.dataclass_params(PicoGmrMotionParams)
    policy_key = ResourceKey[RgmtExternalReferencePolicy](f"{state.name}/policy")
    receiver_key = ResourceKey[LiveReferenceReceiver](f"{state.name}/reference")

    def load_policy(load_context: ResourceLoadContext) -> RgmtExternalReferencePolicy:
        return RgmtExternalReferencePolicy.for_live_reference(
            str(load_context.asset(MODEL_ASSET)),
            reference_yaw_mode=params.reference_yaw_mode,
            backend=params.backend,
        )

    def load_receiver(_load_context: ResourceLoadContext) -> LiveReferenceReceiver:
        return LiveReferenceReceiver(params.host, params.port)

    context.register_resource(policy_key, load_policy, policy="on_demand")
    context.register_resource(receiver_key, load_receiver, policy="on_demand")
    return PicoGmrMotionState(
        state.name,
        state.state_id,
        policy=context.resource(policy_key),
        receiver=context.resource(receiver_key),
        params=params,
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    return ModDefinition(
        state_factories={
            "pico_gmr_motion": lambda state: _build_state(context, state),
        }
    )


__all__ = [
    "MODEL_ASSET",
    "create_mod",
]
