from pathlib import Path

import numpy as np
import yaml

from bxi_example_py_elf3.framework.inference import PolicyOutput
from bxi_example_py_elf3.framework.joints import JointTargetBuffer
from bxi_example_py_elf3.framework.mod_api import ResourceKey, StateBuildContext
from bxi_example_py_elf3.framework.mod_api.transition import MotorFrame
from bxi_example_py_elf3.framework.runtime.mod_loader import (
    _discover_mods,
    _load_definition,
    _remove_module_prefixes,
    load_process_node_spec,
)
from bxi_example_py_elf3.framework.runtime.resource_manager import (
    ResourceManager,
)
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS


MOD_ROOT = Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic"
BASIC_ACTIONS_ROOT = MOD_ROOT.parent / "com.bxi.basic_actions"


def load_manifest():
    with (MOD_ROOT / "mod.yaml").open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_zerolab_nodes_have_distinct_upstream_and_mutually_exclusive_states():
    manifest = load_manifest()
    nodes = manifest["nodes"]
    source = nodes["zerolab_source"]["params"]
    assert set(nodes["pico_manager"]["states"]) == {"sonic_teleop"}
    assert set(nodes["smpl_bridge"]["states"]) == {"sonic_teleop"}
    assert set(nodes["zerolab_source"]["states"]) == {"sonic_zerolab"}
    assert set(nodes["zerolab_bridge"]["states"]) == {"sonic_zerolab"}
    assert nodes["zerolab_source"]["runtime"] == "python"
    assert nodes["zerolab_source"]["execution"] == "process"
    assert nodes["zerolab_source"]["runtime_profile"] == "host_ros"
    assert source["udp_port"] == 18000
    assert source["allowed_sender"] == "192.168.89.171"
    assert source["pose_host"] == "127.0.0.1"
    assert source["pose_port"] == 5558
    assert source["jitter_buffer_seconds"] == 0.04
    assert source["short_recovery_blend_seconds"] == 0.2
    assert source["stale_seconds"] == 0.5
    assert source["recovery_real_frames"] == 10
    assert (
        nodes["zerolab_bridge"]["entrypoint"]
        == "pico.pose_to_smpl_ref_bridge:create_node"
    )
    assert nodes["zerolab_bridge"]["depends_on"] == ["zerolab_source"]
    assert nodes["zerolab_bridge"]["params"]["pico_port"] == 5558
    assert nodes["zerolab_bridge"]["params"]["out_port"] == 5557
    assert nodes["smpl_bridge"]["params"]["out_port"] == 5557
    assert set(nodes["smpl_bridge"]["states"]).isdisjoint(
        nodes["zerolab_bridge"]["states"]
    )


def test_zerolab_event_state_and_routes_are_safe():
    manifest = load_manifest()
    assert manifest["events"]["activate"] == {
        "slot": "btn_10",
        "value": 9,
    }
    assert manifest["events"]["activate_zerolab"] == {
        "slot": "btn_10",
        "value": 11,
    }
    assert manifest["events"]["arm_zerolab"] == {
        "slot": "btn_10",
        "value": 12,
    }
    params = manifest["states"]["sonic_zerolab"]["params"]
    assert manifest["states"]["sonic_zerolab"]["manifest"][
        "confirm_message"
    ] == (
        "请先在ZeroLab厂家软件完成N-pose标定并回到中立姿势；"
        "进入后等待ZeroLab stream ready"
    )
    assert params["require_live_reference"] is False
    assert params["head_control_enabled"] is False
    assert params["hardware_gripper"] is False
    assert params["operator_prompt"] == (
        "机器人保持实时Normal直到stream ready；确认操作者处于中立姿势后，"
        "由安全员发送btn_10=12接管；再次发送可暂停并回到WAIT_ARM"
    )
    assert params["arm_blend_seconds"] == 2.0
    assert params["auto_rearm_on_recovery"] is True
    assert params["auto_rearm_blend_seconds"] == 2.0
    assert params["recovery_real_frames"] == 10
    routes = {(r["from"], r["event"], r["to"]) for r in manifest["routes"]}
    assert (
        "com.bxi.basic_actions/normal",
        "activate_zerolab",
        "sonic_zerolab",
    ) in routes
    assert (
        "sonic_zerolab",
        "com.bxi.basic_actions/normal",
        "com.bxi.basic_actions/normal",
    ) in routes
    forbidden = {
        ("sonic_teleop", "sonic_zerolab"),
        ("sonic_zerolab", "sonic_teleop"),
    }
    assert not any(
        (route["from"], route["to"]) in forbidden
        for route in manifest["routes"]
    )
    actions = {
        (item["from"], item["event"], item["action"])
        for item in manifest["actions"]
    }
    assert ("sonic_zerolab", "arm_zerolab", "arm_zerolab") in actions
    arm_action = next(
        item
        for item in manifest["actions"]
        if item["from"] == "sonic_zerolab"
        and item["event"] == "arm_zerolab"
    )
    assert arm_action["manifest"]["label"] == "ZeroLab ARM / 暂停"
    assert not any(
        source == "sonic_teleop" and event == "arm_zerolab"
        for source, event, _action in actions
    )


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class EntryLifecyclePolicy:
    def __init__(self):
        target = JointTargetBuffer(ELF3_POLICY_JOINTS)
        self.output = PolicyOutput(target.view)
        self.head_joint_target = np.zeros(2, dtype=np.float32)

    def bind_logger(self, _logger):
        pass

    def configure_runtime(self, **_kwargs):
        pass

    def reset(self, _frame=None):
        pass

    def has_fresh_live_reference(self, _timeout_s=None):
        return False


class EntryLifecycleHandle:
    status = "ready"

    def __init__(self):
        self.policy = EntryLifecyclePolicy()

    def get(self):
        return self.policy


class EntryLifecycleContext:
    robot_layout = ELF3_POLICY_JOINTS
    inference_frame = None

    def __init__(self):
        self.last_motor_frame = MotorFrame.empty(self.robot_layout)


def test_source_prompts_and_zerolab_availability_without_live_data():
    resources = ResourceManager()
    module_prefixes = []
    try:
        discovered = _discover_mods((BASIC_ACTIONS_ROOT, MOD_ROOT))
        basic_definition, basic_module = _load_definition(
            discovered["com.bxi.basic_actions"], resources
        )
        module_prefixes.append(basic_module.__name__.split(".", 1)[0])
        definition, module = _load_definition(
            discovered["com.bxi.sonic"], resources
        )
        module_prefixes.append(module.__name__.split(".", 1)[0])
        policy_key = ResourceKey[object]("com.bxi.sonic/policy")
        assert resources.status(policy_key) == "unloaded"
        assert set(definition.state_factories) == {
            "sonic_teleop",
            "sonic_zerolab",
        }

        pico_context = StateBuildContext("com.bxi.sonic/sonic_teleop", 1, {})
        pico = definition.state_factories["sonic_teleop"](pico_context)
        pico_context.finish()
        assert type(pico).__name__ == "SonicTeleopState"
        assert pico.operator_prompt == (
            "PICO同时按住A+B+X+Y请求校准，再按A+X切入实时POSE"
        )

        normal_context = StateBuildContext(
            "com.bxi.basic_actions/normal", 0, {}
        )
        normal = basic_definition.state_factories["normal"](normal_context)
        normal_context.finish()

        prompt = (
            "机器人保持实时Normal直到stream ready；确认操作者处于中立姿势后，"
            "由安全员发送btn_10=12"
        )
        zero_context = StateBuildContext(
            "com.bxi.sonic/sonic_zerolab",
            2,
            {
                "operator_prompt": prompt,
                "require_live_reference": False,
                "head_control_enabled": False,
                "hardware_gripper": False,
                "arm_blend_seconds": 2.0,
                "auto_rearm_on_recovery": True,
                "auto_rearm_blend_seconds": 2.0,
                "recovery_real_frames": 10,
            },
        )
        zero = definition.state_factories["sonic_zerolab"](zero_context)
        zero_context.finish()
        sonic_policy_id = "com.bxi.sonic/policy"
        normal_policy_id = "com.bxi.basic_actions/normal_policy"
        assert [handle.key.id for handle in pico.required_resources] == [
            sonic_policy_id
        ]
        assert [handle.key.id for handle in zero.required_resources] == [
            sonic_policy_id,
            normal_policy_id,
        ]
        assert zero._normal_policy.key.id == normal_policy_id
        assert zero._normal_policy.key == normal._policy.key
        assert type(zero).__name__ == "ZeroLabArmedTeleopState"
        assert zero.arm_blend_seconds == 2.0
        assert zero.auto_rearm_on_recovery is True
        assert zero.auto_rearm_blend_seconds == 2.0
        assert zero.recovery_real_frames == 10
        assert zero._policy is pico._policy
        assert zero.require_live_reference is False
        assert zero.hardware_gripper is False
        assert zero.is_available(None) is True
        logger = CaptureLogger()
        zero._bind_logger(logger)
        zero._policy = EntryLifecycleHandle()
        zero.on_prepare(EntryLifecycleContext(), object())
        zero.on_enter(None)
        assert logger.messages == [
            "ZeroLab ARM phase: WAIT_STREAM",
            "ZeroLab pre-ARM output: live zero-command Normal policy",
            "SONIC遥操已启动；头部跟踪已关闭；" + prompt,
        ]
    finally:
        resources.close()
        _remove_module_prefixes(tuple(module_prefixes))


def test_process_loader_imports_zerolab_source_with_dynamic_package():
    spec, module_prefix = load_process_node_spec(
        MOD_ROOT / "mod.yaml", "zerolab_source"
    )
    try:
        assert callable(spec.factory)
        assert spec.execution == "process"
        assert spec.states == ("com.bxi.sonic/sonic_zerolab",)
        assert spec.params["udp_port"] == 18000
        assert spec.params["pose_port"] == 5558
    finally:
        _remove_module_prefixes((module_prefix,))


def test_zerolab_sources_and_manifest_avoid_runtime_tpose_calibration():
    legacy_terms = ("TPoseCalibrator", "T-pose标定")
    for relative_path in (
        "zerolab/converter.py",
        "zerolab/source_node.py",
        "zerolab/state.py",
    ):
        source = (MOD_ROOT / relative_path).read_text(encoding="utf-8")
        assert not any(term in source for term in legacy_terms)

    zerolab_manifest = yaml.safe_dump(
        load_manifest()["states"]["sonic_zerolab"],
        allow_unicode=True,
    )
    assert not any(term in zerolab_manifest for term in legacy_terms)
