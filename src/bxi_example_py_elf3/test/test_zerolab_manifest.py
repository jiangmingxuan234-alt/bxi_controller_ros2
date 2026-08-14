from pathlib import Path

import yaml

from bxi_example_py_elf3.framework.mod_api import ResourceKey, StateBuildContext
from bxi_example_py_elf3.framework.runtime.mod_loader import (
    _discover_mods,
    _load_definition,
    _remove_module_prefixes,
    load_process_node_spec,
)
from bxi_example_py_elf3.framework.runtime.resource_manager import (
    ResourceManager,
)


MOD_ROOT = Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic"


def load_manifest():
    with (MOD_ROOT / "mod.yaml").open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_zerolab_nodes_have_distinct_upstream_and_mutually_exclusive_states():
    manifest = load_manifest()
    nodes = manifest["nodes"]
    assert set(nodes["pico_manager"]["states"]) == {"sonic_teleop"}
    assert set(nodes["smpl_bridge"]["states"]) == {"sonic_teleop"}
    assert set(nodes["zerolab_source"]["states"]) == {"sonic_zerolab"}
    assert set(nodes["zerolab_bridge"]["states"]) == {"sonic_zerolab"}
    assert nodes["zerolab_source"]["runtime"] == "python"
    assert nodes["zerolab_source"]["execution"] == "process"
    assert nodes["zerolab_source"]["runtime_profile"] == "host_ros"
    assert nodes["zerolab_source"]["params"]["udp_port"] == 18000
    assert nodes["zerolab_source"]["params"]["allowed_sender"] == (
        "192.168.89.171"
    )
    assert nodes["zerolab_source"]["params"]["pose_host"] == "127.0.0.1"
    assert nodes["zerolab_source"]["params"]["pose_port"] == 5558
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
    params = manifest["states"]["sonic_zerolab"]["params"]
    assert params["require_live_reference"] is False
    assert params["head_control_enabled"] is False
    assert params["hardware_gripper"] is False
    assert params["operator_prompt"] == (
        "请保持T-pose，直到ZeroLab stream ready后再开始动作"
    )
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


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


def test_source_prompts_and_zerolab_availability_without_live_data():
    resources = ResourceManager()
    module_prefix = None
    try:
        discovered = _discover_mods((MOD_ROOT,))
        definition, module = _load_definition(
            discovered["com.bxi.sonic"], resources
        )
        module_prefix = module.__name__.split(".", 1)[0]
        policy_key = ResourceKey[object]("com.bxi.sonic/policy")
        assert resources.status(policy_key) == "unloaded"
        assert set(definition.state_factories) == {
            "sonic_teleop",
            "sonic_zerolab",
        }

        pico_context = StateBuildContext("com.bxi.sonic/sonic_teleop", 1, {})
        pico = definition.state_factories["sonic_teleop"](pico_context)
        pico_context.finish()
        assert pico.operator_prompt == (
            "PICO同时按住A+B+X+Y请求校准，再按A+X切入实时POSE"
        )

        prompt = "请保持T-pose，直到ZeroLab stream ready后再开始动作"
        zero_context = StateBuildContext(
            "com.bxi.sonic/sonic_zerolab",
            2,
            {
                "operator_prompt": prompt,
                "require_live_reference": False,
                "head_control_enabled": False,
                "hardware_gripper": False,
            },
        )
        zero = definition.state_factories["sonic_zerolab"](zero_context)
        zero_context.finish()
        assert zero._policy is pico._policy
        assert zero.require_live_reference is False
        assert zero.hardware_gripper is False
        assert zero.is_available(None) is True
        logger = CaptureLogger()
        zero._bind_logger(logger)
        zero.on_enter(None)
        assert logger.messages == [
            "SONIC遥操已启动；头部跟踪已关闭；" + prompt
        ]
    finally:
        resources.close()
        if module_prefix is not None:
            _remove_module_prefixes((module_prefix,))


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
