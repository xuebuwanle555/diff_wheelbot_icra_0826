"""Evaluation environment adapter used by point-goal scripts."""

import json
import os

from src.training.scene_assets import find_usd_path
from src.utils import BatchMPCController


_SCENE_WRAPPER_VERSION = 2


class BatchMPCNEWController(BatchMPCController):
    """Compatibility alias for the legacy evaluation script."""

    def __init__(self, batch=1, **kwargs):
        super().__init__(batch=batch, **kwargs)


def namespace_to_dict(obj):
    """Convert argparse-style namespaces from eval YAML into plain dictionaries."""
    if hasattr(obj, "__dict__"):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    return obj


def find_eval_pointgoal_path(scene_pointgoal_dir: str) -> str | None:
    """Return a point-goal sample file accepted by evaluation."""
    preferred_names = (
        "pointgoal_start_goal_pairs.npy",
        "pointgoal_start_pair_samples.npy",
        "pointgoal_start_pair_samples_safe.npy",
    )
    for name in preferred_names:
        path = os.path.join(scene_pointgoal_dir, name)
        if os.path.isfile(path):
            return path

    if not os.path.isdir(scene_pointgoal_dir):
        return None
    candidates = [
        os.path.join(scene_pointgoal_dir, name)
        for name in sorted(os.listdir(scene_pointgoal_dir))
        if name.endswith(".npy")
        and "pointgoal" in name.lower()
    ]
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple evaluation point-goal files found in {scene_pointgoal_dir}: "
            f"{', '.join(os.path.basename(path) for path in candidates)}"
        )
    return candidates[0] if candidates else None


def _without_embedded_physics_scene(
    usd_path: str,
    *,
    scene_dir: str,
    scene_name: str,
    scene_type: str,
) -> str:
    """Return a tiny wrapper USD with embedded PhysicsScene prims disabled.

    Scene-N1 Home/Commercial USDs contain their own active PhysicsScene.  An
    IsaacLab environment also owns a PhysicsScene, and Isaac Sim 5.1 cannot
    step both when their scene stepping differs.  The wrapper keeps the
    released asset untouched: it references the original default prim and
    authors stronger ``active = false`` opinions only for PhysicsScene prims.
    """
    if scene_type not in ("home", "commercial"):
        return usd_path

    # This module is imported after AppLauncher creates SimulationApp, so USD
    # schemas are available without violating Isaac Sim's import ordering.
    from pxr import Usd, UsdGeom, UsdPhysics

    source_path = os.path.abspath(usd_path)
    source_stage = Usd.Stage.Open(source_path, load=Usd.Stage.LoadNone)
    if source_stage is None:
        raise RuntimeError(f"Unable to open scene USD for sanitizing: {source_path}")
    source_default = source_stage.GetDefaultPrim()
    if not source_default or not source_default.IsValid():
        raise RuntimeError(f"Scene USD has no valid default prim: {source_path}")

    physics_paths = [
        prim.GetPath()
        for prim in source_stage.Traverse()
        if prim.IsA(UsdPhysics.Scene)
    ]
    # Some released Home/Commercial assets contain blanket Xforms with a
    # collision API but no PhysX-compatible geometry.  Disabling collision on
    # these prims preserves their rendering while preventing invalid
    # PxGeometry creation during ManagerBasedRLEnv startup.  This must be
    # authored in the wrapper before the stage is loaded; the legacy blanket
    # scaling in ``adjust_usd_scale`` runs too late.
    blanket_collision_paths = [
        prim.GetPath()
        for prim in source_stage.TraverseAll()
        if "/blanket/" in str(prim.GetPath()).lower()
        and "PhysicsCollisionAPI" in prim.GetAppliedSchemas()
    ]
    if not physics_paths and not blanket_collision_paths:
        return source_path

    wrapper_dir = os.path.join(scene_dir, ".xnavdp_scene_wrappers", scene_type)
    os.makedirs(wrapper_dir, exist_ok=True)
    wrapper_path = os.path.join(wrapper_dir, f"{scene_name}.no_physics.usda")

    regenerate = True
    existing_stage = None
    if os.path.isfile(wrapper_path) and os.path.getmtime(wrapper_path) >= os.path.getmtime(source_path):
        try:
            existing_stage = Usd.Stage.Open(wrapper_path, load=Usd.Stage.LoadNone)
            wrapper_data = existing_stage.GetRootLayer().customLayerData
            regenerate = not (
                existing_stage
                and existing_stage.GetDefaultPrim().IsValid()
                and wrapper_data.get("xnavdpSceneWrapperVersion")
                == _SCENE_WRAPPER_VERSION
                and all(
                    existing_stage.GetPrimAtPath(path).IsValid()
                    and not existing_stage.GetPrimAtPath(path).IsActive()
                    for path in physics_paths
                )
                and all(
                    existing_stage.GetPrimAtPath(path).IsValid()
                    and UsdPhysics.CollisionAPI(
                        existing_stage.GetPrimAtPath(path)
                    ).GetCollisionEnabledAttr().Get() is False
                    for path in blanket_collision_paths
                )
            )
        except Exception:
            regenerate = True

    if regenerate:
        if os.path.exists(wrapper_path):
            # Rebuild our generated cache layer in place.  Unlinking and then
            # calling CreateNew with the same identifier fails when USD still
            # has the old layer in its process-wide registry.
            wrapper_stage = existing_stage or Usd.Stage.Open(
                wrapper_path, load=Usd.Stage.LoadNone
            )
            if wrapper_stage is None:
                raise RuntimeError(
                    f"Unable to reopen scene USD wrapper: {wrapper_path}"
                )
            wrapper_stage.GetRootLayer().Clear()
        else:
            wrapper_stage = Usd.Stage.CreateNew(wrapper_path)
        if wrapper_stage is None:
            raise RuntimeError(f"Unable to create scene USD wrapper: {wrapper_path}")
        default_path = source_default.GetPath()
        default_type = source_default.GetTypeName() or "Xform"
        wrapper_default = wrapper_stage.DefinePrim(default_path, default_type)
        wrapper_default.GetReferences().AddReference(source_path, default_path)
        wrapper_stage.SetDefaultPrim(wrapper_default)
        UsdGeom.SetStageUpAxis(wrapper_stage, UsdGeom.GetStageUpAxis(source_stage))
        UsdGeom.SetStageMetersPerUnit(
            wrapper_stage, UsdGeom.GetStageMetersPerUnit(source_stage)
        )
        wrapper_stage.SetTimeCodesPerSecond(source_stage.GetTimeCodesPerSecond())
        for physics_path in physics_paths:
            wrapper_stage.OverridePrim(physics_path).SetActive(False)
        for collision_path in blanket_collision_paths:
            collision_prim = wrapper_stage.OverridePrim(collision_path)
            UsdPhysics.CollisionAPI(
                collision_prim
            ).CreateCollisionEnabledAttr().Set(False)
        wrapper_stage.GetRootLayer().customLayerData = {
            "xnavdpSceneWrapperVersion": _SCENE_WRAPPER_VERSION,
        }
        wrapper_stage.GetRootLayer().Save()

    print(
        "[scene_loader] sanitized scene via wrapper: "
        f"{wrapper_path} (physics_scenes={len(physics_paths)}, "
        f"invalid_blanket_colliders={len(blanket_collision_paths)})"
    )
    return wrapper_path


def _scene_entries(cfg) -> list[dict]:
    """Build evaluation scenes without changing the configured list or order."""
    scene_dir = cfg.environment.scene_dir
    dataset_dir = getattr(cfg.environment, "dataset_dir", None)
    scene_type = getattr(cfg.environment, "scene_type", None)

    if dataset_dir:
        scene_type = scene_type or "home"
        if scene_type == "home":
            scene_subdir = os.path.join("internscenes_home", "scenes_home")
        elif scene_type == "commercial":
            scene_subdir = os.path.join("internscenes_commercial", "scenes_commercial")
        else:
            raise ValueError(f"Unsupported metadata-backed evaluation scene_type: {scene_type!r}")

        split_file = getattr(
            cfg.environment,
            "scene_split_file",
            os.path.join(scene_dir, "scene_split.json"),
        )
        if not os.path.isfile(split_file):
            split_file = os.path.join(os.path.dirname(dataset_dir), "scene_split.json")
        split = getattr(cfg.environment, "scene_split", "eval")
        if split not in ("train", "eval"):
            raise ValueError(f"environment.scene_split must be 'train' or 'eval', got {split!r}")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"Evaluation requires a scene split file, but none was found at {split_file}"
            )
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        split_key = f"{scene_type}_{split}"
        if split_key not in split_data:
            raise KeyError(f"Scene split {split_key!r} is missing from {split_file}")
        scene_names = list(split_data[split_key])

        if os.path.isdir(os.path.join(scene_dir, scene_subdir)):
            scene_root = os.path.join(scene_dir, scene_subdir)
        else:
            scene_root = scene_dir
        esdf_root = os.path.join(dataset_dir, "esdf")
        pointgoal_root = os.path.join(dataset_dir, "pointgoal_start_pair")
        if not os.path.isdir(esdf_root) or not os.path.isdir(pointgoal_root):
            raise FileNotFoundError(
                f"Evaluation metadata is incomplete under {dataset_dir}: "
                "expected esdf/ and pointgoal_start_pair/"
            )
    else:
        scene_root = scene_dir
        scene_type = scene_type or ("cluttered" if "cluttered" in scene_dir.lower() else "home")
        pointgoal_root = getattr(cfg.environment, "pointgoal_dir", None)
        if pointgoal_root is None and scene_type == "cluttered":
            metadata_candidate = os.path.join(
                os.path.dirname(scene_dir),
                "navigation_metadata",
                os.path.basename(os.path.normpath(scene_dir)),
                "pointgoal_start_pair",
            )
            if os.path.isdir(metadata_candidate):
                pointgoal_root = metadata_candidate
        scene_names = sorted(
            name for name in os.listdir(scene_root)
            if os.path.isdir(os.path.join(scene_root, name))
        )
        if pointgoal_root and os.path.isdir(pointgoal_root):
            scene_names = [
                name for name in sorted(os.listdir(pointgoal_root))
                if os.path.isdir(os.path.join(pointgoal_root, name))
                and os.path.isdir(os.path.join(scene_root, name))
            ]

    entries = []
    for scene_name in scene_names:
        scene_path = os.path.join(scene_root, scene_name)
        if dataset_dir:
            esdf_path = os.path.join(dataset_dir, "esdf", scene_name, "navigable.ply")
            pointgoal_path = find_eval_pointgoal_path(
                os.path.join(dataset_dir, "pointgoal_start_pair", scene_name)
            )
            usd_variant = getattr(cfg.environment, "usd_variant", "navigation")
        else:
            # Scene-N1 clutter evaluation was released without occupancy.ply.
            # It is only needed by optional occupancy-grid visualization, not
            # point-goal evaluation, so keep it disabled for these scenes.
            esdf_path = (
                None
                if scene_type == "cluttered"
                else os.path.join(scene_path, "occupancy.ply")
            )
            pointgoal_path = find_eval_pointgoal_path(
                os.path.join(pointgoal_root, scene_name) if pointgoal_root else scene_path
            )
            usd_variant = "auto"

        scene_data = {
            "scene_name": scene_name,
            "scene_type": scene_type,
            "usd_path": find_usd_path(scene_path, usd_variant),
            "esdf_path": esdf_path,
            "pointgoal_path": pointgoal_path,
        }
        entries.append(scene_data)

    return entries


def _resolve_scene(cfg, scene_index: int) -> tuple[dict, str]:
    scene_data_list = _scene_entries(cfg)
    if not scene_data_list:
        raise ValueError(f"No evaluation scenes are configured for {cfg.environment.scene_dir}")
    if not 0 <= scene_index < len(scene_data_list):
        raise IndexError(
            f"scene_index {scene_index} is out of range for "
            f"{len(scene_data_list)} configured evaluation scenes"
        )

    scene_data = scene_data_list[scene_index]
    scene_name = scene_data["scene_name"]
    required_assets = ["usd_path", "pointgoal_path"]
    if scene_data["esdf_path"] is not None:
        required_assets.append("esdf_path")
    missing = [
        key for key in required_assets
        if not scene_data[key] or not os.path.exists(scene_data[key])
    ]
    if missing:
        raise FileNotFoundError(
            f"Configured evaluation scene {scene_name!r} is missing required assets: "
            f"{', '.join(missing)}. The scene list and index were left unchanged."
        )
    scene_data["usd_path"] = _without_embedded_physics_scene(
        scene_data["usd_path"],
        scene_dir=cfg.environment.scene_dir,
        scene_name=scene_name,
        scene_type=scene_data["scene_type"],
    )
    return scene_data, scene_name


def create_environment(cfg, scene_index: int, device: str, seed: int = 1234):
    """Create the X-NavDP Isaac evaluation environment from an eval config."""
    from src.environment import create_dingoeval_environment

    scene_data, scene_name = _resolve_scene(cfg, scene_index)
    controller_config = namespace_to_dict(cfg.controller) if hasattr(cfg, "controller") else None
    expected_controller_types = {
        "dingo": "differential",
        "unitree_g1": "unitree_g1",
        "unitree_go2": "unitree_go2",
    }
    embodiment = cfg.environment.embodiment
    expected_type = expected_controller_types.get(embodiment)
    configured_type = controller_config.get("type") if controller_config else None
    if expected_type is None:
        raise ValueError(f"Unsupported evaluation embodiment: {embodiment!r}")
    if configured_type is not None and configured_type != expected_type:
        raise ValueError(
            f"controller.type {configured_type!r} is incompatible with "
            f"environment.embodiment {embodiment!r}; expected {expected_type!r}"
        )
    env, controller = create_dingoeval_environment(
        scene_dir=cfg.environment.scene_dir,
        scene_index=0,
        num_envs=cfg.environment.num_envs,
        scene_scale=getattr(cfg.environment, "scene_scale", None),
        device=device,
        embodiment=embodiment,
        scene_data=scene_data,
        controller_config=controller_config,
        camera_profile=getattr(cfg.environment, "camera_profile", "dingo"),
        control_dt=getattr(cfg.environment, "control_dt", None),
        seed=seed,
    )
    return env, controller, scene_name
