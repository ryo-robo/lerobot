"""The dataset's relative-action declaration flows into the policy config, once."""

from types import SimpleNamespace

import pytest

from lerobot.policies.factory import _apply_dataset_relative_action_provenance


def _cfg(**overrides):
    base = {"use_relative_actions": True, "relative_se3_pose_groups": []}
    base.update(overrides)
    return SimpleNamespace(**base)


def _meta(groups):
    declaration = {"chunk_size": 50, "exclude_joints": ["gripper"], "se3_pose_groups": groups}
    return SimpleNamespace(info=SimpleNamespace(relative_action=declaration))


def test_declared_groups_fill_an_unset_config():
    cfg = _cfg()
    _apply_dataset_relative_action_provenance(cfg, _meta([[0, 1, 2, 3, 4, 5]]))
    assert cfg.relative_se3_pose_groups == [[0, 1, 2, 3, 4, 5]]


def test_matching_override_passes_and_contradiction_raises():
    cfg = _cfg(relative_se3_pose_groups=[[0, 1, 2, 3, 4, 5]])
    _apply_dataset_relative_action_provenance(cfg, _meta([[0, 1, 2, 3, 4, 5]]))

    cfg = _cfg(relative_se3_pose_groups=[[1, 2, 3, 4, 5, 6]])
    with pytest.raises(ValueError, match="contradicts the dataset"):
        _apply_dataset_relative_action_provenance(cfg, _meta([[0, 1, 2, 3, 4, 5]]))


def test_ignored_without_relative_actions_or_declaration():
    cfg = _cfg(use_relative_actions=False)
    _apply_dataset_relative_action_provenance(cfg, _meta([[0, 1, 2, 3, 4, 5]]))
    assert cfg.relative_se3_pose_groups == []

    cfg = _cfg()
    _apply_dataset_relative_action_provenance(
        cfg, SimpleNamespace(info=SimpleNamespace(relative_action=None))
    )
    assert cfg.relative_se3_pose_groups == []
