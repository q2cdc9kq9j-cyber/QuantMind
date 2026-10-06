import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from backend.services.simulation.engine import SimulationEngine
from backend.services.simulation.services import simulation_hosted_scheduler as hosted


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_id", ["12", "sys_topk"])
async def test_strategy_parameters_and_template_identity_are_preserved(strategy_id):
    storage = MagicMock()
    storage.get = AsyncMock(
        return_value={
            "parameters": {
                "topk": 50,
                "n_drop": 10,
                "rebalance_days": 3,
                "deterministic_buy_order": True,
            }
        }
    )
    with patch(
        "backend.services.simulation.engine.get_strategy_storage_service",
        return_value=storage,
    ):
        config = await SimulationEngine()._load_strategy_config(
            None, strategy_id, "10000001"
        )
    assert storage.get.await_args.kwargs["strategy_id"] == (
        12 if strategy_id == "12" else "sys_topk"
    )
    assert (config.topk, config.n_drop, config.rebalance_days) == (50, 10, 3)
    assert config.deterministic_buy_order is True


@pytest.mark.asyncio
async def test_runtime_overrides_replace_persisted_dropout_and_flags():
    storage = MagicMock()
    storage.get = AsyncMock(return_value={"parameters": {"n_drop": 10}})
    with patch(
        "backend.services.simulation.engine.get_strategy_storage_service",
        return_value=storage,
    ):
        config = await SimulationEngine()._load_strategy_config(
            None,
            "12",
            "10000001",
            params_override={
                "n_drop": 0,
                "rebalance_days": 5,
                "enable_min_score": True,
            },
        )
    assert (config.n_drop, config.rebalance_days, config.enable_min_score) == (
        0,
        5,
        True,
    )


@pytest.mark.asyncio
async def test_hosted_cycle_passes_runtime_parameters_to_engine():
    run_cycle = AsyncMock(return_value=SimpleNamespace(error=None, order_count=0))
    with (
        patch.object(
            hosted,
            "_resolve_hosted_signal_run_id",
            AsyncMock(return_value=("signal_run", None)),
        ),
        patch(
            "backend.services.simulation.engine.simulation_engine.run_cycle", run_cycle
        ),
    ):
        await hosted.run_simulation_cycle_for_active(
            tenant_id="default",
            user_id="10000001",
            strategy_id="12",
            live_trade_config={"topk": 50, "n_drop": 10, "rebalance_days": 3},
        )
    assert run_cycle.await_args.kwargs["params_override"] == {
        "topk": 50,
        "n_drop": 10,
        "rebalance_days": 3,
    }


@pytest.mark.asyncio
async def test_bootstrap_cycle_denies_observe_only_before_signal_lookup():
    resolver = AsyncMock()
    with patch.object(hosted, "_resolve_hosted_signal_run_id", resolver):
        result = await hosted.run_simulation_cycle_for_active(
            tenant_id="default",
            user_id="10000001",
            strategy_id="12",
            live_trade_config={"trading_permission": "observe_only"},
        )
    assert result["status"] == "skipped"
    resolver.assert_not_awaited()


def _scheduler(payload):
    redis = MagicMock()
    redis.client.get.return_value = json.dumps(
        {
            "mode": "SIMULATION",
            "strategy_id": "12",
            "runtime_tenant_id": "default",
            "runtime_user_id": "10000001",
            **payload,
        }
    )
    redis.client.set.return_value = True
    return hosted.SimulationHostedScheduler(redis)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"trading_permission": "observe_only"},
        {"trading_permission": "blocked"},
        {"execution_config": {"auto_trade_enabled": False}},
        {"live_trade_config": {"auto_trade_enabled": False}},
    ],
)
async def test_scheduler_denies_disabled_permissions_before_creating_jobs(payload):
    scheduler = _scheduler(payload)
    with patch.object(hosted, "_should_trigger") as schedule:
        triggered = await scheduler._process_key(
            "trade:active_strategy:default:10000001",
            now=datetime(2026, 6, 2, 14, 50, tzinfo=ZoneInfo("Asia/Shanghai")),
        )
    assert triggered is False
    schedule.assert_not_called()
    scheduler.redis.client.set.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {"status": "failed", "error": "无可用信号"},
        {"status": "failed", "error": "账户不存在"},
        {"status": "failed"},
        {"status": "succeeded", "order_count": 0},
        {"status": "skipped", "error": "window_pending"},
    ],
)
async def test_job_status_matches_cycle_outcome(result):
    scheduler = _scheduler({})
    jobs = MagicMock()
    for name in (
        "ensure_job",
        "mark_ready",
        "mark_started",
        "mark_finished",
        "mark_skipped",
    ):
        setattr(jobs, name, AsyncMock())
    decision = hosted.SimulationScheduleDecision(True, "ALL", "2026-06-02", "ready")
    with (
        patch.object(hosted, "_should_trigger", return_value=decision),
        patch.object(hosted, "SimulationRebalanceJobService", jobs),
        patch.object(
            hosted, "run_simulation_cycle_for_active", AsyncMock(return_value=result)
        ),
    ):
        if result["status"] == "failed":
            with pytest.raises(RuntimeError):
                await scheduler._process_key(
                    "trade:active_strategy:default:10000001",
                    now=datetime.now(ZoneInfo("Asia/Shanghai")),
                )
            assert jobs.mark_finished.await_args.kwargs["status"] == "failed"
            scheduler.redis.client.delete.assert_called_once()
        else:
            triggered = await scheduler._process_key(
                "trade:active_strategy:default:10000001",
                now=datetime.now(ZoneInfo("Asia/Shanghai")),
            )
            assert triggered is (result["status"] == "succeeded")
            if result["status"] == "skipped":
                jobs.mark_skipped.assert_awaited_once()
                jobs.mark_finished.assert_not_awaited()
            else:
                assert jobs.mark_finished.await_args.kwargs["status"] == "succeeded"


@pytest.mark.asyncio
async def test_oss_parameter_precedence_preserves_code_fallback():
    storage = MagicMock()
    storage.get = AsyncMock(
        return_value={
            "parameters": {"topk": 30},
            "code": "STRATEGY_CONFIG = {'kwargs': {'topk': 50, 'n_drop': 10, 'rebalance_days': 3}}",
        }
    )
    with patch(
        "backend.services.simulation.engine.get_strategy_storage_service",
        return_value=storage,
    ):
        engine = SimulationEngine()
        config = await engine._load_strategy_config(None, "12", "10000001")
        overridden = await engine._load_strategy_config(
            None, "12", "10000001", params_override={"topk": 20, "n_drop": 0}
        )
    assert (config.topk, config.n_drop, config.rebalance_days) == (30, 10, 3)
    assert (overridden.topk, overridden.n_drop, overridden.rebalance_days) == (20, 0, 3)
