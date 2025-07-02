import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from score_backfill_prefect import (
    get_deployment_id,
    trigger_backfill_flow,
    trigger_backfill_flow_v2,
    backfill_months
)

@pytest.mark.asyncio
@patch("score_backfill_prefect.get_client")
async def test_get_deployment_id(mock_get_client):
    # Mock deployment data
    mock_client = AsyncMock()
    mock_client.read_deployments.return_value = [MagicMock(id="fake-deployment-id")]
    mock_get_client.return_value.__aenter__.return_value = mock_client

    result = await get_deployment_id("my-deployment", "my-flow")
    assert result == "fake-deployment-id"
    mock_client.read_deployments.assert_called_once()

@pytest.mark.asyncio
@patch("score_backfill_prefect.get_client")
async def test_trigger_backfill_flow(mock_get_client):
    mock_client = AsyncMock()
    mock_flow_run = MagicMock(id="fake-flow-run-id")
    mock_client.create_flow_run_from_deployment.return_value = mock_flow_run
    mock_get_client.return_value.__aenter__.return_value = mock_client

    run_id = await trigger_backfill_flow(
        deployment_id="12345",
        taxi="green",
        year=2023,
        month=5,
        run_date="2023-05-01",
        tracking_server="http://localhost:5000",
        model_name="model-v1"
    )
    assert run_id == "fake-flow-run-id"

@pytest.mark.asyncio
@patch("score_backfill_prefect.run_deployment", new_callable=AsyncMock)
async def test_trigger_backfill_flow_v2(mock_run_deployment):
    mock_flow_run = MagicMock(id="fake-flow-run-id")
    mock_run_deployment.return_value = mock_flow_run

    run_id = await trigger_backfill_flow_v2(
        deployment_name="flow/deployment",
        taxi="yellow",
        year=2022,
        month=3,
        run_date="2022-03-01",
        tracking_server="http://localhost:5000",
        model_name="model-v2"
    )
    assert run_id == "fake-flow-run-id"

@pytest.mark.asyncio
@patch("score_backfill_prefect.trigger_backfill_flow_v2", new_callable=AsyncMock)
async def test_backfill_months_concurrent_v2(mock_trigger_v2):
    mock_trigger_v2.side_effect = [f"run-{i}" for i in range(3)]

    flow_run_ids = await backfill_months(
        start_month=1,
        end_month=3,
        year=2024,
        taxi="yellow",
        tracking_server="http://localhost:5000",
        model_name="test-model",
        deployment_name="flow/deployment",
        use_run_deployment=True,
        concurrent=True
    )
    assert flow_run_ids == ["run-0", "run-1", "run-2"]
    assert mock_trigger_v2.call_count == 3
