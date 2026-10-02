"""
Watson / watsonx.ai service.

Handles token refresh, scoring endpoint calls, and deployment status checks.
All credentials come from environment variables — never hardcoded.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from app.config import (
    IBM_API_KEY,
    IBM_DEPLOYMENT_ID,
    IBM_REGION,
    IBM_WML_URL,
    watson_configured,
)

logger = logging.getLogger(__name__)

# IAM token cache
_iam_token: str = ""
_iam_expiry: float = 0.0
IAM_URL = "https://iam.cloud.ibm.com/identity/token"


def _get_iam_token() -> str:
    """Fetch (or return cached) IBM Cloud IAM access token."""
    global _iam_token, _iam_expiry

    if _iam_token and time.time() < _iam_expiry - 60:
        return _iam_token

    if not IBM_API_KEY:
        raise RuntimeError("IBM_API_KEY not configured.")

    resp = requests.post(
        IAM_URL,
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": IBM_API_KEY,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    _iam_token = payload["access_token"]
    _iam_expiry = time.time() + payload.get("expires_in", 3600)
    return _iam_token


def check_watson_status() -> dict[str, Any]:
    """
    Returns real status based on actual credential/connection checks.
    Never fakes the connection status.
    """
    if not watson_configured():
        return {
            "configured": False,
            "connected": False,
            "deployment_active": False,
            "status_message": "Not Configured — set IBM_API_KEY, IBM_PROJECT_ID, IBM_DEPLOYMENT_ID",
            "deployment_id_masked": "",
            "model_name": "",
            "region": IBM_REGION,
        }

    try:
        token = _get_iam_token()
        # Check deployment endpoint
        url = f"{IBM_WML_URL}/ml/v4/deployments/{IBM_DEPLOYMENT_ID}?version=2023-07-01"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        resp = requests.get(url, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            model_name = (
                data.get("entity", {})
                .get("asset", {})
                .get("name", "Unknown")
            )
            masked_id = IBM_DEPLOYMENT_ID[:8] + "****" if IBM_DEPLOYMENT_ID else ""
            return {
                "configured": True,
                "connected": True,
                "deployment_active": True,
                "status_message": "Connected — Deployment Active",
                "deployment_id_masked": masked_id,
                "model_name": model_name,
                "region": IBM_REGION,
            }
        else:
            return {
                "configured": True,
                "connected": True,
                "deployment_active": False,
                "status_message": f"Connected but deployment returned HTTP {resp.status_code}",
                "deployment_id_masked": IBM_DEPLOYMENT_ID[:8] + "****",
                "model_name": "",
                "region": IBM_REGION,
            }

    except requests.exceptions.ConnectionError:
        return {
            "configured": True,
            "connected": False,
            "deployment_active": False,
            "status_message": "Connection Failed — cannot reach IBM Cloud",
            "deployment_id_masked": IBM_DEPLOYMENT_ID[:8] + "****" if IBM_DEPLOYMENT_ID else "",
            "model_name": "",
            "region": IBM_REGION,
        }
    except Exception as exc:
        logger.exception("Watson status check failed")
        return {
            "configured": True,
            "connected": False,
            "deployment_active": False,
            "status_message": f"Error: {str(exc)[:120]}",
            "deployment_id_masked": IBM_DEPLOYMENT_ID[:8] + "****" if IBM_DEPLOYMENT_ID else "",
            "model_name": "",
            "region": IBM_REGION,
        }


def predict_with_watson(input_values: list[list[Any]]) -> float:
    """
    Call the IBM Watson ML online deployment scoring endpoint.
    input_values: list of feature rows (already preprocessed or raw, depending on deployment).
    Returns the predicted salary (float).
    """
    if not watson_configured():
        raise RuntimeError("Watson is not configured. Check IBM credentials.")

    token = _get_iam_token()
    url = (
        f"{IBM_WML_URL}/ml/v4/deployments/{IBM_DEPLOYMENT_ID}"
        f"/predictions?version=2023-07-01"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "input_data": [{"values": input_values}]
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()

    result = resp.json()
    # Standard WML response structure
    predictions = result["predictions"][0]["values"]
    predicted_salary = float(predictions[0][0])
    return predicted_salary


def deploy_model_to_watson(model_path: str, pipeline_path: str, metadata: dict[str, Any]) -> str:
    """
    Deploy local sklearn model to IBM Watson ML.
    Returns the new deployment ID.
    This is a simplified deployment flow; full AutoAI deployment
    requires the ibm-watson-machine-learning SDK.
    """
    try:
        from ibm_watson_machine_learning import APIClient  # type: ignore

        wml_credentials = {
            "url": IBM_WML_URL,
            "apikey": IBM_API_KEY,
        }
        client = APIClient(wml_credentials)

        from app.config import IBM_SPACE_ID
        if IBM_SPACE_ID:
            client.set.default_space(IBM_SPACE_ID)

        # Store model
        model_props = {
            client.repository.ModelMetaNames.NAME: f"Salary Predictor - {metadata.get('model_name','')}",
            client.repository.ModelMetaNames.TYPE: "scikit-learn_1.3",
            client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: (
                client.software_specifications.get_uid_by_name("runtime-23.1-py3.10")
            ),
        }
        stored = client.repository.store_model(
            model=model_path,
            meta_props=model_props,
        )
        model_uid = client.repository.get_model_id(stored)

        # Deploy
        deploy_props = {
            client.deployments.ConfigurationMetaNames.NAME: "Salary Predictor Deployment",
            client.deployments.ConfigurationMetaNames.ONLINE: {},
        }
        deployment = client.deployments.create(model_uid, meta_props=deploy_props)
        deployment_id = client.deployments.get_id(deployment)
        logger.info(f"Watson deployment created: {deployment_id}")
        return deployment_id

    except ImportError:
        raise RuntimeError(
            "ibm-watson-machine-learning package not installed. "
            "Run: pip install ibm-watson-machine-learning"
        )
    except Exception as exc:
        logger.exception("Watson deployment failed")
        raise RuntimeError(f"Watson deployment failed: {exc}") from exc
