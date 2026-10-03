import asyncio
import httpx
from datetime import datetime, timezone
import json
from unittest.mock import patch, AsyncMock

# We import the FastAPI app and use httpx.AsyncClient to send requests to it directly
from gateway.server import app, app_state, lifespan
from gateway.ledger.store import SqliteLedgerStore
from gateway.core.types import CallOutcome

async def run_simulation():
    print("--- STARTING SIMULATION ---")
    
    transport = httpx.ASGITransport(app=app)
    
    async with lifespan(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            
            # Scenario 1: A successful OpenAI request
            print("\n[Scenario 1: Successful OpenAI Request]")
            with patch('gateway.adapters.openai.OpenAIAdapter.send', new_callable=AsyncMock) as mock_send:
                mock_send.return_value = httpx.Response(
                    200, 
                    json={"choices": [{"message": {"content": "Hello!"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                    headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "999", "x-ratelimit-reset-requests": "1s"}
                )
                response = await client.post("/v1/proxy", json={
                    "operation": "chat.completions.create",
                    "payload": {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
                })
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.json()}")
                
            # Scenario 2: Policy Engine Blocking (BOLA/AuthZ check)
            print("\n[Scenario 2: Policy Block (AuthZ/BOLA)]")
            ledger: SqliteLedgerStore = app_state["ledger"]
            
            print("Sending 'dangerous_operation' which is not in the allowed policy list...")
            
            with patch('gateway.adapters.anthropic.AnthropicAdapter.send', new_callable=AsyncMock) as mock_send:
                response = await client.post("/v1/proxy", json={
                    "operation": "dangerous_operation",
                    "payload": {}
                })
                print(f"Policy Block Response status: {response.status_code}")
                print(f"Policy Block Response detail: {response.json()}")

            # Scenario 3: Anthropic Overloaded (529) -> triggers Fallback to REST?
            print("\n[Scenario 3: Transient Error & Router Fallback]")
            with patch('gateway.adapters.anthropic.AnthropicAdapter.send', new_callable=AsyncMock) as mock_anthropic, \
                 patch('gateway.adapters.openai.OpenAIAdapter.send', new_callable=AsyncMock) as mock_openai:
                
                # Anthropic fails with 529
                mock_anthropic.return_value = httpx.Response(529, json={"error": {"type": "overloaded_error", "message": "Overloaded"}})
                # OpenAI succeeds on fallback
                mock_openai.return_value = httpx.Response(200, json={"choices": [{"message": {"content": "Fallback works"}}]}, headers={})
                
                response = await client.post("/v1/proxy", json={
                    "operation": "chat",
                    "payload": {}
                })
                print(f"Fallback Response status: {response.status_code}")
                print(f"Fallback Response body: {response.json()}")
                
            # Check Ledger for the trail
            print("\n[Scenario 4: Auditing the Unified Ledger]")
            events = await ledger.query()
            for e in events[-5:]:
                print(f"Ledger Event -> Provider: {e.provider}, Outcome: {e.outcome}, Error Category: {e.error_category}")

if __name__ == "__main__":
    asyncio.run(run_simulation())
