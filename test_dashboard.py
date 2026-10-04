import asyncio
import httpx
from gateway.server import app, app_state, lifespan

async def test_dashboard():
    print("--- TESTING DASHBOARD APIs ---")
    transport = httpx.ASGITransport(app=app)
    
    async with lifespan(app):
        # We need to simulate some traffic first to put data in the ledger
        import simulate_traffic
        await simulate_traffic.run_simulation()
        print("\n--- TRAFFIC INJECTED, FETCHING DASHBOARD DATA ---")
        
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            
            # Fetch the Overview KPI endpoint
            resp1 = await client.get("/v1/dashboard/overview")
            print(f"\n[GET /v1/dashboard/overview] -> Status {resp1.status_code}")
            import json
            print(json.dumps(resp1.json(), indent=2))
            
            # Fetch the Providers endpoint
            resp2 = await client.get("/v1/dashboard/providers")
            print(f"\n[GET /v1/dashboard/providers] -> Status {resp2.status_code}")
            print(json.dumps(resp2.json(), indent=2))
            
            # Fetch the Budget endpoint
            resp3 = await client.get("/v1/dashboard/budget")
            print(f"\n[GET /v1/dashboard/budget] -> Status {resp3.status_code}")
            print(json.dumps(resp3.json(), indent=2))
            
            # Fetch the Events endpoint
            resp4 = await client.get("/v1/dashboard/events?limit=3")
            print(f"\n[GET /v1/dashboard/events?limit=3] -> Status {resp4.status_code}")
            print(json.dumps(resp4.json(), indent=2))

if __name__ == "__main__":
    asyncio.run(test_dashboard())
