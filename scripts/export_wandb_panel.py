"""Screenshot the native W&B parallel-coordinate panel without storing credentials."""

import argparse
import asyncio
import base64
import json
from pathlib import Path

from playwright.async_api import async_playwright

import wandb


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("artifacts/wandb-report.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/wandb-parallel.png"))
    args = parser.parse_args()
    if not wandb.api.api_key:
        raise RuntimeError("Authenticate privately with: uv run --locked wandb login")
    url = json.loads(args.report.read_text())["report_url"]
    authorization = "Basic " + base64.b64encode(("api:" + wandb.api.api_key).encode()).decode()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1600, "height": 1100}, device_scale_factor=2
        )

        async def graphql(route):
            response = await page.request.post(
                "https://api.wandb.ai/graphql",
                data=route.request.post_data,
                headers={"content-type": "application/json", "authorization": authorization},
                timeout=60000,
            )
            await route.fulfill(response=response)

        # The new Forge frontend uses a proxy; authenticate only against W&B itself.
        await page.route("https://forge.coreweave.com/api/wandb/graphql", graphql)
        for host in ("**/browser-intake-*/**", "**/analytics.wandb.ai/**", "**/cdn.segment.com/**"):
            await page.route(host, lambda route: route.abort())
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        panel = page.locator(".editable-panel.parallel-coordinates")
        await panel.wait_for(timeout=60000)
        await panel.scroll_into_view_if_needed()
        await panel.get_by_text("rmsprop", exact=True).wait_for(timeout=60000)
        await panel.get_by_text("Best validation accuracy (%)", exact=True).wait_for(timeout=60000)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        await panel.screenshot(path=str(args.output), timeout=60000)
        await page.unroute_all(behavior="ignoreErrors")
        await browser.close()
    print(f"Exported native W&B parallel-coordinate panel: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
