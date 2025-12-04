import os
import sys
import json
import uuid
import random
import base64
import asyncio
import logging
import argparse

from vastai import Serverless

# ---------------------- Config ----------------------
DEFAULT_PROMPT = "a beautiful sunset over mountains, digital art, highly detailed"
ENDPOINT_NAME = "Comfy-Prod2"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 20
COST = 100  # Fixed cost for ComfyUI requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ---------------------- API Functions ----------------------
async def call_generate(
    client: Serverless,
    *,
    endpoint_name: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> dict:
    """Generate image using Text2Image modifier"""
    endpoint = await client.get_endpoint(name=endpoint_name)
    payload = {
        "input": {
            "request_id": str(uuid.uuid4()),
            "modifier": "Text2Image",
            "modifications": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
            },
        }
    }
    return await endpoint.request("/generate/sync", payload, cost=COST)


async def call_generate_workflow(
    client: Serverless,
    *,
    endpoint_name: str,
    workflow_json: dict,
) -> dict:
    """Generate using custom workflow JSON"""
    endpoint = await client.get_endpoint(name=endpoint_name)
    payload = {
        "input": {
            "request_id": str(uuid.uuid4()),
            "workflow_json": workflow_json,
        }
    }
    return await endpoint.request("/generate/sync", payload, cost=COST)


# ---------------------- Demo Class ----------------------
class APIDemo:
    def __init__(self, client: Serverless, endpoint_name: str):
        self.client = client
        self.endpoint_name = endpoint_name

    def extract_images(self, response: dict) -> list:
        """Extract image info from ComfyUI response"""
        images = []
        
        # Check for output array (S3/webhook configured)
        if "output" in response:
            for item in response["output"]:
                if "url" in item:
                    images.append({"type": "url", "path": item["url"]})
                elif "local_path" in item:
                    images.append({"type": "local", "path": item["local_path"]})
                elif "base64" in item:
                    images.append({"type": "base64", "data": item["base64"]})
        
        # Check for comfyui_response format (default)
        if "comfyui_response" in response:
            for prompt_id, data in response["comfyui_response"].items():
                if isinstance(data, dict) and "outputs" in data:
                    for node_id, node_output in data["outputs"].items():
                        if "images" in node_output:
                            for img in node_output["images"]:
                                images.append({
                                    "type": "remote",
                                    "filename": img.get("filename"),
                                    "subfolder": img.get("subfolder", ""),
                                })
        
        return images

    async def save_images(self, images: list, worker_url: str, prefix: str = "comfy") -> list:
        """Save images locally by fetching from remote server"""
        os.makedirs("generated_images", exist_ok=True)
        saved = []
        seen = set()

        for i, img in enumerate(images):
            if img["type"] == "base64":
                data = img["data"]
                if data.startswith("data:"):
                    data = data.split(",", 1)[-1]
                path = f"generated_images/{prefix}_{i}.png"
                with open(path, "wb") as f:
                    f.write(base64.b64decode(data))
                print(f"  💾 Saved: {path}")
                saved.append(path)
                
            elif img["type"] == "url":
                url = img["path"]
                if url in seen:
                    continue
                seen.add(url)
                try:
                    import urllib.request
                    path = f"generated_images/{prefix}_{len(saved)}.png"
                    urllib.request.urlretrieve(url, path)
                    print(f"  💾 Downloaded: {path}")
                    saved.append(path)
                except Exception as e:
                    print(f"  🔗 URL: {url}")
                    saved.append(url)
                    
            elif img["type"] == "local":
                remote_path = img["path"]
                if remote_path in seen:
                    continue
                seen.add(remote_path)
                filename = os.path.basename(remote_path)
                # Try to fetch via /view endpoint
                local_path = await self._fetch_image(worker_url, filename, "", f"{prefix}_{len(saved)}.png")
                if local_path:
                    saved.append(local_path)
                else:
                    print(f"  📂 Remote: {remote_path}")
                    saved.append(remote_path)
                
            elif img["type"] == "remote":
                filename = img["filename"]
                if filename in seen:
                    continue
                seen.add(filename)
                subfolder = img.get("subfolder", "")
                # Try to fetch via /view endpoint
                local_path = await self._fetch_image(worker_url, filename, subfolder, f"{prefix}_{len(saved)}.png")
                if local_path:
                    saved.append(local_path)
                else:
                    print(f"  🖼️  Remote: {filename}")
                    saved.append(filename)

        return saved

    async def _fetch_image(self, worker_url: str, filename: str, subfolder: str, local_name: str) -> str | None:
        """Fetch image directly from worker's /view endpoint"""
        if not worker_url:
            print(f"  ⚠️  No worker URL available")
            return None
            
        try:
            import aiohttp
            
            params = {"filename": filename, "type": "output"}
            if subfolder:
                params["subfolder"] = subfolder
            
            url = f"{worker_url}/view"
            print(f"  🔗 Fetching from: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, ssl=False) as resp:
                    if resp.status == 200:
                        raw_bytes = await resp.read()
                        path = f"generated_images/{local_name}"
                        with open(path, "wb") as f:
                            f.write(raw_bytes)
                        print(f"  💾 Saved: {path}")
                        return path
                    else:
                        text = await resp.text()
                        print(f"  ❌ HTTP {resp.status}: {text[:100]}")
                        return None
        except Exception as e:
            print(f"  ❌ Fetch error: {e}")
            return None

    async def demo_prompt(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int | None,
    ):
        """Demo: Generate image from text prompt"""
        print("=" * 60)
        print("COMFYUI TEXT-TO-IMAGE DEMO")
        print("=" * 60)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print(f"Size: {width}x{height}, Steps: {steps}, Seed: {seed}")
        print("\n🎨 Generating image...")

        response = await call_generate(
            self.client,
            endpoint_name=self.endpoint_name,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
        )

        print("\n✅ Generation complete!")

        # Get worker URL for fetching images
        worker_url = response.get("url", "")
        print(f"Worker URL: {worker_url}")

        # Extract and handle images
        if "response" in response:
            images = self.extract_images(response["response"])
            if images:
                print(f"\n📁 {len(images)} image(s) generated:")
                await self.save_images(images, worker_url, prefix=f"comfy_{seed}")
            else:
                print("\nNo images found in response")
                print(json.dumps(response, indent=2, default=str)[:2000])
        else:
            print("\nUnexpected response format")
            print(json.dumps(response, indent=2, default=str)[:2000])

    async def demo_workflow(self, workflow_file: str):
        """Demo: Generate using custom workflow file"""
        print("=" * 60)
        print("COMFYUI CUSTOM WORKFLOW DEMO")
        print("=" * 60)

        if not os.path.exists(workflow_file):
            log.error(f"Workflow file not found: {workflow_file}")
            return

        with open(workflow_file, "r") as f:
            workflow_json = json.load(f)

        print(f"Workflow: {workflow_file}")
        print("\n🎨 Generating...")

        response = await call_generate_workflow(
            self.client,
            endpoint_name=self.endpoint_name,
            workflow_json=workflow_json,
        )

        print("\n✅ Generation complete!")

        worker_url = response.get("url", "")

        if "response" in response:
            images = self.extract_images(response["response"])
            if images:
                print(f"\n📁 {len(images)} image(s) generated:")
                await self.save_images(images, worker_url, prefix="workflow")
            else:
                print("\nNo images found in response")
                print(json.dumps(response, indent=2, default=str)[:2000])
        else:
            print("\nUnexpected response format")
            print(json.dumps(response, indent=2, default=str)[:2000])


# ---------------------- CLI ----------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vast ComfyUI-JSON Demo (Serverless SDK)")
    p.add_argument("--endpoint", default=ENDPOINT_NAME, help=f"Vast endpoint name (default: {ENDPOINT_NAME})")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, metavar="TEXT",
                   help=f"Prompt text (default: '{DEFAULT_PROMPT[:30]}...')")
    p.add_argument("--workflow", type=str, metavar="FILE", help="Use custom workflow JSON file instead")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Image width (default: {DEFAULT_WIDTH})")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Image height (default: {DEFAULT_HEIGHT})")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Steps (default: {DEFAULT_STEPS})")
    p.add_argument("--seed", type=int, default=None, help="Seed (default: random)")
    return p


async def main_async():
    args = build_arg_parser().parse_args()

    print("=" * 60)
    print(f"Using endpoint: {args.endpoint}")

    try:
        async with Serverless() as client:
            demo = APIDemo(client, args.endpoint)

            if args.workflow:
                await demo.demo_workflow(workflow_file=args.workflow)
            else:
                await demo.demo_prompt(
                    prompt=args.prompt,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    seed=args.seed,
                )

    except AttributeError as e:
        if "API key" in str(e):
            log.error("API key missing. Set VAST_API_KEY environment variable.")
        else:
            log.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
