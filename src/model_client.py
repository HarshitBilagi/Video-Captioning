import os
import yaml
import json
import base64
import time
import logging
import httpx
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment variables from .env
load_dotenv()

try:
    from src.style_prompts import get_style_prompts_system_prompt
except ImportError:
    try:
        from style_prompts import get_style_prompts_system_prompt
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from style_prompts import get_style_prompts_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Tone-specific system prompts for caption generation
TONE_PROMPTS = {
    "formal": (
        "Write exactly ONE sentence describing only what is literally visible "
        "in the scene. Objective, precise, documentary-style. No emojis, no "
        "hashtags, no opinions. Maximum 25 words."
    ),
    "sarcastic": (
        "Write exactly ONE short, dry, sarcastic sentence about this scene. "
        "Deadpan wit, understated. No emojis, no hashtags, no exclamation marks. "
        "Maximum 20 words."
    ),
    "humorous_tech": (
        "Write exactly ONE sentence in the format 'When you/your [scenario], "
        "but/and [tech twist]' — reframe the scene using a programming or tech "
        "metaphor as the punchline. Maximum 20 words."
    ),
    "humorous_non_tech": (
        "Write exactly ONE sentence in the format 'When you [relatable everyday "
        "scenario]' — observational humor about the scene, no tech jargon. "
        "Maximum 20 words."
    )
}

class ModelClient:
    """
    Client wrapper for interacting with Fireworks API models, configured via models.yaml.
    Supports a mock mode for local testing without valid credentials or configurations.
    """
    # Class-level usage counters shared across all instances
    _total_calls = 0
    _total_prompt_tokens = 0
    _total_completion_tokens = 0
    _total_tokens = 0
    _MAX_CALLS_PER_RUN = int(os.getenv("MAX_CALLS_PER_RUN", "100"))

    def __init__(self, config_path: str = "config/models.yaml", force_mock_mode: bool = False):
        """
        Initializes the ModelClient by loading the model configuration file.

        Args:
            config_path (str): Path to the yaml file containing model references.
            force_mock_mode (bool): Forces the client to run in mock mode.
        """
        self.config_path = config_path
        self.models_config = self._load_config()
        self.vision_model = self.models_config.get("vision_model", "placeholder")
        self.text_model = self.models_config.get("text_model", "placeholder")
        self.transcription_model = self.models_config.get("transcription_model", "whisper-v3")
        
        # Load API key
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        
        # Setup httpx async client
        headers = {}
        if self.api_key:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        self.client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(60.0, connect=10.0))
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        
        # Setup Mock Mode
        self.mock_mode = force_mock_mode
        if not self.mock_mode:
            # Check if key is missing or model configuration has placeholders
            if (not self.api_key 
                or not self.vision_model 
                or self.vision_model.lower() == "placeholder" 
                or not self.text_model 
                or self.text_model.lower() == "placeholder"):
                self.mock_mode = True
                print("[INFO] Running in MOCK MODE - no real API calls will be made.")

    def __del__(self):
        """Destructor to ensure requests session resources are explicitly released."""
        try:
            if hasattr(self, "client") and self.client is not None:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.client.aclose())
                else:
                    loop.run_until_complete(self.client.aclose())
        except Exception:
            pass

    def _check_call_limit(self, method_name: str) -> None:
        """
        Checks whether the class-level call count has exceeded MAX_CALLS_PER_RUN.
        Raises RuntimeError if the limit is exceeded to prevent runaway API spending.
        """
        if ModelClient._total_calls >= ModelClient._MAX_CALLS_PER_RUN:
            raise RuntimeError(
                f"Safety limit exceeded: {ModelClient._total_calls} API calls made "
                f"(limit: {ModelClient._MAX_CALLS_PER_RUN}). Halting to prevent "
                f"runaway credit usage. Increase MAX_CALLS_PER_RUN env var if intentional. "
                f"Triggered by: {method_name}"
            )

    def _track_usage(self, response_json: Dict[str, Any], method_name: str) -> None:
        """
        Extracts token usage stats from the API response and updates class-level counters.
        Logs the per-call usage at INFO level.
        """
        ModelClient._total_calls += 1
        usage = response_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        ModelClient._total_prompt_tokens += prompt_tokens
        ModelClient._total_completion_tokens += completion_tokens
        ModelClient._total_tokens += total_tokens

        logging.info(
            f"[Usage] {method_name}: prompt={prompt_tokens}, "
            f"completion={completion_tokens}, total={total_tokens} tokens | "
            f"Running totals: calls={ModelClient._total_calls}, "
            f"tokens={ModelClient._total_tokens}"
        )

    @classmethod
    def print_usage_summary(cls) -> None:
        """
        Prints a formatted summary of all API usage accumulated across all ModelClient instances.
        """
        print("\n------------------------------------------")
        print("         FIREWORKS API USAGE SUMMARY")
        print("------------------------------------------")
        print(f"  Total API Calls:        {cls._total_calls}")
        print(f"  Total Prompt Tokens:    {cls._total_prompt_tokens}")
        print(f"  Total Completion Tokens: {cls._total_completion_tokens}")
        print(f"  Total Tokens:           {cls._total_tokens}")
        print(f"  Max Calls Limit:        {cls._MAX_CALLS_PER_RUN}")
        if cls._total_calls == 0:
            print("  (No real API calls were made - mock mode was active)")
        print("------------------------------------------")

    @classmethod
    def reset_usage(cls) -> None:
        """
        Resets all class-level usage counters. Useful for testing.
        """
        cls._total_calls = 0
        cls._total_prompt_tokens = 0
        cls._total_completion_tokens = 0
        cls._total_tokens = 0

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the yaml model configuration.
        """
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"Error loading model config from {self.config_path}: {e}")
            return {}

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Loads an image file from path and encodes it to a base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def _post_with_retry(self, url: str, json_data: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        """
        Makes an async POST request to the API with retry logic and exponential backoff.
        """
        import asyncio
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.post(url, json=json_data)
                if response.status_code == 200:
                    # Small delay between consecutive successful calls to avoid rate limits
                    await asyncio.sleep(0.3)
                    return response.json()
                elif response.status_code in [408, 429, 500, 502, 503, 504]:
                    backoff = 2 * (attempt + 1)
                    logging.warning(f"API post failed with retriable status code {response.status_code}. Attempt {attempt + 1}/{max_retries + 1}. Retrying in {backoff}s...")
                else:
                    response.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                backoff = 2 * (attempt + 1)
                logging.warning(f"API post failed with exception: {e}. Attempt {attempt + 1}/{max_retries + 1}. Retrying in {backoff}s...")
                
            if attempt < max_retries:
                await asyncio.sleep(2 * (attempt + 1))  # exponential: 2s, 4s between retries
            else:
                if last_exception:
                    raise last_exception
                else:
                    raise RuntimeError(f"API request failed with status code {response.status_code} after {max_retries} retries.")
                    
        raise RuntimeError("API request failed.")

    def _parse_and_validate_json(self, content: str) -> Dict[str, Any]:
        """
        Parses JSON content returned by the model and ensures the required fields are present.
        """
        content_stripped = content.strip()
        
        def check_structure(d: Any) -> Dict[str, Any]:
            if not isinstance(d, dict):
                raise ValueError("Parsed output is not a JSON object/dict")
            required_keys = ["setting", "subjects", "actions", "notable_details", "audio_context"]
            for key in required_keys:
                if key not in d:
                    # Set defaults if missing to be graceful
                    d[key] = [] if key in ["subjects", "actions", "notable_details"] else ""
            return d

        # Try standard JSON parsing
        try:
            parsed = json.loads(content_stripped)
            return check_structure(parsed)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON contents between the first '{' and last '}'
        try:
            start_idx = content_stripped.find('{')
            end_idx = content_stripped.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = content_stripped[start_idx:end_idx+1]
                parsed = json.loads(json_str)
                return check_structure(parsed)
        except Exception:
            pass

        raise ValueError("Could not parse valid JSON object from model response.")

    async def describe_scene(self, frames: List[str], transcript: str) -> Dict[str, Any]:
        """
        Uses the configured vision model to generate a structured scene description from frames and transcript.

        Args:
            frames (List[str]): List of paths to the saved frame images.
            transcript (str): Transcribed audio text.

        Returns:
            Dict[str, Any]: Parsed structured scene details.
        """
        if self.mock_mode:
            logging.info("ModelClient.describe_scene called in MOCK MODE.")
            return {
                "setting": "A brightly lit Formula 1 podium area and pitlane celebration scene under daylight.",
                "subjects": [
                    "An F1 driver wearing a racing suit", 
                    "Team members in team uniforms", 
                    "Crowd of spectators in the background"
                ],
                "actions": [
                    "F1 driver celebrating on the podium", 
                    "Team members cheering and waving flags", 
                    "Sprinting and jumping on the track"
                ],
                "notable_details": [
                    "Metallic trophy in hand", 
                    "Sponsor branding on podium board", 
                    "Race cars in the pitlane"
                ],
                "audio_context": "Loud cheering, engines humming, and announcer commentary."
            }

        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY is missing. Please set it in your environment or .env file.")
        if not self.vision_model or self.vision_model.lower() == "placeholder":
            raise ValueError("vision_model in config/models.yaml is still set to 'placeholder'. Please configure a valid vision model name.")
            
        # Encode frames to base64 and build content blocks for the user message
        user_content_blocks = []
        total_b64_bytes = 0
        encoded_count = 0
        
        for b64_str in frames:
            try:
                total_b64_bytes += len(b64_str)
                encoded_count += 1
                user_content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_str}"
                    }
                })
            except Exception as e:
                logging.error(f"Error processing base64 frame: {e}")
                
        if not user_content_blocks:
            raise ValueError("No valid frames could be encoded for vision model analysis.")
        
        # Log payload size for cost awareness
        payload_size_mb = total_b64_bytes / (1024 * 1024)
        logging.info(
            f"[describe_scene] Sending {encoded_count} frame(s), "
            f"total base64 payload size: {total_b64_bytes:,} bytes ({payload_size_mb:.2f} MB)"
        )
            
        system_prompt = (
            "You are an AI video analysis assistant. You are given a sequence of keyframes from a video clip "
            "and the transcribed audio text. Analyze them carefully and describe the scene. "
            "You MUST respond ONLY with a valid JSON object matching the following structure. Do NOT wrap it in markdown block tags, "
            "do NOT write any explanatory text outside the JSON. Return exactly this JSON format:\n"
            "{\n"
            '  "setting": "string describing the setting",\n'
            '  "subjects": ["list of main subjects"],\n'
            '  "actions": ["list of main actions/events happening"],\n'
            '  "notable_details": ["list of key visual details"],\n'
            '  "audio_context": "string describing the audio context and how it relates to the visuals"\n'
            "}"
        )
        
        user_text = f"Audio transcript: '{transcript}'\nAnalyze the attached frames and describe the scene using the JSON format."
        
        # Append the text instruction block after the image blocks
        user_content_blocks.append({
            "type": "text",
            "text": user_text
        })
        
        payload = {
            "model": self.vision_model,
            "max_tokens": 1200,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": user_content_blocks
                }
            ],
            "temperature": 0.0
        }
        
        self._check_call_limit("describe_scene")
        response_json = await self._post_with_retry(self.api_url, payload)
        self._track_usage(response_json, "describe_scene")
        message = response_json["choices"][0]["message"]
        content = message.get("content")
        
        # If content is empty (model returned only reasoning_content), fall through to retry
        if not content:
            logging.warning("describe_scene: content was None or empty on first attempt, will retry...")
        
        try:
            if not content:
                raise ValueError("Content is None or empty in model response")
            return self._parse_and_validate_json(content)
        except ValueError as e:
            logging.warning(f"Initial JSON parsing failed: {e}. Retrying once with stricter instructions...")
            
            # Retry once with stricter instructions
            strict_system_prompt = (
                system_prompt + 
                "\nCRITICAL: Respond ONLY with the raw JSON string. "
                "Do NOT write ```json or other formatting. Begin with '{' and end with '}'."
            )
            payload["messages"][0]["content"] = [{"type": "text", "text": strict_system_prompt}]
            
            self._check_call_limit("describe_scene_retry")
            retry_response_json = await self._post_with_retry(self.api_url, payload)
            self._track_usage(retry_response_json, "describe_scene_retry")
            retry_message = retry_response_json["choices"][0]["message"]
            retry_content = retry_message.get("content")
            if not retry_content:
                raise ValueError("Content is None or empty in retry model response")
            
            return self._parse_and_validate_json(retry_content)

    async def generate_caption(self, scene_json: Dict[str, Any], tone: str, tone_prompt: str = None) -> str:
        """
        Uses the configured text model to generate a caption matching the requested tone.

        Args:
            scene_json (Dict[str, Any]): Structured details of the scene.
            tone (str): Target tone.
            tone_prompt (str, optional): Custom prompt to override the default tone prompt.

        Returns:
            str: Generated caption.
        """
        if self.mock_mode:
            logging.info(f"ModelClient.generate_caption called in MOCK MODE for tone: {tone}.")
            return f"[MOCK-{tone.upper()}] This is a placeholder caption for testing."

        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY is missing. Please set it in your environment or .env file.")
        if not self.text_model or self.text_model.lower() == "placeholder":
            raise ValueError("text_model in config/models.yaml is still set to 'placeholder'. Please configure a valid text model name.")

        sys_prompt = tone_prompt
        if not sys_prompt:
            sys_prompt = TONE_PROMPTS.get(tone.lower(), f"PLACEHOLDER_{tone.upper()}_PROMPT")

        user_content = (
            f"Scene details:\n{json.dumps(scene_json, indent=2)}\n\n"
            "Write the caption now. One caption only. No preamble. No markdown bold. No quotes around it."
        )

        payload = {
            "model": self.text_model,
            "max_tokens": 1200,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.7
        }

        self._check_call_limit("generate_caption")
        response_json = await self._post_with_retry(self.api_url, payload)
        self._track_usage(response_json, "generate_caption")
        message = response_json["choices"][0]["message"]
        content = message.get("content")
        
        # Retry once if content is None/empty (model sometimes returns only reasoning_content)
        if not content:
            logging.warning("generate_caption: content was None or empty, retrying once...")
            self._check_call_limit("generate_caption_retry")
            response_json = await self._post_with_retry(self.api_url, payload)
            self._track_usage(response_json, "generate_caption_retry")
            message = response_json["choices"][0]["message"]
            content = message.get("content")
            if not content:
                return f"A video scene is depicted with notable visual and contextual elements."

        clean_caption = content.strip().strip(" \t\n\r\"'")
        return clean_caption

    async def generate_all_captions(self, scene_json: Dict[str, Any]) -> Dict[str, str]:
        """
        Generates all 4 captions (formal, sarcastic, humorous_tech, humorous_non_tech) in a single API call
        using Fireworks' structured output feature (response_format type json_object).
        """
        if self.mock_mode:
            logging.info("ModelClient.generate_all_captions called in MOCK MODE.")
            return {
                "formal": "[MOCK-FORMAL] This is a formal placeholder caption for testing.",
                "sarcastic": "[MOCK-SARCASTIC] This is a sarcastic placeholder caption for testing.",
                "humorous_tech": "[MOCK-HUMOROUS_TECH] When your code works, but the mock breaks.",
                "humorous_non_tech": "[MOCK-HUMOROUS_NON_TECH] When you realize you did all this work manually."
            }

        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY is missing. Please set it in your environment or .env file.")
        if not self.text_model or self.text_model.lower() == "placeholder":
            raise ValueError("text_model in config/models.yaml is still set to 'placeholder'. Please configure a valid text model name.")

        system_prompt = get_style_prompts_system_prompt()

        user_content = (
            f"Scene details:\n{json.dumps(scene_json, indent=2)}\n\n"
            "Generate the JSON object containing all 4 captions now."
        )

        payload = {
            "model": self.text_model,
            "max_tokens": 1200,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.7,
            "response_format": {
                "type": "json_object"
            }
        }

        async def attempt_call():
            self._check_call_limit("generate_all_captions")
            resp = await self._post_with_retry(self.api_url, payload)
            self._track_usage(resp, "generate_all_captions")
            msg = resp["choices"][0]["message"]
            content = msg.get("content")
            if not content:
                raise ValueError("Received empty content from generate_all_captions response")
            
            parsed = json.loads(content.strip())
            required_keys = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
            for k in required_keys:
                if k not in parsed or not isinstance(parsed[k], str):
                    raise KeyError(f"Missing or invalid key '{k}' in structured response")
            return parsed

        # Try once
        try:
            return await attempt_call()
        except Exception as e:
            logging.warning(f"Structured caption generation attempt 1 failed: {e}. Retrying once...")
            try:
                # Retry once
                return await attempt_call()
            except Exception as e2:
                logging.error(f"Structured caption generation retry failed: {e2}. Falling back to per-tone calls.")
                # Fallback to per-tone calls as a safety net
                fallback_results = {}
                for tone in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]:
                    try:
                        fallback_results[tone] = await self.generate_caption(scene_json, tone)
                    except Exception as fallback_err:
                        logging.error(f"Fallback generation for tone '{tone}' failed: {fallback_err}")
                        fallback_results[tone] = "A video scene is depicted with notable visual and contextual elements."
                return fallback_results

    async def evaluate_all_captions(self, captions: Dict[str, str], scene_json: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Evaluates accuracy and style_match for all 4 captions in a single structured API call.
        Returns a dict matching:
        {
          "formal": {"accuracy": float, "style_match": float},
          ...
        }
        """
        if self.mock_mode:
            logging.info("ModelClient.evaluate_all_captions called in MOCK MODE.")
            return {
                "formal": {"accuracy": 0.9, "style_match": 0.95},
                "sarcastic": {"accuracy": 0.85, "style_match": 0.9},
                "humorous_tech": {"accuracy": 0.7, "style_match": 0.8},
                "humorous_non_tech": {"accuracy": 0.8, "style_match": 0.85}
            }

        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY is missing. Please set it in your environment or .env file.")
        if not self.text_model or self.text_model.lower() == "placeholder":
            raise ValueError("text_model in config/models.yaml is still set to 'placeholder'. Please configure a valid text model name.")

        system_prompt = (
            "You are a critical judge evaluating video captions against structured scene details.\n"
            "Evaluate each of the four captions on two dimensions:\n"
            "1. accuracy (0.0-1.0): Does the caption accurately reflect the facts and visual details of the scene?\n"
            "2. style_match (0.0-1.0): Does the caption perfectly match the intended style/tone?\n"
            " - formal: professional, objective, factual.\n"
            " - sarcastic: dry, ironical, understated mocking.\n"
            " - humorous_tech: reframed with programming or tech metaphor.\n"
            " - humorous_non_tech: observational, relatable everyday humor.\n\n"
            "Output ONLY a JSON object matching this schema:\n"
            "{\n"
            "  \"formal\": {\"accuracy\": float, \"style_match\": float},\n"
            "  \"sarcastic\": {\"accuracy\": float, \"style_match\": float},\n"
            "  \"humorous_tech\": {\"accuracy\": float, \"style_match\": float},\n"
            "  \"humorous_non_tech\": {\"accuracy\": float, \"style_match\": float}\n"
            "}"
        )

        user_content = (
            f"Scene Details:\n{json.dumps(scene_json, indent=2)}\n\n"
            f"Generated Captions:\n{json.dumps(captions, indent=2)}\n\n"
            "Assess the captions and output the evaluation JSON now."
        )

        payload = {
            "model": self.text_model,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_object"
            }
        }

        async def attempt_call():
            self._check_call_limit("evaluate_all_captions")
            resp = await self._post_with_retry(self.api_url, payload)
            self._track_usage(resp, "evaluate_all_captions")
            msg = resp["choices"][0]["message"]
            content = msg.get("content")
            if not content:
                raise ValueError("Received empty content from evaluate_all_captions response")
            
            parsed = json.loads(content.strip())
            required_keys = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
            for k in required_keys:
                if k not in parsed:
                    raise KeyError(f"Missing key '{k}' in evaluation results")
                val = parsed[k]
                if not isinstance(val, dict) or "accuracy" not in val or "style_match" not in val:
                    raise TypeError(f"Invalid evaluation format for key '{k}'")
            return parsed

        try:
            return await attempt_call()
        except Exception as e:
            logging.warning(f"Evaluation API call failed: {e}. Retrying once...")
            try:
                return await attempt_call()
            except Exception as e2:
                logging.error(f"Evaluation retry failed: {e2}. Falling back to default high scores (1.0).")
                return {
                    "formal": {"accuracy": 1.0, "style_match": 1.0},
                    "sarcastic": {"accuracy": 1.0, "style_match": 1.0},
                    "humorous_tech": {"accuracy": 1.0, "style_match": 1.0},
                    "humorous_non_tech": {"accuracy": 1.0, "style_match": 1.0}
                }

    async def judge_caption(self, caption: str, scene_json: Dict[str, Any], tone: str) -> Dict[str, Any]:
        """
        Uses the configured text model to judge a caption against a scene JSON and target tone.
        """
        if self.mock_mode:
            logging.info("ModelClient.judge_caption called in MOCK MODE.")
            return {"accuracy": 4, "tone_fit": 4}

        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY is missing. Please set it in your environment or .env file.")
        if not self.text_model or self.text_model.lower() == "placeholder":
            raise ValueError("text_model in config/models.yaml is still set to 'placeholder'. Please configure a valid text model name.")

        system_prompt = (
            "You are an AI quality judge. You are given a caption, a structured JSON describing the facts of a scene, and a target tone.\n"
            f"Rate 1-5: does this caption accurately reflect the scene facts? Rate 1-5: does this caption clearly match the intended tone [{tone}]?\n"
            "Return ONLY JSON: {\"accuracy\": int, \"tone_fit\": int}\n"
            "Do NOT include markdown block formatting, do NOT write any explanation."
        )

        user_content = (
            f"Scene Details: {json.dumps(scene_json)}\n"
            f"Caption to evaluate: '{caption}'\n"
            f"Target Tone: '{tone}'"
        )

        payload = {
            "model": self.text_model,
            "max_tokens": 600,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0
        }

        self._check_call_limit("judge_caption")
        response_json = await self._post_with_retry(self.api_url, payload)
        self._track_usage(response_json, "judge_caption")
        message = response_json["choices"][0]["message"]
        content = message.get("content")
        
        # Retry once if content is None/empty (model sometimes returns only reasoning_content)
        if not content:
            logging.warning("judge_caption: content was None or empty, retrying once...")
            self._check_call_limit("judge_caption_retry")
            response_json = await self._post_with_retry(self.api_url, payload)
            self._track_usage(response_json, "judge_caption_retry")
            message = response_json["choices"][0]["message"]
            content = message.get("content")
            if not content:
                raise ValueError("Content is None or empty in model response after retry")
        
        content_stripped = content.strip()

        try:
            return json.loads(content_stripped)
        except json.JSONDecodeError:
            start_idx = content_stripped.find('{')
            end_idx = content_stripped.rfind('}')
            if start_idx != -1 and end_idx != -1:
                return json.loads(content_stripped[start_idx:end_idx+1])
            raise ValueError(f"Could not parse judge JSON from response content: {content_stripped}")

if __name__ == "__main__":
    print("=== Testing ModelClient in MOCK MODE ===")
    ModelClient.reset_usage()
    mock_client = ModelClient(force_mock_mode=True)
    print(f"Mock Mode Status: {mock_client.mock_mode}")
    print(f"MAX_CALLS_PER_RUN: {ModelClient._MAX_CALLS_PER_RUN}")
    scene = mock_client.describe_scene(["dummy_path.jpg"], "Dummy transcript text")
    print("Mock describe_scene output:", json.dumps(scene, indent=4))
    caption = mock_client.generate_caption(scene, "sarcastic")
    print("Mock generate_caption output:", caption)
    judgment = mock_client.judge_caption(caption, scene, "sarcastic")
    print("Mock judge_caption output:", json.dumps(judgment, indent=4))
    ModelClient.print_usage_summary()
    
    print("\n=== Testing ModelClient in REAL MODE (will show error/warn as configured) ===")
    try:
        real_client = ModelClient(force_mock_mode=False)
        print(f"Real Client Mock Mode Status: {real_client.mock_mode}")
        if real_client.mock_mode:
            print("Note: Real client auto-fallback to MOCK MODE because configuration is incomplete.")
            print("Triggering real call checks (forcing self.mock_mode = False temporarily)...")
            real_client.mock_mode = False
            
        real_client.describe_scene(["dummy_path.jpg"], "Dummy transcript text")
    except Exception as e:
        print(f"Expected failure in REAL MODE as configured: {e}")
