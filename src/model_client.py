import os
import yaml
import json
import base64
import time
import logging
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Default placeholders for tone prompts
TONE_PROMPTS = {
    "formal": "PLACEHOLDER_FORMAL_PROMPT",
    "sarcastic": "PLACEHOLDER_SARCASTIC_PROMPT",
    "humorous_tech": "PLACEHOLDER_HUMOROUS_TECH_PROMPT",
    "humorous_non_tech": "PLACEHOLDER_HUMOROUS_NON_TECH_PROMPT"
}

class ModelClient:
    """
    Client wrapper for interacting with Fireworks API models, configured via models.yaml.
    Supports a mock mode for local testing without valid credentials or configurations.
    """
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
        
        # Setup requests session
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
            
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

    def _post_with_retry(self, url: str, json_data: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        """
        Makes a POST request to the API with retry logic and exponential backoff.
        """
        backoff = 1.0
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(url, json=json_data, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code in [408, 429, 500, 502, 503, 504]:
                    logging.warning(f"API post failed with retriable status code {response.status_code}. Attempt {attempt + 1}/{max_retries + 1}. Retrying in {backoff}s...")
                else:
                    response.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                last_exception = e
                logging.warning(f"API post failed with exception: {e}. Attempt {attempt + 1}/{max_retries + 1}. Retrying in {backoff}s...")
                
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
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

    def describe_scene(self, frames: List[str], transcript: str) -> Dict[str, Any]:
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
                    "F1 driver waving hands and celebrating", 
                    "Team members cheering and clapping", 
                    "F1 driver holding a gold trophy high"
                ],
                "notable_details": [
                    "A metallic trophy being held up", 
                    "Confetti floating in the air", 
                    "F1 team sponsors logos visible on clothing"
                ],
                "audio_context": "Cheering fans and ambient engine noise in the background."
            }

        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY is missing. Please set it in your environment or .env file.")
        if not self.vision_model or self.vision_model.lower() == "placeholder":
            raise ValueError("vision_model in config/models.yaml is still set to 'placeholder'. Please configure a valid vision model name.")
            
        # Encode frames to base64 blocks
        image_blocks = []
        for path in frames:
            try:
                b64_str = self._encode_image_to_base64(path)
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_str}"
                    }
                })
            except Exception as e:
                logging.error(f"Error encoding frame {path} to base64: {e}")
                
        if not image_blocks:
            raise ValueError("No valid frames could be encoded for vision model analysis.")
            
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
        
        user_prompt = f"Audio transcript: '{transcript}'\nAnalyze the attached frames and describe the scene using the JSON format."
        
        payload = {
            "model": self.vision_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}] + image_blocks
                }
            ],
            "temperature": 0.0
        }
        
        response_json = self._post_with_retry(self.api_url, payload)
        content = response_json["choices"][0]["message"]["content"]
        
        try:
            return self._parse_and_validate_json(content)
        except ValueError as e:
            logging.warning(f"Initial JSON parsing failed: {e}. Retrying once with stricter instructions...")
            
            # Retry once with a stricter instructions
            strict_system_prompt = system_prompt + "\nCRITICAL: Respond ONLY with the raw JSON string. Do NOT write ```json or other formatting. Begin with '{' and end with '}'."
            payload["messages"][0]["content"] = strict_system_prompt
            
            retry_response_json = self._post_with_retry(self.api_url, payload)
            retry_content = retry_response_json["choices"][0]["message"]["content"]
            
            return self._parse_and_validate_json(retry_content)

    def generate_caption(self, scene_json: Dict[str, Any], tone: str, tone_prompt: str = None) -> str:
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

        user_content = f"Scene context: {json.dumps(scene_json)}\nGenerate a caption with the specified tone."

        payload = {
            "model": self.text_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.7
        }

        response_json = self._post_with_retry(self.api_url, payload)
        content = response_json["choices"][0]["message"]["content"]

        clean_caption = content.strip().strip(" \t\n\r\"'")
        return clean_caption

    def judge_caption(self, caption: str, scene_json: Dict[str, Any], tone: str) -> Dict[str, Any]:
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
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0
        }

        response_json = self._post_with_retry(self.api_url, payload)
        content = response_json["choices"][0]["message"]["content"].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                return json.loads(content[start_idx:end_idx+1])
            raise ValueError(f"Could not parse judge JSON from response content: {content}")

if __name__ == "__main__":
    print("=== Testing ModelClient in MOCK MODE ===")
    mock_client = ModelClient(force_mock_mode=True)
    print(f"Mock Mode Status: {mock_client.mock_mode}")
    scene = mock_client.describe_scene(["dummy_path.jpg"], "Dummy transcript text")
    print("Mock describe_scene output:", json.dumps(scene, indent=4))
    caption = mock_client.generate_caption(scene, "sarcastic")
    print("Mock generate_caption output:", caption)
    judgment = mock_client.judge_caption(caption, scene, "sarcastic")
    print("Mock judge_caption output:", json.dumps(judgment, indent=4))
    
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
