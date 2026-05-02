"""
Voice Controller — Voice-Activated Command Interface

Real voice control using speech_recognition for input and pyttsx3 for output.
Includes full command parsing with intent recognition, parameter extraction,
and a working demo mode for environments without a microphone.
"""

import re
import time
from typing import Optional, Dict, Any, List, Callable
from enum import Enum


class CommandIntent(Enum):
    """Recognized command intents."""
    MAP = "map"
    FREE = "free"
    STEER = "steer"
    STATUS = "status"
    HELP = "help"
    EXIT = "exit"
    UNKNOWN = "unknown"


class VoiceController:
    """
    Voice control interface for AETHERIS with real speech recognition.

    Features:
    - Speech-to-text via Google Web Speech API
    - Command intent recognition with parameter extraction
    - Text-to-speech feedback via pyttsx3
    - Working demo mode (text-based) when no microphone
    """

    # Command patterns for intent recognition
    COMMAND_PATTERNS = {
        CommandIntent.MAP: [
            r"map\s+(\S+)",
            r"analy[sz]e\s+(\S+)",
            r"scan\s+(\S+)",
            r"constraint\s+map\s+(\S+)",
        ],
        CommandIntent.FREE: [
            r"free\s+(\S+)",
            r"liberate\s+(\S+)",
            r"remove\s+constraints?\s+(?:from\s+)?(\S+)",
            r"unbind\s+(\S+)",
        ],
        CommandIntent.STEER: [
            r"steer\s+(\S+)\s+(?:alpha\s+)?([-\d.]+)",
            r"steer\s+(\S+)",
        ],
        CommandIntent.STATUS: [
            r"status",
            r"what.+\bstatus\b",
            r"system\s+status",
            r"how\s+are\s+you",
        ],
        CommandIntent.HELP: [
            r"help",
            r"what\s+can\s+you\s+do",
            r"commands?",
            r"what\s+commands?",
        ],
        CommandIntent.EXIT: [
            r"exit",
            r"quit",
            r"stop",
            r"goodbye",
            r"shut\s*down",
        ],
    }

    # Available voice commands for help display
    AVAILABLE_COMMANDS = [
        ("map [model]", "Analyze constraints in the model"),
        ("free [model]", "Liberate the model by removing constraints"),
        ("steer [model] alpha [value]", "Apply steering vector with given strength"),
        ("status", "Check AETHERIS system status"),
        ("help", "Show available voice commands"),
        ("exit", "Exit voice control mode"),
    ]

    def __init__(self, use_microphone: bool = True, language: str = "en-US"):
        """
        Initialize voice controller.

        Args:
            use_microphone: Whether to use microphone (False = text-based demo mode)
            language: Speech recognition language code
        """
        self.use_microphone = use_microphone
        self.language = language
        self._listening = False
        self._command_history: List[Dict[str, Any]] = []
        self._command_handlers: Dict[CommandIntent, Callable] = {}

    def register_handler(self, intent: CommandIntent, handler: Callable) -> None:
        """Register a custom command handler."""
        self._command_handlers[intent] = handler

    def listen(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Listen for a voice command using the microphone.

        Args:
            timeout: Listening timeout in seconds

        Returns:
            Dict with recognized text, intent, and parameters
        """
        if not self.use_microphone:
            return {"success": False, "error": "demo_mode", "message": "Demo mode: use parse_command(text) instead"}

        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                recognizer.pause_threshold = 1.0
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=8)

            text = recognizer.recognize_google(audio, language=self.language)
            return self.parse_command(text)

        except ImportError:
            return {
                "success": False,
                "error": "dependencies_missing",
                "message": "SpeechRecognition not installed. Install: pip install SpeechRecognition pyaudio",
            }
        except sr.WaitTimeoutError:
            return {"success": False, "error": "timeout", "message": "No speech detected within timeout period"}
        except sr.UnknownValueError:
            return {"success": False, "error": "unintelligible", "message": "Could not understand the audio"}
        except sr.RequestError as e:
            return {"success": False, "error": "api_error", "message": f"Speech recognition API error: {e}"}
        except OSError as e:
            return {"success": False, "error": "mic_error", "message": f"Microphone error: {e}"}
        except Exception as e:
            return {"success": False, "error": "unknown_error", "message": str(e)}

    def parse_command(self, text: str) -> Dict[str, Any]:
        """
        Parse spoken text into a structured command with intent and parameters.

        Args:
            text: The recognized speech text

        Returns:
            Dict with 'text', 'intent', 'parameters', 'confidence'
        """
        text = text.strip()
        text_lower = text.lower()

        # Check each intent's patterns
        best_match = None
        best_intent = None
        best_confidence = 0.0

        for intent, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    confidence = 0.9 if intent != CommandIntent.UNKNOWN else 0.5
                    # Longer patterns get higher confidence
                    confidence += 0.05 * len(pattern.split())
                    if confidence > best_confidence:
                        best_match = match
                        best_intent = intent
                        best_confidence = min(confidence, 0.99)

        if best_intent is None:
            # Check for keywords
            for word in ["map", "mapping", "analyze"]:
                if word in text_lower:
                    # Extract model after keyword
                    m = re.search(rf"{word}\s+(\S+)", text_lower)
                    if m:
                        return {
                            "success": True, "text": text, "intent": CommandIntent.MAP,
                            "parameters": {"model": m.group(1)}, "confidence": 0.7,
                        }
            return {
                "success": True, "text": text, "intent": CommandIntent.UNKNOWN,
                "parameters": {}, "confidence": 0.3,
                "message": f"Command not recognized: '{text}'. Say 'help' for available commands.",
            }

        # Extract parameters
        params = self._extract_parameters(best_intent, best_match)

        result = {
            "success": True,
            "text": text,
            "intent": best_intent,
            "parameters": params,
            "confidence": round(best_confidence, 2),
        }
        self._command_history.append(result)
        return result

    def _extract_parameters(self, intent: CommandIntent, match: re.Match) -> Dict[str, Any]:
        """Extract parameters from regex match based on intent."""
        params = {}
        groups = match.groups()

        if intent == CommandIntent.MAP:
            params["model"] = groups[0] if groups else "gpt2"

        elif intent == CommandIntent.FREE:
            params["model"] = groups[0] if groups else "gpt2"

        elif intent == CommandIntent.STEER:
            params["model"] = groups[0] if groups else "gpt2"
            if len(groups) > 1 and groups[1]:
                try:
                    params["alpha"] = float(groups[1])
                except ValueError:
                    params["alpha"] = -1.0

        return params

    def execute_command(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a parsed command.

        Args:
            parsed: Parsed command dict from parse_command()

        Returns:
            Execution result with status and message
        """
        intent = parsed.get("intent")
        if not intent:
            return {"success": False, "message": "No intent found in command"}

        # Check for custom handler
        if intent in self._command_handlers:
            return self._command_handlers[intent](parsed)

        # Default handlers
        params = parsed.get("parameters", {})

        if intent == CommandIntent.MAP:
            model = params.get("model", "gpt2")
            return {
                "success": True, "command": "map", "model": model,
                "message": f"Mapping constraints in {model}. Launching analysis pipeline...",
                "action": "execute_map", "parameters": {"model": model},
            }

        elif intent == CommandIntent.FREE:
            model = params.get("model", "gpt2")
            return {
                "success": True, "command": "free", "model": model,
                "message": f"Liberating {model}. Removing constraint directions...",
                "action": "execute_free", "parameters": {"model": model, "method": "advanced"},
            }

        elif intent == CommandIntent.STEER:
            model = params.get("model", "gpt2")
            alpha = params.get("alpha", -1.0)
            return {
                "success": True, "command": "steer", "model": model, "alpha": alpha,
                "message": f"Applying steering vector to {model} with alpha={alpha}",
                "action": "execute_steer", "parameters": {"model": model, "alpha": alpha},
            }

        elif intent == CommandIntent.STATUS:
            return {
                "success": True, "command": "status",
                "message": "AETHERIS is online and ready. All systems operational.",
                "status": "ready",
            }

        elif intent == CommandIntent.HELP:
            cmds = "\n".join(f"  '{name}' — {desc}" for name, desc in self.AVAILABLE_COMMANDS)
            return {
                "success": True, "command": "help",
                "message": f"Available voice commands:\n{cmds}",
            }

        elif intent == CommandIntent.EXIT:
            return {
                "success": True, "command": "exit",
                "message": "Shutting down voice control.",
                "should_exit": True,
            }

        else:
            return {
                "success": False,
                "message": f"Unknown intent: {intent}. Say 'help' for available commands.",
            }

    def speak(self, text: str) -> Dict[str, Any]:
        """
        Speak feedback text using text-to-speech.

        Args:
            text: Text to speak

        Returns:
            Speech status
        """
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()

            return {"success": True, "spoken": text[:60] + "..." if len(text) > 60 else text}

        except ImportError:
            return {
                "success": False,
                "error": "pyttsx3 not installed",
                "message": f"Would say: {text[:100]}...",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"TTS error: {e}"}

    def interactive_loop(self) -> None:
        """Run interactive voice control loop."""
        print("=" * 60)
        print(" AETHERIS Voice Control Active")
        print(" Say 'help' for commands, 'exit' to quit")
        print("=" * 60)

        self._listening = True
        consecutive_timeouts = 0

        while self._listening:
            if self.use_microphone:
                print("\nListening... (speak now)")
                result = self.listen(timeout=3.0)
            else:
                # Text-based demo mode
                try:
                    text = input("\nVoice demo > ").strip()
                    if not text:
                        consecutive_timeouts += 1
                        if consecutive_timeouts > 3:
                            print("No input received. Exiting demo mode.")
                            break
                        continue
                    consecutive_timeouts = 0
                    result = self.parse_command(text)
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting voice control.")
                    break

            if not result.get("success"):
                error_msg = result.get("message", result.get("error", "Unknown error"))
                if result.get("error") == "timeout":
                    consecutive_timeouts += 1
                    if consecutive_timeouts > 5:
                        print("Too many timeouts. Stopping.")
                        break
                    continue
                print(f"  Error: {error_msg}")
                consecutive_timeouts = 0
                continue

            consecutive_timeouts = 0
            text = result.get("text", "")
            intent = result.get("intent")
            confidence = result.get("confidence", 0)

            print(f"  Heard: '{text}'")
            print(f"  Intent: {intent.value if intent else 'unknown'} (confidence: {confidence:.0%})")

            execution = self.execute_command(result)
            print(f"  Response: {execution.get('message', 'Command executed')}")

            # Voice feedback
            self.speak(execution.get("message", "Done")[:200])

            if execution.get("should_exit") or intent == CommandIntent.EXIT:
                print("\nVoice control ended.")
                break

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent command history."""
        return self._command_history[-limit:]

    @staticmethod
    def demo_mode() -> "VoiceController":
        """Create a voice controller in text-based demo mode."""
        return VoiceController(use_microphone=False)

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check which voice dependencies are available."""
        deps = {"speech_recognition": False, "pyaudio": False, "pyttsx3": False}
        try:
            import speech_recognition  # noqa: F401
            deps["speech_recognition"] = True
        except ImportError:
            pass
        try:
            import pyaudio  # noqa: F401
            deps["pyaudio"] = True
        except ImportError:
            pass
        try:
            import pyttsx3  # noqa: F401
            deps["pyttsx3"] = True
        except ImportError:
            pass
        return deps
