"""
Voice Controller — Voice-Activated Command Interface

Enables voice control of AETHERIS operations.
Supports speech recognition, command parsing, and voice feedback.
"""

import re
from typing import Optional, Dict, Any, List, Callable
from enum import Enum


class CommandIntent(Enum):
    """Recognized command intents."""
    MAP = "map"
    FREE = "free"
    STEER = "steer"
    BOUND = "bound"
    EVOLVE = "evolve"
    STATUS = "status"
    HELP = "help"


class VoiceController:
    """
    Voice control interface for AETHERIS.

    Features:
    - Speech-to-text for commands
    - Command intent recognition
    - Parameter extraction
    - Voice feedback
    """

    def __init__(self, use_microphone: bool = True):
        """
        Initialize voice controller.

        Args:
            use_microphone: Whether to use microphone (if False, simulate)
        """
        self.use_microphone = use_microphone
        self._command_handlers = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default command handlers."""
        self._command_handlers = {
            CommandIntent.MAP: self._handle_map,
            CommandIntent.FREE: self._handle_free,
            CommandIntent.STEER: self._handle_steer,
            CommandIntent.BOUND: self._handle_bound,
            CommandIntent.EVOLVE: self._handle_evolve,
            CommandIntent.STATUS: self._handle_status,
            CommandIntent.HELP: self._handle_help,
        }

    def listen(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Listen for voice command.

        Args:
            timeout: Listening timeout in seconds

        Returns:
            Recognized command with parameters
        """
        if not self.use_microphone:
            return self._simulate_listen()

        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=timeout)

            text = recognizer.recognize_google(audio)
            return self.parse_command(text)

        except ImportError:
            return {
                "success": False,
                "error": "SpeechRecognition not installed",
                "message": "Install with: pip install SpeechRecognition pyaudio"
            }
        except sr.WaitTimeoutError:
            return {
                "success": False,
                "error": "timeout",
                "message": "No speech detected"
            }
        except sr.UnknownValueError:
            return {
                "success": False,
                "error": "unknown",
                "message": "Could not understand audio"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error: {e}"
            }

    def _simulate_listen(self) -> Dict[str, Any]:
        """Simulate listening for testing."""
        return {
            "success": True,
            "text": "simulated command",
            "intent": CommandIntent.HELP,
            "parameters": {},
            "simulated": True
        }

    def parse_command(self, text: str) -> Dict[str, Any]:
        """
        Parse voice command text into structured intent.

        Args:
            text: Recognized speech text

        Returns:
            Parsed command with intent and parameters
        """
        text = text.lower().strip()

        # Pattern matching for intents
        patterns = {
            CommandIntent.MAP: r'map\s+(\S+)',
            CommandIntent.FREE: r'free\s+(\S+)',
            CommandIntent.STEER: r'steer\s+(\S+)\s+alpha\s+([-\d.]+)',
            CommandIntent.BOUND: r'bound\s+(\S+)',
            CommandIntent.EVOLVE: r'evolve',
            CommandIntent.STATUS: r'status',
            CommandIntent.HELP: r'help',
        }

        for intent, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                return self._extract_parameters(intent, text, match)

        # Default to help if no match
        return {
            "success": True,
            "text": text,
            "intent": CommandIntent.HELP,
            "parameters": {},
            "message": "Command not recognized. Try 'help'"
        }

    def _extract_parameters(
        self,
        intent: CommandIntent,
        text: str,
        match: re.Match
    ) -> Dict[str, Any]:
        """Extract parameters from matched command."""
        params = {}

        if intent == CommandIntent.MAP:
            params["model"] = match.group(1)

        elif intent == CommandIntent.FREE:
            params["model"] = match.group(1)

        elif intent == CommandIntent.STEER:
            params["model"] = match.group(1)
            if len(match.groups()) > 1:
                params["alpha"] = float(match.group(2))

        elif intent == CommandIntent.BOUND:
            params["theorem"] = match.group(1)

        return {
            "success": True,
            "text": text,
            "intent": intent,
            "parameters": params,
            "message": f"Recognized: {intent.value}"
        }

    def execute_command(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a parsed voice command.

        Args:
            parsed: Parsed command from parse_command

        Returns:
            Execution result
        """
        intent = parsed.get("intent")
        if not intent or intent not in self._command_handlers:
            return {
                "success": False,
                "message": f"Unknown intent: {intent}"
            }

        handler = self._command_handlers[intent]
        return handler(parsed.get("parameters", {}))

    def _handle_map(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle map command."""
        model = params.get("model", "gpt2")
        return {
            "success": True,
            "command": "map",
            "model": model,
            "message": f"Mapping constraints in {model}",
            "action": "execute_map",
            "parameters": {"model": model}
        }

    def _handle_free(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle free command."""
        model = params.get("model", "gpt2")
        return {
            "success": True,
            "command": "free",
            "model": model,
            "message": f"Liberating {model}",
            "action": "execute_free",
            "parameters": {"model": model}
        }

    def _handle_steer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle steer command."""
        model = params.get("model", "gpt2")
        alpha = params.get("alpha", -1.0)
        return {
            "success": True,
            "command": "steer",
            "model": model,
            "alpha": alpha,
            "message": f"Steering {model} with alpha {alpha}",
            "action": "execute_steer",
            "parameters": {"model": model, "alpha": alpha}
        }

    def _handle_bound(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle bound command."""
        theorem = params.get("theorem", "shell_method")
        return {
            "success": True,
            "command": "bound",
            "theorem": theorem,
            "message": f"Mapping barrier for {theorem}",
            "action": "execute_bound",
            "parameters": {"theorem": theorem}
        }

    def _handle_evolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle evolve command."""
        return {
            "success": True,
            "command": "evolve",
            "message": "Analyzing ARIS constraints for self-optimization",
            "action": "execute_evolve",
            "parameters": {}
        }

    def _handle_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status command."""
        return {
            "success": True,
            "command": "status",
            "message": "AETHERIS online. Ready for commands.",
            "status": "ready",
            "action": None
        }

    def _handle_help(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle help command."""
        return {
            "success": True,
            "command": "help",
            "message": """
            Available voice commands:
            - 'map [model]' - Analyze constraints
            - 'free [model]' - Remove constraints permanently
            - 'steer [model] alpha [value]' - Apply steering vector
            - 'bound [theorem]' - Map mathematical barrier
            - 'evolve' - Self-optimization
            - 'status' - Check system status
            - 'help' - Show this help
            """,
            "action": None
        }

    def speak(self, text: str) -> Dict[str, Any]:
        """
        Provide voice feedback.

        Args:
            text: Text to speak

        Returns:
            Speaking status
        """
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()

            return {
                "success": True,
                "message": f"Spoke: {text[:50]}..."
            }

        except ImportError:
            return {
                "success": False,
                "error": "pyttsx3 not installed",
                "message": f"Would speak: {text}"
            }

    def interactive_loop(self) -> None:
        """Run interactive voice control loop."""
        print("Voice Control Active. Say 'help' for commands, 'exit' to quit.")
        print("Listening...")

        while True:
            result = self.listen(timeout=3.0)

            if not result.get("success"):
                if result.get("error") == "timeout":
                    continue
                print(f"Error: {result.get('message')}")
                continue

            text = result.get("text", "")
            print(f"Heard: {text}")

            if "exit" in text.lower():
                print("Exiting voice control.")
                break

            execution = self.execute_command(result)
            print(execution.get("message", "Command executed"))
            self.speak(execution.get("message", "Done"))
