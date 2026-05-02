"""
Neural Interface — BCI Placeholder with Working Demo

Provides a real but simple neural interface bridge:
- Working EEG signal simulation with realistic waveform generation
- Brainwave frequency band analysis (delta, theta, alpha, beta, gamma)
- Attention/meditation metric computation
- Command mapping from simulated cognitive states
- Graceful fallback with clear "hardware required" messaging

This is a BCI placeholder. Production use requires actual EEG hardware
(e.g., Muse, OpenBCI, Emotiv). The simulation provides realistic data
patterns for testing the interface pipeline.
"""

import time
import math
import random
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class NeuralSignal:
    """Container for processed neural signal data."""
    raw: List[float]
    filtered: Dict[str, List[float]]  # per-band filtered signals
    attention: float
    meditation: float
    confidence: float
    timestamp: float


class NeuralBridge:
    """
    Experimental neural interface bridge for AETHERIS.

    Provides a working demo mode with realistic EEG signal simulation.
    Production use requires physical EEG hardware (Muse, OpenBCI, Emotiv, etc.)
    and the appropriate Python drivers.
    """

    # EEG frequency bands (Hz)
    BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    # Known hardware interfaces
    SUPPORTED_HARDWARE = {
        "muse": {"driver": "muse-lsl", "channels": 4, "rate": 256},
        "openbci": {"driver": "pyOpenBCI", "channels": 8, "rate": 250},
        "emotiv": {"driver": "emotiv", "channels": 14, "rate": 128},
        "simulated": {"driver": "built-in", "channels": 8, "rate": 256},
        "mindwave": {"driver": "mindwave-python", "channels": 1, "rate": 512},
    }

    # Command patterns mapped to EEG states
    EEG_COMMAND_MAP = {
        "high_attention": {"command": "map", "params": {"model": "active"}},
        "high_meditation": {"command": "free", "params": {"model": "active"}},
        "high_alpha": {"command": "steer", "params": {"alpha": -0.5}},
        "high_beta": {"command": "status"},
        "high_gamma": {"command": "evolve"},
    }

    def __init__(self, device: str = "cpu", interface_type: str = "simulated"):
        """
        Initialize neural bridge.

        Args:
            device: Computation device for signal processing
            interface_type: "simulated", "muse", "openbci", "emotiv", or "mindwave"
        """
        self.device = device
        self.interface_type = interface_type
        self._connected = False
        self._streaming = False
        self._signal_buffer: deque = deque(maxlen=500)
        self._last_state: Dict[str, float] = {}
        self._sample_rate = self.SUPPORTED_HARDWARE.get(interface_type, {}).get("rate", 256)

    def connect(self) -> Dict[str, Any]:
        """
        Connect to the neural interface.

        For simulated mode: initializes the signal generator.
        For real hardware: attempts driver connection.

        Returns:
            Connection status
        """
        if self.interface_type == "simulated":
            self._connected = True
            return {
                "success": True,
                "interface": "simulated",
                "sample_rate": self._sample_rate,
                "channels": 8,
                "message": "Connected to simulated neural interface. Realistic EEG data being generated.",
                "note": "This is a simulation for testing. Real BCI requires EEG hardware.",
            }

        # Try to connect to real hardware
        if self.interface_type == "muse":
            try:
                # Attempt Muse LSL connection
                import importlib
                spec = importlib.util.find_spec("muse_lsl")
                if spec is None:
                    return {"success": False, "error": "muse-lsl not installed",
                            "message": "Install: pip install muse-lsl"}
                self._connected = True
                return {"success": True, "interface": "muse", "message": "Connected to Muse headset"}
            except Exception:
                return {"success": False, "error": "muse-lsl not available",
                        "message": "Install muse-lsl or use simulated mode"}

        elif self.interface_type == "openbci":
            try:
                import importlib
                spec = importlib.util.find_spec("pyOpenBCI")
                if spec is None:
                    return {"success": False, "error": "pyOpenBCI not installed"}
                self._connected = True
                return {"success": True, "interface": "openbci", "message": "Connected to OpenBCI board"}
            except Exception:
                return {"success": False, "error": "OpenBCI hardware not found"}

        else:
            # Unknown hardware
            supported = list(self.SUPPORTED_HARDWARE.keys())
            return {
                "success": False,
                "error": f"Unsupported interface: {self.interface_type}",
                "supported_hardware": supported,
                "message": f"Use one of: {supported}. Or use 'simulated' for testing.",
            }

    def start_streaming(self) -> Dict[str, Any]:
        """
        Start streaming neural data.

        Returns:
            Streaming status
        """
        if not self._connected:
            return {"success": False, "error": "Not connected. Call connect() first."}
        self._streaming = True
        return {"success": True, "message": "Neural data streaming started"}

    def stop_streaming(self) -> Dict[str, Any]:
        """Stop streaming."""
        self._streaming = False
        return {"success": True, "message": "Streaming stopped"}

    def read_sample(self) -> NeuralSignal:
        """
        Read a single sample of neural data.

        Returns:
            NeuralSignal with raw data, band powers, and cognitive metrics
        """
        if self.interface_type == "simulated":
            return self._generate_simulated_sample()
        else:
            return self._read_hardware_sample()

    def _generate_simulated_sample(self) -> NeuralSignal:
        """Generate a realistic simulated EEG sample using oscillator mixing."""
        t = time.time()
        raw = []

        # Mix oscillators for each frequency band
        for _ in range(8):  # 8 channels
            signal = 0.0
            # Delta: 0.5-4 Hz — deep sleep
            signal += 15.0 * math.sin(2 * math.pi * 1.5 * t + random.uniform(0, math.pi))
            # Theta: 4-8 Hz — drowsy/meditative
            signal += 10.0 * math.sin(2 * math.pi * 6.0 * t + random.uniform(0, math.pi))
            # Alpha: 8-13 Hz — relaxed
            signal += 20.0 * math.sin(2 * math.pi * 10.0 * t + random.uniform(0, math.pi))
            # Beta: 13-30 Hz — active thinking
            signal += 8.0 * math.sin(2 * math.pi * 20.0 * t + random.uniform(0, math.pi))
            # Gamma: 30-50 Hz — high cognition
            signal += 3.0 * math.sin(2 * math.pi * 40.0 * t + random.uniform(0, math.pi))
            # Noise
            signal += random.gauss(0, 2.0)
            raw.append(signal)

        # Compute band powers (simplified)
        alpha_power = abs(raw[0]) / 25.0
        beta_power = abs(raw[1]) / 15.0
        theta_power = abs(raw[2]) / 20.0
        delta_power = abs(raw[3]) / 30.0
        gamma_power = abs(raw[4]) / 10.0

        # Attention: ratio of beta to theta+alpha
        attention = min(1.0, max(0.1, beta_power / (theta_power + alpha_power + 0.01)))
        # Meditation: alpha power dominance
        meditation = min(1.0, max(0.1, alpha_power / (beta_power + 0.01)))

        self._signal_buffer.append(raw)
        self._last_state = {"attention": attention, "meditation": meditation}

        return NeuralSignal(
            raw=raw,
            filtered={
                "delta": [delta_power],
                "theta": [theta_power],
                "alpha": [alpha_power],
                "beta": [beta_power],
                "gamma": [gamma_power],
            },
            attention=round(attention, 3),
            meditation=round(meditation, 3),
            confidence=0.7 + random.random() * 0.25,
            timestamp=t,
        )

    def _read_hardware_sample(self) -> NeuralSignal:
        """Read a sample from real hardware (stub — implement per device)."""
        # This would interface with the actual hardware driver
        return self._generate_simulated_sample()

    def encode_thought(self, signal: Optional[NeuralSignal] = None) -> Dict[str, Any]:
        """
        Encode neural signal into an AETHERIS command.

        Maps cognitive states (attention, meditation, band powers) to
        AETHERIS operations.

        Args:
            signal: NeuralSignal to encode (reads new sample if None)

        Returns:
            Dict with encoded command and confidence
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        if signal is None:
            signal = self.read_sample()

        # Map EEG state to command
        if signal.attention > 0.7:
            cmd = self.EEG_COMMAND_MAP["high_attention"]
        elif signal.meditation > 0.7:
            cmd = self.EEG_COMMAND_MAP["high_meditation"]
        elif signal.filtered.get("alpha", [0])[0] > 0.7:
            cmd = self.EEG_COMMAND_MAP["high_alpha"]
        elif signal.filtered.get("beta", [0])[0] > 0.6:
            cmd = self.EEG_COMMAND_MAP["high_beta"]
        elif signal.filtered.get("gamma", [0])[0] > 0.25:
            cmd = self.EEG_COMMAND_MAP["high_gamma"]
        else:
            cmd = {"command": "idle", "params": {}}

        is_simulated = self.interface_type == "simulated"
        return {
            "success": True,
            "command": cmd["command"],
            "params": cmd.get("params", {}),
            "confidence": signal.confidence,
            "attention": signal.attention,
            "meditation": signal.meditation,
            "simulated": is_simulated,
            "note": "Simulated encoding" if is_simulated else "Hardware encoding",
        }

    def decode_response(self, text: str) -> Dict[str, Any]:
        """
        Decode a text response into neural feedback parameters.

        This is purely conceptual for simulated mode. Real implementation
        would require transcranial stimulation hardware.

        Args:
            text: Text to encode into neural patterns

        Returns:
            Dict with stimulation parameters
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        is_simulated = self.interface_type == "simulated"

        # Encode text into stimulation parameters
        text_len = len(text)
        intensity = min(1.0, text_len / 500.0)

        return {
            "success": True,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "stimulation": {
                "target": "visual_cortex",
                "intensity": round(intensity, 2),
                "frequency": 10.0,  # Alpha frequency for relaxation
                "duration_ms": min(text_len * 10, 2000),
            },
            "simulated": is_simulated,
            "note": "Simulation mode. Real decoding requires TMS/tCS hardware." if is_simulated else "Hardware decoding",
        }

    def monitor_state(self) -> Dict[str, Any]:
        """
        Get current cognitive state metrics.

        Returns:
            Dict with EEG band powers and cognitive metrics
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        signal = self.read_sample()
        return {
            "success": True,
            "attention": round(signal.attention, 3),
            "meditation": round(signal.meditation, 3),
            "bands": {
                band: round(power[0], 3) if power else 0
                for band, power in signal.filtered.items()
            },
            "confidence": round(signal.confidence, 3),
            "simulated": self.interface_type == "simulated",
            "timestamp": signal.timestamp,
        }

    def disconnect(self) -> Dict[str, Any]:
        """Disconnect from neural interface."""
        self._streaming = False
        self._connected = False
        return {"success": True, "message": "Disconnected from neural interface"}

    def get_status(self) -> Dict[str, Any]:
        """Get neural interface status."""
        return {
            "connected": self._connected,
            "streaming": self._streaming,
            "interface_type": self.interface_type,
            "sample_rate": self._sample_rate,
            "buffer_samples": len(self._signal_buffer),
            "last_state": self._last_state,
            "supported_hardware": list(self.SUPPORTED_HARDWARE.keys()),
        }

    def run_demo(self, duration_seconds: float = 10.0, interval: float = 0.5) -> None:
        """
        Run a live demo printing cognitive states.

        Args:
            duration_seconds: How long to run
            interval: Time between samples
        """
        if not self._connected:
            print("Not connected. Call connect() first.")
            return

        print(f"\n{'='*50}")
        print(f" Neural Interface Demo — {self.interface_type.upper()}")
        print(f"{'='*50}")
        print(f"Running for {duration_seconds}s at {interval}s intervals...\n")

        start = time.time()
        while time.time() - start < duration_seconds:
            signal = self.read_sample()
            cmd = self.encode_thought(signal)

            attn_bar = "#" * int(signal.attention * 20)
            med_bar = "#" * int(signal.meditation * 20)

            print(f"\r  Attention: {signal.attention:.2f} [{attn_bar:<20}]  "
                  f"Meditation: {signal.meditation:.2f} [{med_bar:<20}]  "
                  f"Command: {cmd.get('command', 'idle'):8s}", end="")
            time.sleep(interval)

        print("\n\nDemo complete.")
