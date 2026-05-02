"""
Holographic Visualizer — 3D Constraint Geometry Visualization

Real 3D visualization of constraint geometry using matplotlib for static
plots and three.js for interactive WebGL browser visualization.
Includes direction vectors, solid angle cones, cross-layer heatmaps,
and barrier surface rendering.
"""

import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class VisualizationConfig:
    """Configuration for 3D visualization."""
    title: str = "Constraint Geometry"
    width: int = 800
    height: int = 600
    background_color: str = "#0a0a1a"
    point_size: float = 5.0
    line_width: float = 1.0
    show_axes: bool = True
    show_grid: bool = True
    dpi: int = 150
    colors: List[str] = field(default_factory=lambda: [
        "#ff3366", "#33ff66", "#3366ff", "#ff33ff", "#33ffff", "#ff9933",
    ])


class HolographicViz:
    """
    Generate 3D visualizations of constraint geometry for AETHERIS.

    Features:
    - 3D direction vector rendering (matplotlib)
    - Solid angle cone visualization
    - Cross-layer alignment heatmaps (seaborn)
    - Barrier surface geometry
    - Interactive WebGL/Three.js export for browser viewing
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self._figures: List[Any] = []

    def render_directions_3d(
        self,
        directions: List[np.ndarray],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Render 3D geometry of constraint direction vectors.

        Each direction is normalized and displayed as an arrow from origin
        to the corresponding point on the unit sphere.

        Args:
            directions: List of direction vectors (at least 3D components)
            labels: Optional labels for legend
            colors: Optional custom colors
            output_path: Save path for the figure (PNG)

        Returns:
            Dict with rendering status and metadata
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(self.config.width / 100, self.config.height / 100), facecolor=self.config.background_color)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(self.config.background_color)

            ax.set_title(self.config.title, color="white", fontsize=14)
            ax.set_xlabel("Dim 1", color="#888")
            ax.set_ylabel("Dim 2", color="#888")
            ax.set_zlabel("Dim 3", color="#888")

            # Style axes
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.fill = False
                axis.pane.set_edgecolor("#333")
                axis.line.set_color("#555")
                axis._axinfo["grid"]["color"] = "#222"

            ax.tick_params(colors="#888")

            # Plot each direction
            default_colors = self.config.colors
            for i, direction in enumerate(directions):
                norm = np.linalg.norm(direction)
                if norm < 1e-8:
                    continue
                unit_dir = direction / norm

                color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
                label = labels[i] if labels and i < len(labels) else f"d{i}"

                # Arrow from origin
                ax.quiver(0, 0, 0,
                          unit_dir[0], unit_dir[1], unit_dir[2] if len(unit_dir) > 2 else 0,
                          color=color, arrow_length_ratio=0.15, linewidth=2, label=label)

                # Point at tip
                ax.scatter(unit_dir[0], unit_dir[1], unit_dir[2] if len(unit_dir) > 2 else 0,
                          color=color, s=self.config.point_size * 20, edgecolors="white", linewidth=0.5)

            # Unit sphere wireframe
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 15)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="#333333", alpha=0.15, linewidth=0.5)

            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_zlim([-1.2, 1.2])

            if labels:
                ax.legend(loc="upper right", fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

            self._figures.append(fig)

            if output_path:
                plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight", facecolor=self.config.background_color)
                plt.close(fig)

            return {
                "success": True,
                "n_directions": len(directions),
                "output_path": output_path,
            }

        except ImportError as e:
            return {"success": False, "error": f"Missing dependency: {e}", "message": "Install: pip install matplotlib"}

    def render_constraint_surface(
        self,
        solid_angle: float,
        rank: int = 3,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Render constraint surface as a cone based on solid angle.

        The solid angle determines the cone aperture — larger angles mean
        broader constraint regions.

        Args:
            solid_angle: Solid angle in steradians (0 to 4*pi)
            rank: Rank dimension for visual label
            output_path: Save path for the figure

        Returns:
            Dict with rendering status and cone parameters
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(9, 7), facecolor=self.config.background_color)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(self.config.background_color)

            # Calculate cone half-angle from solid angle: Omega = 2*pi*(1 - cos(theta))
            solid_angle_clamped = min(solid_angle, 4 * np.pi)
            half_angle = np.arccos(max(-1, 1 - solid_angle_clamped / (2 * np.pi)))
            cone_angle = 2 * half_angle

            # Create cone surface
            theta = np.linspace(0, 2 * np.pi, 60)
            phi = np.linspace(0, cone_angle, 30)
            theta, phi = np.meshgrid(theta, phi)

            r = 1.0
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            ax.plot_surface(x, y, z, alpha=0.5, color="#ff3366", edgecolor="none")

            # Add constraint direction axis
            ax.quiver(0, 0, 0, 0, 0, 1.3, color="white", arrow_length_ratio=0.1, linewidth=2, label="Constraint Axis")

            # Base ring
            ring_theta = np.linspace(0, 2 * np.pi, 100)
            ring_r = np.sin(cone_angle)
            ring_z = np.cos(cone_angle)
            ax.plot(ring_r * np.cos(ring_theta), ring_r * np.sin(ring_theta),
                   [ring_z] * 100, color="#ff3366", linewidth=2, alpha=0.8)

            ax.set_title(f"Constraint Surface\nSolid Angle: {solid_angle:.3f} sr | Cone Angle: {np.degrees(cone_angle):.1f} deg | Rank: {rank}",
                        color="white", fontsize=12)

            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.fill = False
                axis.pane.set_edgecolor("#222")
                axis.line.set_color("#333")
            ax.tick_params(colors="#666")
            ax.set_xlim([-1.3, 1.3]); ax.set_ylim([-1.3, 1.3]); ax.set_zlim([-1.3, 1.3])

            self._figures.append(fig)

            if output_path:
                plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight", facecolor=self.config.background_color)
                plt.close(fig)

            return {
                "success": True,
                "solid_angle_sr": solid_angle,
                "cone_half_angle_deg": round(np.degrees(half_angle), 2),
                "cone_angle_deg": round(np.degrees(cone_angle), 2),
                "output_path": output_path,
            }

        except ImportError:
            return {"success": False, "error": "matplotlib not installed"}

    def render_alignment_heatmap(
        self,
        alignment_matrix: Dict[int, Dict[int, float]],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Render cross-layer alignment as a heatmap.

        Args:
            alignment_matrix: Layer-to-layer cosine similarity dictionary
            output_path: Save path for the figure

        Returns:
            Dict with rendering status
        """
        try:
            import matplotlib.pyplot as plt

            layers = sorted(alignment_matrix.keys())
            n = len(layers)
            matrix = np.zeros((n, n))

            for i, l1 in enumerate(layers):
                for j, l2 in enumerate(layers):
                    matrix[i, j] = alignment_matrix.get(l1, {}).get(l2, 0.0)

            fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.config.background_color)
            ax.set_facecolor(self.config.background_color)

            im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

            # Add text annotations
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    color = "white" if abs(val) > 0.5 else "#aaa"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

            ax.set_xticks(range(n)); ax.set_xticklabels(layers, color="#888", fontsize=9)
            ax.set_yticks(range(n)); ax.set_yticklabels(layers, color="#888", fontsize=9)
            ax.set_xlabel("Layer", color="#888"); ax.set_ylabel("Layer", color="#888")
            ax.set_title("Cross-Layer Constraint Alignment", color="white", fontsize=14)

            cbar = plt.colorbar(im, ax=ax, shrink=0.85)
            cbar.set_label("Cosine Similarity", color="#888")
            cbar.ax.yaxis.set_tick_params(color="#888")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#888")

            self._figures.append(fig)

            if output_path:
                plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight", facecolor=self.config.background_color)
                plt.close(fig)

            return {"success": True, "n_layers": n, "output_path": output_path}

        except ImportError:
            return {"success": False, "error": "matplotlib not installed"}

    def render_layer_profile(
        self,
        layer_stats: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Render a bar chart of direction counts per layer.

        Args:
            layer_stats: List of {"layer": int, "directions": int, "explained_variance": float}
            output_path: Save path

        Returns:
            Rendering result
        """
        try:
            import matplotlib.pyplot as plt

            layers = [s["layer"] for s in layer_stats]
            counts = [s["directions"] for s in layer_stats]
            variances = [s.get("explained_variance", 0) for s in layer_stats]

            fig, ax1 = plt.subplots(figsize=(12, 5), facecolor=self.config.background_color)
            ax1.set_facecolor(self.config.background_color)

            bars = ax1.bar(layers, counts, color="#7c3aed", alpha=0.8, label="Directions")
            ax1.set_xlabel("Layer", color="#888"); ax1.set_ylabel("Number of Directions", color="#c084fc")
            ax1.tick_params(colors="#888")

            if any(v > 0 for v in variances):
                ax2 = ax1.twinx()
                ax2.plot(layers, variances, color="#ff3366", linewidth=2, marker="o", label="Explained Variance")
                ax2.set_ylabel("Explained Variance", color="#ff6699")
                ax2.tick_params(colors="#ff6699")

            ax1.set_title("Constraint Distribution Across Layers", color="white", fontsize=14)
            ax1.grid(axis="y", color="#222", alpha=0.5)
            ax1.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

            self._figures.append(fig)

            if output_path:
                plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight", facecolor=self.config.background_color)
                plt.close(fig)

            return {"success": True, "n_layers": len(layers), "output_path": output_path}

        except ImportError:
            return {"success": False, "error": "matplotlib not installed"}

    def export_webgl(
        self,
        directions: List[np.ndarray],
        labels: Optional[List[str]] = None,
        output_path: str = "holographic_viz.html",
    ) -> Dict[str, Any]:
        """
        Export an interactive 3D WebGL visualization using Three.js.

        Creates a self-contained HTML file that can be opened in any browser.
        Features: orbit controls, axis helpers, vector arrows, legend.

        Args:
            directions: List of direction vectors (first 3 components used)
            labels: Optional labels for each vector
            output_path: Path for the HTML output

        Returns:
            Dict with export status
        """
        dirs_json = json.dumps([d[:3].tolist() if len(d) >= 3 else d.tolist() + [0] * (3 - len(d)) for d in directions])
        lbls_json = json.dumps(labels if labels else [f"d{i}" for i in range(len(directions))])
        colors_json = json.dumps(self.config.colors)

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AETHERIS Constraint Geometry</title>
<style>
    * {{ margin: 0; padding: 0; }}
    body {{ background: #0a0a1a; overflow: hidden; font-family: monospace; }}
    #overlay {{
        position: absolute; top: 16px; left: 16px; color: #c084fc;
        background: rgba(10,10,30,0.85); padding: 16px 20px;
        border: 1px solid #303060; border-radius: 8px; z-index: 100;
        pointer-events: none; backdrop-filter: blur(4px);
    }}
    #overlay h2 {{ font-size: 18px; margin-bottom: 8px; }}
    #overlay p {{ color: #888; font-size: 13px; margin: 2px 0; }}
    #legend {{ position: absolute; bottom: 20px; left: 20px; color: #888; font-size: 12px; z-index: 100; }}
    .legend-item {{ display: inline-block; margin-right: 16px; }}
    .legend-swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="overlay">
    <h2>AETHERIS Constraint Geometry</h2>
    <p>Directions: {len(directions)}</p>
    <p>Drag to rotate | Scroll to zoom | Right-drag to pan</p>
</div>
<div id="legend"></div>

<script type="importmap">
{{"imports": {{"three": "https://unpkg.com/three@0.157.0/build/three.module.js",
              "three/addons/": "https://unpkg.com/three@0.157.0/examples/jsm/"}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);
scene.fog = new THREE.Fog(0x0a0a1a, 3, 15);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(2.5, 1.5, 2.5);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.4;

// Lighting
const ambient = new THREE.AmbientLight(0x303050);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 2);
dirLight.position.set(2, 3, 1);
scene.add(dirLight);

// Axes and grid
scene.add(new THREE.AxesHelper(1.3));
const grid = new THREE.GridHelper(2.4, 30, 0x222244, 0x111122);
scene.add(grid);

// Unit sphere wireframe
const sphereGeo = new THREE.SphereGeometry(1, 48, 24);
const sphereWire = new THREE.LineBasicMaterial({{ color: 0x222244, transparent: true, opacity: 0.15 }});
scene.add(new THREE.LineSegments(new THREE.WireframeGeometry(sphereGeo), sphereWire));

// Directions
const directions = {dirs_json};
const labels = {lbls_json};
const colors = {colors_json};
const legendDiv = document.getElementById('legend');

directions.forEach((dir, i) => {{
    const len = Math.sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    if (len < 0.001) return;
    const norm = [dir[0]/len, dir[1]/len, dir[2]/len];
    const color = colors[i % colors.length];

    const arrow = new THREE.ArrowHelper(
        new THREE.Vector3(norm[0], norm[1], norm[2]),
        new THREE.Vector3(0, 0, 0), 1.0, color, 0.2, 0.12
    );
    scene.add(arrow);

    const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.04, 16, 16),
        new THREE.MeshStandardMaterial({{ color, emissive: color, emissiveIntensity: 0.5 }})
    );
    sphere.position.set(norm[0], norm[1], norm[2]);
    scene.add(sphere);

    legendDiv.innerHTML += `<span class="legend-item"><span class="legend-swatch" style="background:${{'#' + new THREE.Color(color).getHexString()}}"></span>${{labels[i]}}</span>`;
}});

// Animation loop
function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>'''

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return {
            "success": True,
            "output_path": str(Path(output_path).resolve()),
            "n_directions": len(directions),
            "message": f"WebGL visualization saved to {output_path}. Open in browser.",
        }

    def close_all(self) -> None:
        """Close all open matplotlib figures."""
        try:
            import matplotlib.pyplot as plt
            for _ in self._figures:
                plt.close()
            self._figures.clear()
        except ImportError:
            pass
