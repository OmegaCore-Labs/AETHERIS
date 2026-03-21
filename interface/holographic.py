"""
Holographic Visualizer — 3D Constraint Geometry Visualization

Creates 3D visualizations of constraint geometry, barrier surfaces,
and cross-layer alignment.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for 3D visualization."""
    title: str = "Constraint Geometry"
    width: int = 800
    height: int = 600
    background_color: str = "black"
    point_size: float = 5.0
    line_width: float = 1.0
    show_axes: bool = True
    show_grid: bool = True


class HolographicViz:
    """
    Generate 3D visualizations of constraint geometry.

    Features:
    - 3D direction vectors
    - Solid angle cones
    - Cross-layer alignment heatmaps
    - Barrier surfaces
    - WebGL export for browser viewing
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self._figures = []

    def render_geometry(
        self,
        directions: List[np.ndarray],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Render 3D geometry of constraint directions.

        Args:
            directions: List of direction vectors (3D)
            labels: Optional labels for each direction
            colors: Optional colors for each direction
            output_path: Path to save visualization

        Returns:
            Visualization result
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(self.config.width / 100, self.config.height / 100))
            ax = fig.add_subplot(111, projection='3d')

            ax.set_title(self.config.title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            if self.config.show_grid:
                ax.grid(True)
            if self.config.show_axes:
                ax.axhline(0, color='gray', linewidth=0.5)
                ax.axvline(0, color='gray', linewidth=0.5)

            # Default colors
            default_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

            for i, direction in enumerate(directions):
                # Normalize to unit sphere
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm

                color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
                label = labels[i] if labels and i < len(labels) else f"v{i}"

                # Plot vector from origin
                ax.quiver(0, 0, 0, direction[0], direction[1], direction[2],
                          color=color, arrow_length_ratio=0.2, label=label)

                # Plot point at tip
                ax.scatter(direction[0], direction[1], direction[2],
                          color=color, s=self.config.point_size * 10)

            # Add legend if labels
            if labels:
                ax.legend()

            self._figures.append(fig)

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            return {
                "success": True,
                "n_directions": len(directions),
                "output_path": output_path,
                "message": f"Rendered {len(directions)} directions"
            }

        except ImportError:
            return {
                "success": False,
                "error": "matplotlib not installed",
                "message": "Install with: pip install matplotlib"
            }

    def render_constraint_surface(
        self,
        solid_angle: float,
        rank: int,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Render constraint surface as a cone.

        Args:
            solid_angle: Solid angle in steradians
            rank: Rank of constraint (1-3 for visualization)
            output_path: Path to save visualization

        Returns:
            Visualization result
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(self.config.width / 100, self.config.height / 100))
            ax = fig.add_subplot(111, projection='3d')

            # Calculate cone half-angle from solid angle
            # Solid angle = 2π(1 - cos(θ/2)) for a cone
            half_angle = np.arccos(1 - solid_angle / (2 * np.pi))
            cone_angle = half_angle * 2

            # Create cone surface
            theta = np.linspace(0, 2 * np.pi, 50)
            phi = np.linspace(0, cone_angle, 25)
            theta, phi = np.meshgrid(theta, phi)

            r = 1.0
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            ax.plot_surface(x, y, z, alpha=0.6, color='red', edgecolor='none')

            # Add axes
            ax.quiver(0, 0, 0, 1.2, 0, 0, color='gray', arrow_length_ratio=0.1)
            ax.quiver(0, 0, 0, 0, 1.2, 0, color='gray', arrow_length_ratio=0.1)
            ax.quiver(0, 0, 0, 0, 0, 1.2, color='gray', arrow_length_ratio=0.1)

            ax.set_title(f"Constraint Surface\nSolid Angle: {solid_angle:.2f} sr, Rank: {rank}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            self._figures.append(fig)

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            return {
                "success": True,
                "solid_angle": solid_angle,
                "cone_angle_deg": np.degrees(cone_angle),
                "output_path": output_path
            }

        except ImportError:
            return {
                "success": False,
                "error": "matplotlib not installed"
            }

    def render_cross_layer_heatmap(
        self,
        alignment_matrix: Dict[int, Dict[int, float]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Render cross-layer alignment as a heatmap.

        Args:
            alignment_matrix: Layer-to-layer alignment dictionary
            output_path: Path to save visualization

        Returns:
            Visualization result
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Convert to matrix
            layers = sorted(alignment_matrix.keys())
            n = len(layers)
            matrix = np.zeros((n, n))

            for i, l1 in enumerate(layers):
                for j, l2 in enumerate(layers):
                    matrix[i, j] = alignment_matrix.get(l1, {}).get(l2, 0)

            fig, ax = plt.subplots(figsize=(10, 8))

            sns.heatmap(matrix,
                        xticklabels=layers,
                        yticklabels=layers,
                        annot=True,
                        fmt='.2f',
                        cmap='RdBu_r',
                        center=0,
                        vmin=-1,
                        vmax=1,
                        ax=ax)

            ax.set_title("Cross-Layer Alignment Heatmap")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Layer")

            self._figures.append(fig)

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            return {
                "success": True,
                "n_layers": n,
                "output_path": output_path
            }

        except ImportError:
            return {
                "success": False,
                "error": "matplotlib/seaborn not installed"
            }

    def export_webgl(
        self,
        directions: List[np.ndarray],
        output_path: str = "geometry.html"
    ) -> Dict[str, Any]:
        """
        Export 3D visualization to WebGL HTML.

        Args:
            directions: List of direction vectors
            output_path: Path for HTML file

        Returns:
            Export result
        """
        # Generate Three.js HTML
        html_content = self._generate_threejs_html(directions)

        with open(output_path, 'w') as f:
            f.write(html_content)

        return {
            "success": True,
            "output_path": output_path,
            "message": "WebGL visualization exported. Open in browser."
        }

    def _generate_threejs_html(self, directions: List[np.ndarray]) -> str:
        """Generate Three.js HTML for 3D visualization."""
        # Convert directions to JSON
        dirs_json = [d.tolist() for d in directions]

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>AETHERIS Constraint Geometry</title>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        #info {{
            position: absolute;
            top: 20px;
            left: 20px;
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            font-family: monospace;
            z-index: 100;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h2>AETHERIS Constraint Geometry</h2>
        <p>{len(directions)} constraint directions</p>
    </div>

    <script type="importmap">
        {{
            "imports": {{
                "three": "https://unpkg.com/three@0.128.0/build/three.module.js"
            }}
        }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'https://unpkg.com/three@0.128.0/examples/jsm/controls/OrbitControls.js';

        // Setup scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050510);

        // Camera
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(2, 2, 2);
        camera.lookAt(0, 0, 0);

        // Renderer
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Axes helper
        const axesHelper = new THREE.AxesHelper(1.2);
        scene.add(axesHelper);

        // Grid helper
        const gridHelper = new THREE.GridHelper(3, 20);
        scene.add(gridHelper);

        // Directions data
        const directions = {dirs_json};

        // Colors
        const colors = [0xff3333, 0x33ff33, 0x3333ff, 0xff33ff, 0x33ffff, 0xff9933];

        // Add arrows for each direction
        directions.forEach((dir, i) => {{
            // Normalize
            const len = Math.sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
            const norm = [dir[0]/len, dir[1]/len, dir[2]/len];

            // Arrow helper
            const origin = new THREE.Vector3(0, 0, 0);
            const direction = new THREE.Vector3(norm[0], norm[1], norm[2]);
            const arrowColor = colors[i % colors.length];

            const arrowHelper = new THREE.ArrowHelper(direction, origin, 1.0, arrowColor, 0.3, 0.2);
            scene.add(arrowHelper);

            // Add sphere at tip
            const sphereGeometry = new THREE.SphereGeometry(0.05, 16, 16);
            const sphereMaterial = new THREE.MeshStandardMaterial({{ color: arrowColor }});
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            sphere.position.copy(direction.clone().multiplyScalar(1.0));
            scene.add(sphere);
        }});

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);

        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(1, 2, 1);
        scene.add(directionalLight);

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>
"""
