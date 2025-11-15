// importing three.js core library using import map
import * as THREE from "three";

// importing orbit controls from examples (resolved by import map)
import {OrbitControls} from 'jsm/controls/OrbitControls.js';

// getting full window size for renderer
const w = window.innerWidth;
const h = window.innerHeight;

// creating WebGL renderer with anti-aliasing for smooth edges
const renderer = new THREE.WebGLRenderer({ antialias: true });

// setting renderer resolution to window size
renderer.setSize(w, h);

// adding renderer canvas to HTML body
document.body.appendChild(renderer.domElement);

// creating perspective camera
const camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 10);
//parameters = feild of view, aspect ratio, near plane, far plane

// positioning camera on Z axis
camera.position.z = 2;
camera.lookAt(0, 0, 0); // Ensure camera looks at the scene origin

// creating a new scene to hold objects
const scene = new THREE.Scene();

// enabling orbit controls (mouse to rotate, zoom, pan)
const control = new OrbitControls(camera, renderer.domElement);
control.enableDamping = true;   // smooth movement
control.dampingFactor = 0.05;   // damping amount

// creating an icosahedron geometry
const geo = new THREE.IcosahedronGeometry(1.0, 2);

// creating material with flat shading and solid color
const mat = new THREE.MeshStandardMaterial({ 
	color: 0xffffff,   
	flatShading: true,
	//wireframe: false
});

// creating mesh (geometry + material)
const mesh = new THREE.Mesh(geo, mat);

// adding mesh to the scene
scene.add(mesh);  

// creating wireframe material
const wireMat = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true });

// creating wireframe mesh to overlay original mesh
const wireMesh = new THREE.Mesh(geo, wireMat);

// slightly scaling wireframe so it sits outside main mesh
wireMesh.scale.setScalar(1.001);

//scene.add(wireMesh);
mesh.add(wireMesh);  // attaching wireframe to main mesh

// adding hemisphere light (sky color + ground color)
const hemisphereLight = new THREE.HemisphereLight(0x0099ff, 0xaa5500, 1.0);
hemisphereLight.position.set(0, 90, 0);  // placing light above
scene.add(hemisphereLight);


// Use a simple animation loop and the correct renderer object
renderer.setPixelRatio(window.devicePixelRatio);

// animation loop
function animate(t = 0) {
	//console.log(t);  //for checking time value in console

	requestAnimationFrame(animate); // request next animation frame

	mesh.rotation.x += -0.01;
	mesh.rotation.y += 0.01;
	//uncomment either (line 79 or/and 80) above two or only below statement (line 82)
    //mesh.scale.setScalar(Math.cos(t * 0.001)+1.0);

	renderer.render(scene, camera); // draw the scene
	control.update();               // update orbit controls (commented out - control not initialized)
}

// Handle window resize
window.addEventListener('resize', () => {
	const w = window.innerWidth;
	const h = window.innerHeight;
	renderer.setSize(w, h);
	camera.aspect = w / h;
	camera.updateProjectionMatrix();
});

animate(); // start animation
