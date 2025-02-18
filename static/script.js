// Get the canvas element and context
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');

// Initialize canvas with white background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Set up drawing configurations
ctx.lineWidth = 20;  // Thicker lines (improves visibility)
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

// Variables for drawing
let drawing = false;

// Handle mouse down (start drawing)
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

// Handle mouse move (drawing)
canvas.addEventListener('mousemove', (e) => {
    if (drawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

// Handle mouse up (stop drawing)
canvas.addEventListener('mouseup', () => {
    drawing = false;
});

// Handle mouse out (stop drawing)
canvas.addEventListener('mouseout', () => {
    drawing = false;
});

// Clear the canvas (fill with white)
function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();  // Reset drawing path
}

// Convert canvas to image and send to backend
function submitCanvas() {
    const imageData = canvas.toDataURL('image/png');

    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify({ image: imageData }),
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then((response) => response.json())
    .then((data) => {
        document.getElementById('predictionResult').innerText = `Prediction: ${data.digit}`;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}