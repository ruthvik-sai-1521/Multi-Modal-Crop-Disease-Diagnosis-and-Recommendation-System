let currentImageBlob = null;

function useFile() {
    document.getElementById('file-section').style.display = 'block';
    document.getElementById('camera-section').style.display = 'none';
    document.getElementById('btn-file').style.background = '#27ae60';
    document.getElementById('btn-cam').style.background = '#2ecc71';
}

function useCamera() {
    document.getElementById('file-section').style.display = 'none';
    document.getElementById('camera-section').style.display = 'block';
    document.getElementById('btn-cam').style.background = '#27ae60';
    document.getElementById('btn-file').style.background = '#2ecc71';
    startCamera();
}

async function startCamera() {
    const video = document.getElementById('video');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        alert("Camera access denied or not found!");
    }
}

function capture() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    // Stop stream
    video.srcObject.getTracks().forEach(track => track.stop());
    
    canvas.toBlob(blob => {
        currentImageBlob = blob;
        document.getElementById('cam-preview').src = URL.createObjectURL(blob);
        document.getElementById('cam-preview').style.display = 'block';
        video.style.display = 'none';
    }, 'image/jpeg');
}

function previewFile() {
    const file = document.getElementById('fileInput').files[0];
    currentImageBlob = file;
    document.getElementById('file-preview').src = URL.createObjectURL(file);
    document.getElementById('file-preview').style.display = 'block';
}

async function predictDisease() {
    if (!currentImageBlob) return alert("Please upload or capture an image first!");
    
    document.getElementById('loader').style.display = 'block';
    document.getElementById('result-area').style.display = 'none';
    
    let formData = new FormData();
    formData.append("file", currentImageBlob, "image.jpg");
    formData.append("temp", document.getElementById('temp').value);
    formData.append("humidity", document.getElementById('humidity').value);
    formData.append("ph", document.getElementById('ph').value);

    try {
        let response = await fetch('/predict', { method: "POST", body: formData });
        let result = await response.json();
        
        document.getElementById('loader').style.display = 'none';
        document.getElementById('result-area').style.display = 'block';
        
        document.getElementById('disease-name').innerText = result.disease;
        document.getElementById('confidence').innerText = "Confidence: " + result.confidence;
        
        // Reset AI boxes
        document.getElementById('ai-explain-box').innerHTML = "Click 'Explain Disease' button...";
        document.getElementById('ai-remedy-box').innerHTML = "Click 'Natural Remedies' button...";
    } catch (err) {
        alert("Error connecting to server!");
        document.getElementById('loader').style.display = 'none';
    }
}

async function getAI(mode) {
    let disease = document.getElementById('disease-name').innerText;
    // Get environment values to send to AI
    let temp = document.getElementById('temp').value;
    let humidity = document.getElementById('humidity').value;

    let targetBox = mode === 'explain' ? 'ai-explain-box' : 'ai-remedy-box';
    
    document.getElementById(targetBox).innerHTML = "⏳ <b>AI is analyzing... please wait...</b>";
    
    try {
        let response = await fetch('/explain_remedy', {
            method: "POST",
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                disease: disease, 
                mode: mode,
                temp: temp,         // Added Context
                humidity: humidity  // Added Context
            })
        });
        
        let result = await response.json();
        // Convert newlines to HTML line breaks for better readability
        document.getElementById(targetBox).innerHTML = result.response.replace(/\n/g, "<br>");
    } catch (err) {
        document.getElementById(targetBox).innerHTML = "⚠️ Error fetching AI response.";
    }
}