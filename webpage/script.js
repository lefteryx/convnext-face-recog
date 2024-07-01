const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const statusElement = document.getElementById('status');
let model, faceCascade;

// Load the model
async function loadModel() {
    model = await tf.loadGraphModel('model/model.json');
    statusElement.innerText = 'Status: Model Loaded';
}

async function loadCascade() {
    try {
        const response = await fetch('/haar-cascade/haarcascade_frontalface_default.xml');
        if (!response.ok) {
            throw new Error(`Failed to load Haar Cascade XML file: ${response.statusText}`);
        }
        const xmlText = await response.text();
        const parser = new DOMParser();
        const xml = parser.parseFromString(xmlText, 'application/xml');
        console.log('Type is: ' + typeof xml);
        if (xml.getElementsByTagName('parsererror').length > 0) {
            throw new Error('Error parsing Haar Cascade XML file');
        }
        
        faceCascade = new cv.CascadeClassifier();
        let cascadeFile = 'haar-cascade/haarcascade_frontalface_default.xml'; // Adjust the path as necessary

        fetch(cascadeFile).then(response => {
            response.arrayBuffer().then(buffer => {
                let data = new Uint8Array(buffer);
                cv.FS_createDataFile('/', 'haarcascade_frontalface_default.xml', data, true, false, false);
                faceCascade.load('haarcascade_frontalface_default.xml');
                console.log('Cascade loaded');
            });
        }).catch(err => {
            console.error('Failed to load cascade file:', err);
        });

        statusElement.innerText = 'Status: Cascade Loaded';
    } catch (error) {
        console.error('Error loading cascade:', error);
        statusElement.innerText = 'Status: Error loading cascade';
    }
}



// Load Employee Embeddings
async function loadEmbeddings() {
    const response = await fetch('employee_embeddings/embeddings.json');
    return await response.json();
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

async function init() {
    await loadModel();
    await loadCascade();
    const embeddings = await loadEmbeddings();
    await setupCamera();

    statusElement.innerText = 'Status: Ready';

    video.addEventListener('play', () => {
        const frameRate = 100;
        let frameCount = 0;

        setInterval(() => {
            if (video.paused || video.ended) return;

            frameCount++;
            if (frameCount % frameRate !== 0) return;

            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const image = cv.imread(canvas);
            const gray = new cv.Mat();
            cv.cvtColor(image, gray, cv.COLOR_RGBA2GRAY);
            const faces = new cv.RectVector();
            const msize = new cv.Size(0, 0);
            faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);
            
            console.log('Faces: ' + faces.size());
            if (faces.size() === 0) {
                statusElement.innerText = 'No face detected';
            } else if (faces.size() > 1) {
                statusElement.innerText = 'More than one face detected';
            } else {
                console.log("Faces = 1 ke andar");
                const face = faces.get(0);
                console.log("const face ke baad");
                const faceImage = image.roi(face);
                console.log("const faceImage ke baad");
                const tensor = tf.browser.fromPixels(faceImage).resizeBilinear([224, 224]).expandDims(0).toFloat().div(255);
                console.log("Tensor ke baad, embedding ke pehle");
                const embedding = model.predict(tensor).dataSync();
                let maxSimilarity = 0;
                let verified = false;

                statusElement.innerText = 'Checking authorization...';

                // console log

                for (const employee of embeddings) {
                    for (const empEmbedding of employee.embeddings) {
                        const similarity = cosineSimilarity(embedding, empEmbedding);
                        if (similarity > maxSimilarity) maxSimilarity = similarity;
                        if (similarity > 0.4) {
                            statusElement.innerText = `Employee Verified: ${employee.name}`;
                            verified = true;
                            break;
                        }
                    }
                    if (verified) break;
                }

                if (!verified) {
                    statusElement.innerText = 'Not authorized';
                }
            }

            image.delete();
            gray.delete();
            faces.delete();
        }, 100);
    });
}

function cosineSimilarity(a, b) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;

    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

document.addEventListener('DOMContentLoaded', init);
