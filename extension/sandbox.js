let cvReady = false;
let session = null;

// Escuchar cuando OpenCV.js esté cargado
if (typeof cv !== 'undefined' && cv.onRuntimeInitialized) {
    cv.onRuntimeInitialized = initializeAI;
} else if (typeof cv !== 'undefined' && typeof cv.Mat !== 'undefined') {
    initializeAI();
} else {
    window.onload = () => setTimeout(initializeAI, 1000);
}

let pendingRequest = null;

async function initializeAI() {
    if (cvReady && session) return;
    cvReady = true;
    console.log("[VeriSight Sandbox] OpenCV inicializado.");
    
    try {
        // Configurar ONNX Runtime para que use el motor WASM del navegador
        ort.env.wasm.numThreads = 1;
        
        // Cargar el modelo exportado de PyTorch (Evitamos el caché de Chrome añadiendo un timestamp)
        const modelUrl = 'onnx_model/model.onnx?v=' + new Date().getTime();
        session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
        console.log("[VeriSight Sandbox] Modelo neuronal ONNX cargado y listo.");
        
        // Si había una foto esperando, la procesamos ahora
        if (pendingRequest) {
            processImage(pendingRequest.data, pendingRequest.source, pendingRequest.origin);
            pendingRequest = null;
        }
    } catch (error) {
        console.error("[VeriSight Sandbox] Error crítico cargando modelo ONNX:", error);
    }
}

// Escuchar peticiones del content.js
window.addEventListener("message", async (event) => {
    if (event.data && event.data.action === "ANALYZE_IMAGE") {
        if (!cvReady || !session) {
            // Guardamos la petición en la sala de espera
            pendingRequest = { data: event.data, source: event.source, origin: event.origin };
            // Avisamos al usuario de que está cargando
            event.source.postMessage({ 
                action: "ANALYSIS_LOADING", 
                message: "Arrancando motor neuronal Edge AI..." 
            }, event.origin);
            return;
        }
        
        processImage(event.data, event.source, event.origin);
    }
});

async function processImage(data, source, origin) {
    try {
        const prediction = await runInference(data.imageData);
        
        // Enviar la probabilidad de Deepfake de vuelta al usuario
        source.postMessage({ 
            action: "ANALYSIS_COMPLETE", 
            prediction: prediction 
        }, origin);
        
    } catch (error) {
        console.error(error);
        source.postMessage({ 
            action: "ANALYSIS_ERROR", 
            error: "Error interno de matemáticas: " + error.toString() 
        }, origin);
    }
}

/**
 * 1. Extrae espectro con OpenCV
 * 2. Normaliza para PyTorch
 * 3. Ejecuta CNN a través de ONNX
 */
async function runInference(imageData) {
    // ---- FASE 1: OpenCV (Transformada de Fourier) ----
    let src = cv.matFromImageData(imageData);
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY);
    
    // Redimensionar a 128x128 como en el dataset
    let resized = new cv.Mat();
    cv.resize(src, resized, new cv.Size(128, 128));
    
    let m = cv.getOptimalDFTSize(resized.rows);
    let n = cv.getOptimalDFTSize(resized.cols);
    let padded = new cv.Mat();
    cv.copyMakeBorder(resized, padded, 0, m - resized.rows, 0, n - resized.cols, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    
    let planes = new cv.MatVector();
    let paddedF32 = new cv.Mat();
    padded.convertTo(paddedF32, cv.CV_32F);
    planes.push_back(paddedF32);
    planes.push_back(cv.Mat.zeros(padded.size(), cv.CV_32F));
    
    let complexI = new cv.Mat();
    cv.merge(planes, complexI);
    cv.dft(complexI, complexI, cv.DFT_COMPLEX_OUTPUT);
    
    // FFTSHIFT manual
    let cx = complexI.cols / 2;
    let cy = complexI.rows / 2;
    let q0 = complexI.roi(new cv.Rect(0, 0, cx, cy));
    let q1 = complexI.roi(new cv.Rect(cx, 0, cx, cy));
    let q2 = complexI.roi(new cv.Rect(0, cy, cx, cy));
    let q3 = complexI.roi(new cv.Rect(cx, cy, cx, cy));
    
    let tmp = new cv.Mat();
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    // Filtro Paso Alto (Radio 30)
    let radius = 30;
    for (let y = 0; y < complexI.rows; y++) {
        for (let x = 0; x < complexI.cols; x++) {
            if (Math.pow(x - cx, 2) + Math.pow(y - cy, 2) <= radius * radius) {
                complexI.floatPtr(y, x)[0] = 0;
                complexI.floatPtr(y, x)[1] = 0;
            }
        }
    }
    
    // Extraer Magnitud
    cv.split(complexI, planes);
    let mag = new cv.Mat();
    cv.magnitude(planes.get(0), planes.get(1), mag);
    
    // Escala Logarítmica
    let ones = cv.Mat.ones(mag.size(), cv.CV_32F);
    cv.add(mag, ones, mag);
    cv.log(mag, mag);
    
    // Normalizar a 0-255
    cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX);
    
    // Extraer a formato Uint8
    let mag8U = new cv.Mat();
    mag.convertTo(mag8U, cv.CV_8U);
    let uint8Array = mag8U.data;
    
    // ---- FASE 2: Preparación de Tensores para ONNX ----
    // PyTorch espera un Tensor FLOAT32 (1, 1, 128, 128) con valores [0, 1]
    const numElements = 128 * 128;
    const float32Data = new Float32Array(numElements);
    
    for (let i = 0; i < numElements; i++) {
        float32Data[i] = uint8Array[i] / 255.0; // Normalización al vuelo
    }
    
    const tensor = new ort.Tensor('float32', float32Data, [1, 1, 128, 128]);
    
    // ---- FASE 3: Inferencia ONNX ----
    const results = await session.run({ 'input': tensor });
    const outputProbability = results.output.data[0];
    
    // Liberar memoria C++ (Vital para que Chrome no crashee por fugas de memoria)
    src.delete(); resized.delete(); padded.delete(); paddedF32.delete();
    planes.delete(); complexI.delete(); q0.delete(); q1.delete();
    q2.delete(); q3.delete(); tmp.delete(); mag.delete(); ones.delete();
    mag8U.delete();
    
    return outputProbability;
}
