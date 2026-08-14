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

        // Cargar el modelo exportado de PyTorch Two-Stream
        const modelUrl = 'onnx_model/model_twostream.onnx?v=' + new Date().getTime();
        session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
        console.log("[VeriSight Sandbox] Modelo Two-Stream ONNX cargado y listo.");

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
                message: "Arrancando motor neuronal..."
            }, event.origin);
            return;
        }

        processImage(event.data, event.source, event.origin);
    }
});

async function processImage(data, source, origin) {
    try {
        let src = cv.matFromImageData(data.imageData);

        // Ya recibimos el parche de 224x224 directamente desde content.js
        const prediction = await runInference(src);

        src.delete();

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


//Pípline completo Two-Stream (Espacial + Frecuencias)
async function runInference(faceMat) {
    let resized = new cv.Mat();
    // Salvaguarda: Asegurarnos de que mida 224x224 
    cv.resize(faceMat, resized, new cv.Size(224, 224), 0, 0, cv.INTER_AREA);

    // FASE 1: TENSOR ESPACIAL (Mantenemos RGB Real)
    let rgbMat = new cv.Mat();
    cv.cvtColor(resized, rgbMat, cv.COLOR_RGBA2RGB);

    const spatialElements = 3 * 224 * 224;
    const spatialFloat32 = new Float32Array(spatialElements);

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    let pixelCount = 224 * 224;
    for (let y = 0; y < 224; y++) {
        for (let x = 0; x < 224; x++) {
            let r = rgbMat.ucharPtr(y, x)[0] / 255.0;
            let g = rgbMat.ucharPtr(y, x)[1] / 255.0;
            let b = rgbMat.ucharPtr(y, x)[2] / 255.0;

            r = (r - mean[0]) / std[0];
            g = (g - mean[1]) / std[1];
            b = (b - mean[2]) / std[2];

            let index = y * 224 + x;
            spatialFloat32[index] = r;                 // Canal R
            spatialFloat32[pixelCount + index] = g;    // Canal G
            spatialFloat32[pixelCount * 2 + index] = b;// Canal B
        }
    }

    const tensorSpatial = new ort.Tensor('float32', spatialFloat32, [1, 3, 224, 224]);


    // FASE 2: TENSOR FRECUENCIAL (Transformada FFT)
    let grayMat = new cv.Mat();
    cv.cvtColor(resized, grayMat, cv.COLOR_RGBA2GRAY);

    // Aplicar Ventana de Hann (224x224)
    const N = 224;
    const hann = new Float32Array(N);
    for (let i = 0; i < N; i++) {
        hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1));
    }

    let resizedFloat = new cv.Mat();
    grayMat.convertTo(resizedFloat, cv.CV_32F);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            let val = resizedFloat.floatPtr(y, x)[0];
            resizedFloat.floatPtr(y, x)[0] = val * hann[y] * hann[x];
        }
    }

    // Transformada Discreta de Fourier
    let planes = new cv.MatVector();
    planes.push_back(resizedFloat);
    planes.push_back(cv.Mat.zeros(resizedFloat.size(), cv.CV_32F));

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

    // Filtro Paso Alto (Radio 10 proporcional a 224)
    let radius = 10;
    for (let y = 0; y < complexI.rows; y++) {
        for (let x = 0; x < complexI.cols; x++) {
            if (Math.pow(x - cx, 2) + Math.pow(y - cy, 2) <= radius * radius) {
                complexI.floatPtr(y, x)[0] = 1;
                complexI.floatPtr(y, x)[1] = 0;
            }
        }
    }

    // Extraer Magnitud
    cv.split(complexI, planes);
    let mag = planes.get(0);
    cv.magnitude(planes.get(0), planes.get(1), mag);

    // Extracción Matemática Exacta y Normalización Z-Score
    const freqElements = 224 * 224;
    const freqFloat32 = new Float32Array(freqElements);

    let sum = 0;
    // Magnitud Logarítmica: 20 * log(|f| + 1)
    for (let i = 0; i < freqElements; i++) {
        let val = mag.data32F[i];
        let transformed = 20 * Math.log(Math.abs(val) + 1);
        freqFloat32[i] = transformed;
        sum += transformed;
    }

    let freqMean = sum / freqElements;
    let freqVarianceSum = 0;
    for (let i = 0; i < freqElements; i++) {
        freqVarianceSum += Math.pow(freqFloat32[i] - freqMean, 2);
    }
    let freqStd = Math.sqrt(freqVarianceSum / freqElements);

    if (freqStd > 0) {
        for (let i = 0; i < freqElements; i++) {
            freqFloat32[i] = (freqFloat32[i] - freqMean) / freqStd;
        }
    }

    const tensorFreq = new ort.Tensor('float32', freqFloat32, [1, 1, 224, 224]);

    // FASE 3: INFERENCIA TWO-STREAM ONNX
    const results = await session.run({
        'input_spatial': tensorSpatial,
        'input_frequency': tensorFreq
    });

    // La red devuelve un Logit. Pasamos por Sigmoide.
    const logit = results.output.data[0];
    const outputProbability = 1 / (1 + Math.exp(-logit));

    // Limpieza de memoria
    resized.delete(); rgbMat.delete(); grayMat.delete();
    resizedFloat.delete(); planes.delete(); complexI.delete();
    q0.delete(); q1.delete(); q2.delete(); q3.delete(); tmp.delete(); mag.delete();

    return outputProbability;
}
