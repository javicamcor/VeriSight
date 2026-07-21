/**
 * VeriSight - Sandbox Script
 * Se ejecuta en un origen aislado (Sandboxed Iframe) con políticas de CSP relajadas.
 * Aquí podemos ejecutar OpenCV y WebAssembly (Emscripten eval) sin bloqueos de Chrome.
 */

window.addEventListener('message', async (event) => {
    // Escuchar mensajes de content.js
    if (event.data && event.data.action === "ANALYZE_IMAGE") {
        try {
            await waitForOpenCV();
            
            const resultImageData = await analizarImagen(event.data.imageData);
            
            // Devolver el resultado (espectro en formato ImageData) a content.js
            event.source.postMessage({
                action: "ANALYSIS_COMPLETE",
                resultImageData: resultImageData
            }, event.origin);
            
        } catch (error) {
            console.error("[Sandbox Error]", error);
            event.source.postMessage({
                action: "ANALYSIS_ERROR",
                error: error.message
            }, event.origin);
        }
    }
});

function waitForOpenCV() {
    return new Promise((resolve, reject) => {
        let attempts = 0;
        const check = setInterval(() => {
            // Comprobamos si el objeto cv y la función matFromImageData existen
            if (typeof cv !== 'undefined' && cv.Mat && cv.matFromImageData) {
                clearInterval(check);
                resolve();
            } else {
                attempts++;
                if (attempts > 100) {  // 10 segundos
                    clearInterval(check);
                    reject(new Error("Timeout inicializando OpenCV.js en el Sandbox"));
                }
            }
        }, 100);
    });
}

async function analizarImagen(imageData) {
    let src, padded, complexI, planes, mag, zeros, ones;

    try {
        // a) Cargar la imagen desde el objeto ImageData proporcionado por content.js
        src = cv.matFromImageData(imageData);

        // b) Convertir a escala de grises
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);

        // c) Aplicar la transformada de Fourier (DFT)
        const m = cv.getOptimalDFTSize(src.rows);
        const n = cv.getOptimalDFTSize(src.cols);
        padded = new cv.Mat();
        cv.copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));

        planes = new cv.MatVector();
        padded.convertTo(padded, cv.CV_32F);
        planes.push_back(padded); 
        
        zeros = cv.Mat.zeros(padded.size(), cv.CV_32F);
        planes.push_back(zeros);  

        complexI = new cv.Mat();
        cv.merge(planes, complexI);

        cv.dft(complexI, complexI, cv.DFT_COMPLEX_OUTPUT);

        // d) Aplicar fftshift
        fftShift(complexI);

        // e) Filtro de paso alto
        const radius = 30; // Ajustable según resolución y sensibilidad
        applyHighPassFilter(complexI, radius);

        // f) Calcular magnitud y pasarla a escala logarítmica
        cv.split(complexI, planes);
        mag = new cv.Mat();
        cv.magnitude(planes.get(0), planes.get(1), mag);

        ones = cv.Mat.ones(mag.size(), cv.CV_32F);
        cv.add(mag, ones, mag);
        cv.log(mag, mag);

        // Normalizar a rango 0-255 para visualización en Canvas
        cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX);
        mag.convertTo(mag, cv.CV_8U); 

        // Convertir la matriz final cv.Mat (1 canal, grises) a ImageData RGBA para pasarlo de vuelta por postMessage
        cv.cvtColor(mag, mag, cv.COLOR_GRAY2RGBA);
        
        const resultImgData = new ImageData(
            new Uint8ClampedArray(mag.data),
            mag.cols,
            mag.rows
        );

        return resultImgData;

    } finally {
        // Liberar toda la memoria WebAssembly
        if (src && !src.isDeleted()) src.delete();
        if (padded && !padded.isDeleted()) padded.delete();
        if (complexI && !complexI.isDeleted()) complexI.delete();
        if (planes && !planes.isDeleted()) planes.delete();
        if (mag && !mag.isDeleted()) mag.delete();
        if (zeros && !zeros.isDeleted()) zeros.delete();
        if (ones && !ones.isDeleted()) ones.delete();
    }
}

function fftShift(mat) {
    const cx = Math.floor(mat.cols / 2);
    const cy = Math.floor(mat.rows / 2);

    let q0 = mat.roi(new cv.Rect(0, 0, cx, cy));   
    let q1 = mat.roi(new cv.Rect(cx, 0, cx, cy));  
    let q2 = mat.roi(new cv.Rect(0, cy, cx, cy));  
    let q3 = mat.roi(new cv.Rect(cx, cy, cx, cy)); 

    let tmp = new cv.Mat();
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    q0.delete(); q1.delete(); q2.delete(); q3.delete(); tmp.delete();
}

function applyHighPassFilter(complexMat, radius) {
    const cx = Math.floor(complexMat.cols / 2);
    const cy = Math.floor(complexMat.rows / 2);
    const cols = complexMat.cols;
    const rows = complexMat.rows;
    const view = complexMat.data32F; 

    for (let i = 0; i < rows; i++) {
        const dy = i - cy;
        const dy2 = dy * dy;
        for (let j = 0; j < cols; j++) {
            const dx = j - cx;
            if (Math.sqrt(dx * dx + dy2) <= radius) {
                const idx = (i * cols + j) * 2;
                view[idx] = 0.0;     
                view[idx + 1] = 0.0; 
            }
        }
    }
}
