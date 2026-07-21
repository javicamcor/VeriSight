/**
 * VeriSight - Content Script
 * Se inyecta en la página para actuar de puente entre la imagen web y el Sandboxed Iframe.
 */

let sandboxWindow = null;

// Escuchar mensajes del background.js (clic derecho)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "ANALYZE_IMAGE") {
        iniciarAnalisis(request.srcUrl);
    }
});

/**
 * Función principal que orquesta la captura de imagen y delegación al Sandbox
 */
async function iniciarAnalisis(url) {
    try {
        showOverlay("Iniciando análisis...");

        // 1. Aseguramos que el Sandbox (iframe oculto) está inyectado
        await setupSandbox();

        // 2. Descargar la imagen de la URL resolviendo CORS localmente y convertirla a ImageData pura
        showOverlay("Descargando píxeles de la imagen...");
        const imageData = await getImageData(url);

        showOverlay("Calculando Espectro de Fourier en Sandbox...");

        // 3. Enviar los píxeles al iframe Sandboxed mediante postMessage
        sandboxWindow.postMessage({
            action: "ANALYZE_IMAGE",
            imageData: imageData
        }, "*");

    } catch (error) {
        console.error("[VeriSight] Error:", error);
        showOverlay("Error: " + error.message, true);
    }
}

/**
 * Inyecta un iframe oculto que carga el sandbox.html (entorno seguro con permisos de eval)
 */
function setupSandbox() {
    return new Promise((resolve) => {
        let iframe = document.getElementById('verisight-sandbox');
        if (iframe) {
            sandboxWindow = iframe.contentWindow;
            resolve();
            return;
        }

        iframe = document.createElement('iframe');
        iframe.id = 'verisight-sandbox';
        iframe.style.display = 'none'; // Invisible para el usuario
        // Cargamos la página aislada declarada en el manifest.json
        iframe.src = chrome.runtime.getURL('sandbox.html');

        iframe.onload = () => {
            sandboxWindow = iframe.contentWindow;
            resolve();
        };

        document.body.appendChild(iframe);
    });
}

/**
 * Escuchar las respuestas (espectro o error) que llegan de vuelta desde el Sandbox
 */
window.addEventListener("message", (event) => {
    if (event.data && event.data.action === "ANALYSIS_COMPLETE") {
        const prob = event.data.prediction * 100;
        const isFake = prob >= 50;
        const icon = isFake ? "Alerta IA" : "Real";
        const extraText = isFake ? "Posible Deepfake" : "Imagen auténtica";

        const resultMsg = `${icon}\n${prob.toFixed(2)}% de probabilidad de IA.\n${extraText}`;
        showOverlay(resultMsg, isFake);

    } else if (event.data && event.data.action === "ANALYSIS_LOADING") {
        showOverlay("⏳ " + event.data.message, false);
    } else if (event.data && event.data.action === "ANALYSIS_ERROR") {
        showOverlay("Error en Sandbox (AI): " + event.data.error, true);
    }
});

/**
 * Carga una imagen y extrae sus píxeles (ImageData) usando un Canvas temporal offscreen
 */
function getImageData(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";

        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');

            // Dibujamos la imagen web
            ctx.drawImage(img, 0, 0);

            // Extraemos array de píxeles puros para poder enviarlos por postMessage
            resolve(ctx.getImageData(0, 0, img.width, img.height));
        };

        img.onerror = () => reject(new Error("Error de CORS al descargar la imagen. Intenta probar en otra web."));
        img.src = url;
    });
}

/**
 * Crea o actualiza un overlay temporal visual en la página web
 */
function showOverlay(message, isError = false, canvasElement = null) {
    let overlay = document.getElementById("verisight-overlay");
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.id = "verisight-overlay";
        Object.assign(overlay.style, {
            position: "fixed",
            bottom: "20px",
            right: "20px",
            width: "300px",
            padding: "15px 25px",
            backgroundColor: isError ? "#ff4c4c" : "#1a1a1a",
            color: "#ffffff",
            fontFamily: "system-ui, sans-serif",
            fontSize: "14px",
            fontWeight: "500",
            borderRadius: "8px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
            zIndex: "999999",
            transition: "opacity 0.3s ease",
            pointerEvents: "auto" // Auto para poder pulsar el botón de cerrar
        });
        document.body.appendChild(overlay);
    }

    overlay.style.backgroundColor = isError ? "#ff4c4c" : "#1a1a1a";
    overlay.innerText = message;

    // Limpiamos contenido anterior si había otro análisis
    const oldCanvas = overlay.querySelector('canvas');
    if (oldCanvas) oldCanvas.remove();
    const oldBtn = overlay.querySelector('button');
    if (oldBtn) oldBtn.remove();

    // Si pasamos un canvas (el espectro), lo añadimos al overlay
    if (canvasElement) {
        overlay.appendChild(canvasElement);
    }

    // Si es un string con saltos de línea, lo formateamos bonito
    if (message.includes("\n")) {
        overlay.innerHTML = message.replace(/\n/g, '<br/>');
        overlay.style.fontSize = "16px";
        overlay.style.textAlign = "center";
    }

    overlay.style.opacity = "1";

    if (!isError && message.includes("completado") && !canvasElement) {
        setTimeout(() => {
            overlay.style.opacity = "0";
            setTimeout(() => overlay.remove(), 300);
        }, 3000);
    } else if (canvasElement || isError) {
        // Añadimos botón manual de cierre si hay espectro visible o un error largo
        const closeBtn = document.createElement("button");
        closeBtn.innerText = "Cerrar";
        Object.assign(closeBtn.style, {
            display: "block", marginTop: "10px", padding: "5px 10px",
            background: "#444", color: "white", border: "none", borderRadius: "4px", cursor: "pointer"
        });
        closeBtn.onclick = () => overlay.remove();
        overlay.appendChild(closeBtn);
    }
}
