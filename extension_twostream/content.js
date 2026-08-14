let sandboxWindow = null;

// Escuchar mensajes del background.js (clic derecho)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "ANALYZE_IMAGE") {
        iniciarSeleccion(request.srcUrl);
    }
});

// 1. Descarga la imagen y abre la interfaz de recorte (Bounding Box)

async function iniciarSeleccion(url) {
    try {
        await setupSandbox();
        showOverlay("Abriendo modo lupa...");

        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {
            const overlayInfo = document.getElementById("verisight-overlay");
            if (overlayInfo) overlayInfo.remove();

            abrirLupa(img);
        };
        img.onerror = () => {
            showOverlay("Error al descargar la imagen. Inténtalo de nuevo.", true);
        };
        img.src = url;

    } catch (error) {
        console.error("[VeriSight] Error:", error);
        showOverlay("Error: " + error.message, true);
    }
}

function abrirLupa(img) {
    const oldContainer = document.getElementById('verisight-crop-container');
    if (oldContainer) oldContainer.remove();

    const container = document.createElement('div');
    container.id = 'verisight-crop-container';
    Object.assign(container.style, {
        position: 'fixed', top: '0', left: '0', width: '100vw', height: '100vh',
        backgroundColor: 'rgba(0, 0, 0, 0.85)', zIndex: '9999999',
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        userSelect: 'none', cursor: 'crosshair'
    });

    const title = document.createElement('h2');
    title.innerText = "Haz clic en la zona a analizar";
    Object.assign(title.style, {
        color: 'white', fontFamily: 'sans-serif', marginBottom: '20px', pointerEvents: 'none'
    });
    container.appendChild(title);

    const imgWrapper = document.createElement('div');
    imgWrapper.style.position = 'relative';
    imgWrapper.style.maxWidth = '90vw';
    imgWrapper.style.maxHeight = '80vh';

    const displayImg = document.createElement('img');
    displayImg.src = img.src;
    displayImg.draggable = false;
    Object.assign(displayImg.style, {
        display: 'block', maxWidth: '100%', maxHeight: '80vh', objectFit: 'contain'
    });
    imgWrapper.appendChild(displayImg);
    container.appendChild(imgWrapper);

    const selectionBox = document.createElement('div');
    Object.assign(selectionBox.style, {
        position: 'absolute', border: '3px solid #ff4c4c', backgroundColor: 'rgba(255, 76, 76, 0.2)',
        pointerEvents: 'none', display: 'none'
    });
    imgWrapper.appendChild(selectionBox);
    document.body.appendChild(container);

    const cancelBtn = document.createElement('button');
    cancelBtn.innerText = "Cancelar";
    Object.assign(cancelBtn.style, {
        marginTop: '20px', padding: '10px 20px', background: '#444', color: 'white', border: 'none',
        borderRadius: '5px', cursor: 'pointer', fontSize: '16px'
    });
    cancelBtn.onclick = () => container.remove();
    container.appendChild(cancelBtn);

    setTimeout(() => {
        const rect = displayImg.getBoundingClientRect();
        const scaleX = rect.width / img.naturalWidth;
        const scaleY = rect.height / img.naturalHeight;

        // Forzar 224x224 en el espacio visual
        const visualWidth = 224 * scaleX;
        const visualHeight = 224 * scaleY;

        selectionBox.style.width = visualWidth + 'px';
        selectionBox.style.height = visualHeight + 'px';
        selectionBox.style.display = 'block';

        displayImg.addEventListener('mousemove', (e) => {
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;

            let left = currentX - visualWidth / 2;
            let top = currentY - visualHeight / 2;

            left = Math.max(0, Math.min(left, rect.width - visualWidth));
            top = Math.max(0, Math.min(top, rect.height - visualHeight));

            selectionBox.style.left = left + 'px';
            selectionBox.style.top = top + 'px';
        });

        displayImg.addEventListener('click', (e) => {
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;

            let visualLeft = currentX - visualWidth / 2;
            let visualTop = currentY - visualHeight / 2;

            visualLeft = Math.max(0, Math.min(visualLeft, rect.width - visualWidth));
            visualTop = Math.max(0, Math.min(visualTop, rect.height - visualHeight));

            const realLeft = visualLeft / scaleX;
            const realTop = visualTop / scaleY;

            container.remove();
            extraerParcheYAnalizar(img, realLeft, realTop);
        });
    }, 100);
}

function extraerParcheYAnalizar(img, realLeft, realTop) {
    showOverlay("Procesando recorte seleccionado...");

    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');

    // Fondo negro por si la imagen es pequeña
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 224, 224);

    let sWidth = 224;
    let sHeight = 224;
    let dx = 0;
    let dy = 0;

    if (img.naturalWidth < 224) {
        sWidth = img.naturalWidth;
        dx = (224 - sWidth) / 2;
        realLeft = 0;
    }

    if (img.naturalHeight < 224) {
        sHeight = img.naturalHeight;
        dy = (224 - sHeight) / 2;
        realTop = 0;
    }

    ctx.drawImage(img, realLeft, realTop, sWidth, sHeight, dx, dy, sWidth, sHeight);

    const imageData = ctx.getImageData(0, 0, 224, 224);

    sandboxWindow.postMessage({
        action: "ANALYZE_IMAGE",
        imageData: imageData
    }, "*");
}


//Inyecta un iframe oculto que carga el sandbox.html
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
        iframe.style.display = 'none';
        iframe.src = chrome.runtime.getURL('sandbox.html');

        iframe.onload = () => {
            sandboxWindow = iframe.contentWindow;
            resolve();
        };

        document.body.appendChild(iframe);
    });
}

// Escuchar las respuestas del Sandbox
window.addEventListener("message", (event) => {
    if (event.data && event.data.action === "ANALYSIS_COMPLETE") {
        const prob = event.data.prediction * 100;
        const facesCount = event.data.facesCount || 1;
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


//Overlay de notificaciones
function showOverlay(message, isError = false) {
    let overlay = document.getElementById("verisight-overlay");
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.id = "verisight-overlay";
        Object.assign(overlay.style, {
            position: "fixed", bottom: "20px", right: "20px", width: "300px", padding: "15px 25px",
            backgroundColor: isError ? "#ff4c4c" : "#1a1a1a", color: "#ffffff",
            fontFamily: "system-ui, sans-serif", fontSize: "14px", fontWeight: "500",
            borderRadius: "8px", boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
            zIndex: "9999999", transition: "opacity 0.3s ease", pointerEvents: "auto"
        });
        document.body.appendChild(overlay);
    }

    overlay.style.backgroundColor = isError ? "#ff4c4c" : "#1a1a1a";
    overlay.innerText = message;

    const oldBtn = overlay.querySelector('button');
    if (oldBtn) oldBtn.remove();

    if (message.includes("\n")) {
        overlay.innerHTML = message.replace(/\n/g, '<br/>');
        overlay.style.fontSize = "16px";
        overlay.style.textAlign = "center";
    }

    overlay.style.opacity = "1";

    if (!isError && message.includes("IA")) { // Mensaje de resultado
        setTimeout(() => {
            overlay.style.opacity = "0";
            setTimeout(() => overlay.remove(), 300);
        }, 5000);
    } else if (isError) {
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
