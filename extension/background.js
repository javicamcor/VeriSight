/**
 * VeriSight - Background Service Worker
 * Se encarga de gestionar el ciclo de vida de la extensión y 
 * proporcionar el menú contextual nativo.
 */

// Crear el menú contextual al instalar la extensión
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "verisight-analyze",
        title: "Analizar con VeriSight",
        contexts: ["image"] // Solo aparece al hacer clic derecho sobre una imagen
    });
});

// Función segura para convertir ArrayBuffer a Base64 en un Service Worker (donde FileReader no existe)
async function arrayBufferToBase64(buffer, mimeType) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    // Chunking para evitar el límite del Call Stack en imágenes muy grandes
    const chunkSize = 8192;
    for (let i = 0; i < bytes.length; i += chunkSize) {
        const chunk = bytes.subarray(i, i + chunkSize);
        binary += String.fromCharCode.apply(null, chunk);
    }
    return `data:${mimeType};base64,${btoa(binary)}`;
}

// Escuchar los clics en el menú contextual
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "verisight-analyze") {
        try {
            // El background script tiene súper-poderes de CORS gracias a host_permissions
            const response = await fetch(info.srcUrl);
            const blob = await response.blob();
            const buffer = await blob.arrayBuffer();
            
            // Convertimos la imagen pura a texto Base64
            const base64Url = await arrayBufferToBase64(buffer, blob.type);
            
            // Enviamos la imagen incrustada al content.js
            chrome.tabs.sendMessage(tab.id, {
                action: "ANALYZE_IMAGE",
                srcUrl: base64Url
            });
            
        } catch (error) {
            console.error("Error saltando CORS:", error);
            // Fallback a URL normal
            chrome.tabs.sendMessage(tab.id, {
                action: "ANALYZE_IMAGE",
                srcUrl: info.srcUrl
            });
        }
    }
});
