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

// Escuchar los clics en el menú contextual
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "verisight-analyze") {
        // Enviar un mensaje al content.js de la pestaña actual con la URL de la imagen
        chrome.tabs.sendMessage(tab.id, {
            action: "ANALYZE_IMAGE",
            srcUrl: info.srcUrl
        });
    }
});
