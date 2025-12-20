// background.js - chỉ xử lý trạng thái và logging
let isRecording = false;
let currentTabId = null;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('📨 Background received:', request.action);
    
    switch (request.action) {
        case 'capture_started':
            isRecording = true;
            currentTabId = request.tabId;
            console.log('🎯 Capture started for tab:', currentTabId);
            sendResponse({ success: true });
            break;
            
        case 'capture_stopped':
            isRecording = false;
            console.log('🛑 Capture stopped');
            sendResponse({ success: true });
            break;
            
        case 'get_status':
            sendResponse({ 
                isRecording, 
                tabId: currentTabId 
            });
            break;
    }
    
    return true;
});