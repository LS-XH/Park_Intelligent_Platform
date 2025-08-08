import { useState, useEffect, useRef } from 'react';
import WebSocketClient from '../websocket';

export const useWebSocket = (url) => {
    const [messages, setMessages] = useState([]);
    const [status, setStatus] = useState('disconnected');
    const websocketRef = useRef(null);

    useEffect(() => {
        const ws = new WebSocketClient(url);

        ws.on('open', () => {
            setStatus('connected');
        });

        ws.on('message', (event) => {
            setMessages(prev => [...prev, event.data]);
        });

        ws.on('close', () => {
            setStatus('disconnected');
        });

        ws.on('error', () => {
            setStatus('error');
        });

        // 添加重连尝试事件监听
        ws.on('maxReconnectAttemptsReached', () => {
            setStatus('reconnect-failed');
        });
        // 连接WebSocket服务器
        ws.connect();
        websocketRef.current = ws;

        return () => {
            ws.close();
        };
    }, [url]);

    const sendMessage = (data) => {
        if (websocketRef.current) {
            console.log("发送数据：", data);
            websocketRef.current.send(data);
        }
    };

    const clearMessages = () => {
        setMessages([]);
    };
    const reconnect = () => {
        if (websocketRef.current) {
            // 重置重连尝试次数
            websocketRef.current.reconnectAttempts = 0;
            websocketRef.current.shouldReconnect = true;
            websocketRef.current.connect();
        }
    };

    return {
        messages,
        status,
        sendMessage,
        clearMessages,
        reconnect
    };
};