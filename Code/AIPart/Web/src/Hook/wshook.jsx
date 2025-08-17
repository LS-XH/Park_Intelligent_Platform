import { useState, useEffect } from 'react';
import getWebSocketClient from '../../util/websocket';
import { message } from 'antd';
import '@ant-design/v5-patch-for-react-19';

export const useWebSocket = (url) => {
    const [messages, setMessages] = useState([]);
    const [status, setStatus] = useState('disconnected');

    useEffect(() => {
        // 获取单例 WebSocket 客户端
        const ws = getWebSocketClient(url);

        // 定义事件处理函数
        const handleMessage = (event) => {
            try {
                console.log("接收到消息：", event.data);
                console.log("总的message", messages);

                const data = JSON.parse(event.data); // 解析新消息

                //监听紧急事件
                if (data.response.emergency === "")
                    message.info("出现突发情况");

                setMessages(prev => {
                    const filteredMessages = prev.filter(msg =>
                        msg?.status !== data.status
                    );
                    return [...filteredMessages, data];
                });
            } catch (error) {
                // 处理非JSON数据
                setMessages(prev => [...prev, { raw: event.data }]);
            }
        };

        const handleOpen = () => {
            console.log('WebSocket连接已打开');

            setStatus('connected');
        };

        const handleClose = () => {
            setStatus('disconnected');
        };

        const handleError = () => {
            console.log('WebSocket连接发生错误');
            setStatus('error');
        };

        const handleMaxReconnectAttemptsReached = () => {
            setStatus('reconnect-failed');
        };

        // 添加事件监听器
        ws.on('message', handleMessage);//接收到消息会调用这个函数
        ws.on('open', handleOpen);
        ws.on('close', handleClose);
        ws.on('error', handleError);
        ws.on('maxReconnectAttemptsReached', handleMaxReconnectAttemptsReached);

        // 如果还没有连接，则连接
        ws.connect();


        // 清理函数 - 由于没有 off 方法，我们需要改进处理方式
        return () => {
            // 清空所有事件处理程序
            ws.eventHandlers = {};
        };
    }, [url]);

    const sendMessage = (data) => {
        const ws = getWebSocketClient(url);
        console.log("发送数据：", data);
        ws.send(data);
    };

    const clearMessages = () => {
        setMessages([]);
    };

    const reconnect = () => {
        const ws = getWebSocketClient(url);
        ws.reconnectAttempts = 0;
        ws.shouldReconnect = true;
        ws.connect();
    };

    return {
        messages,
        status,
        sendMessage,
        clearMessages,
        reconnect
    };
};