import React, { createContext, useContext } from 'react';
import { useWebSocket } from '../Hook/wshook';


//===知识注解===
// createContext: 用于创建 React 上下文对象，允许跨组件共享数据。
// useContext: 用于在组件中访问上下文的值。


// 创建一个上下文对象，通过 Provider 注入数据，通过 useContext 消费数据。
const WebSocketContext = createContext();


export const WebSocketProvider = ({ children, url }) => {
    const { messages, status, sendMessage } = useWebSocket(url);

    return (
        <WebSocketContext.Provider value={{ messages, status, sendMessage }}>
            {children}
        </WebSocketContext.Provider>
    );
};

//自定义Hook访问 WebSocket 上下文。
export const useWebSocketContext = () => {
    //从上下文获取值value
    const context = useContext(WebSocketContext);
    if (!context) {
        throw new Error('useWebSocketContext must be used within a WebSocketProvider');
    }
    return context;
};