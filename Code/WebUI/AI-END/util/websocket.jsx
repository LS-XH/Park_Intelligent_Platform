class WebSocketClient {
    constructor(url, protocols = []) {
        this.url = url;
        this.protocols = protocols;//子协议
        this.websocket = null;
        this.eventHandlers = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 1000;
        this.shouldReconnect = true;
    }

    // 连接WebSocket服务器
    connect() {
        try {
            this.websocket = new WebSocket(this.url, this.protocols);

            console.log("尝试创建一个实例", this.websocket);


            this.websocket.onopen = (event) => {
                this.reconnectAttempts = 0; // 重置重连次数
                this.handleEvent('open', event);
            };

            this.websocket.onmessage = (event) => {
                console.log("WebSocket收到消息:", event);

                this.handleEvent('message', event);
            };

            this.websocket.onclose = (event) => {
                this.handleEvent('close', event);
                // 这里是判断是否要进行重连
                //通过调用 close() 方法正常关闭或者服务器端正常关闭连接，wasClean是true
                console.log("尝试重连...");
                this.attemptReconnect();
            }

            this.websocket.onerror = (error) => {
                console.log("WebSocket错误:", error);
                this.handleEvent('error', error);
            };

            return this.websocket;
        } catch (error) {
            console.error('WebSocket连接失败:', error);
            this.handleEvent('error', error);
            this.attemptReconnect();
        }
    }

    // 尝试重连
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`尝试重连 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => {
                this.connect();
            }, this.reconnectInterval * this.reconnectAttempts); // 逐渐增加重连间隔
        } else {
            console.error('达到最大重连次数，停止重连');
            this.handleEvent('maxReconnectAttemptsReached');
        }
    }

    // 发送消息
    send(data) {
        console.log("websocket的状态", this.websocket);

        if (this.websocket.readyState === WebSocket.OPEN)
            this.websocket.send(data);
    }

    // 关闭连接（不触发重连）
    close(code, reason) {
        this.shouldReconnect = false;
        if (this.websocket) {
            this.websocket.close(code, reason);
        }
    }

    // 添加事件监听器
    on(event, handler) {
        if (!this.eventHandlers[event]) {
            this.eventHandlers[event] = [];
        }
        this.eventHandlers[event].push(handler);
    }

    // 处理事件
    handleEvent(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`处理${event}事件时出错:`, error);
                }
            });
        }
    }

    // 获取连接状态
    get readyState() {
        return this.websocket ? this.websocket.readyState : WebSocketClient.CLOSED;
    }
}

// 创建单例映射，支持不同 URL 创建不同实例
const websocketInstances = new Map();

// 导出工厂函数而不是类本身
export default function getWebSocketClient(url, protocols = []) {
    if (!websocketInstances.has(url)) {
        console.log("创建新的WebSocketClient实例");
        websocketInstances.set(url, new WebSocketClient(url, protocols));
    }
    return websocketInstances.get(url);
}