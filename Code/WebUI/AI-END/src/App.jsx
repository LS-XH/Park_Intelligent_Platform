import { useWebSocket } from './Hook/wshook';
import './App.css';

function App() {
  // 使用 useWebSocket 钩子 - 更新解构以匹配Hook的返回值
  const { messages, status, sendMessage, clearMessages, reconnect } = useWebSocket('ws://192.168.1.243:5566/ws');

  return (
    <>
      <div>
        <h1>WebSocket 状态: {status}</h1>
        <button
          onClick={() => sendMessage('Hello from React!')}
          disabled={status !== 'connected'}
        >
          发送消息
        </button>
        <button onClick={clearMessages}>
          清除消息
        </button>
        {/* 添加重连按钮 */}
        <button onClick={reconnect} disabled={status === 'connected'}>
          重新连接
        </button>
        <div>
          <h2>收到的消息:</h2>
          <ul>
            {messages.map((msg, index) => (
              <li key={index}>{msg}</li>
            ))}
          </ul>
        </div>
      </div>
    </>
  );
}

export default App;