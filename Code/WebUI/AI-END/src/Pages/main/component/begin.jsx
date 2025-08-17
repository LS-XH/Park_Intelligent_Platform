import { useWebSocketContext } from "../../../Context/wsContext";
import { Button } from "antd";
import styles from "../../../App.module.css";
import { useState } from "react";

export const Buttoncomponent = () => {
    const { status, sendMessage } = useWebSocketContext();
    const [isRunning, setIsRunning] = useState(false);

    const handleToggle = () => {
        console.log("检查连接状态".status);

        if (status === 'connected') {
            const newStatus = !isRunning;
            sendMessage(JSON.stringify({
                status: 12,
                message: {
                    process: newStatus ? 'start' : 'stop',
                    time: 30
                }
            }));
            setIsRunning(newStatus);
            console.log(`发送${newStatus ? '启动' : '停止'}指令`);
        }
    };

    return (
        <Button
            style={{
                backgroundColor: 'rgba(255, 255, 255, 0.2)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                color: '#00d9ff'
            }}
            className={styles.button}
            onClick={handleToggle}
            disabled={status !== 'connected'}
        >
            {isRunning ? '模拟结束' : '模拟开始'}
        </Button>
    )
};