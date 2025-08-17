import React, { useState } from 'react';
import styles from '../modulecss/contorl.module.css';
import { useWebSocketContext } from "../../../Context/wsContext";

const EmergencyControl = () => {
    const [selectedEmergency, setSelectedEmergency] = useState('');
    const { messages, status, sendMessage } = useWebSocketContext();

    const emergencyOptions = ['车辆失灵', '车祸', '道路维修', '严重灾害'];

    const handleEmergencyChange = (event) => {
        const emergency = event.target.value;
        setSelectedEmergency(emergency);

        const message = messages.filter(msg => msg.status === 5)

        if (emergency && status === 'connected') {
            sendMessage(
                JSON.stringify({
                    status: 11,
                    message: {
                        name: message[0].response.name,
                        emergencyType: emergency
                    }
                })
            );
            console.log(`发送突发事件: ${emergency}`);
        }
    };

    return (
        <div className={styles.container}>
            <div className={styles.background} />
            <div className={styles.glassContainer}>
                <div className={styles.header}>
                    <div className={styles.title}>突发事件</div>
                </div>
                <div className={styles.emergencyControl}>
                    <div className={styles.selectContainer}>
                        <select
                            className={styles.emergencySelect}
                            value={selectedEmergency}
                            onChange={handleEmergencyChange}
                        >
                            <option value="">请选择突发事件</option>
                            {emergencyOptions.map((option, index) => (
                                <option key={index} value={option}>
                                    {option}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className={styles.statusDisplay}>
                        <span className={styles.statusText}>
                            {selectedEmergency ? `已选择: ${selectedEmergency}` : '无突发事件'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default EmergencyControl;