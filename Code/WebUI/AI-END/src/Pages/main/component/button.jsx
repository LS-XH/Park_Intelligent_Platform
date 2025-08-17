import React, { useState } from 'react';
import styles from '../modulecss/button.module.css';
import { useWebSocketContext } from "../../../Context/wsContext";


const ControlButtons = () => {
    const [activeControls, setActiveControls] = useState({
        optimalRoute: false,
        crowdEvacuation: false,
        trafficLight: false,
        cavControl: false
    });

    const handleControlToggle = (controlName) => {
        const updatedControls = {
            ...activeControls,
            [controlName]: !activeControls[controlName]
        };

        setActiveControls(updatedControls);
        if (status === 'connected') {
            sendMessage(
                JSON.stringify(
                    {
                        status: 9,
                        message: activeControls
                    }
                ));
            console.log("发送智能按钮选择成功");
        }
    };

    const { messages, status, sendMessage } = useWebSocketContext();

    return (
        <div className={styles.container}>
            <div className={styles.background} />
            <div className={styles.glassContainer}>
                <div className={styles.header}>
                    <div className={styles.title}>智能控制系统</div>
                    <div className={styles.legend}>
                        <div className={styles.legendItem}>
                            <div className={`${styles.legendColorBox} ${styles.inactiveBox}`}></div>
                            <span className={styles.legendText}>未选中</span>
                        </div>
                        <div className={styles.legendItem}>
                            <div className={`${styles.legendColorBox} ${styles.activeBox}`}></div>
                            <span className={styles.legendText}>已选中</span>
                        </div>
                    </div>
                </div>

                <div className={styles.buttonGroup}>
                    <button
                        className={`${styles.controlButton} ${activeControls.optimalRoute ? styles.active : ''}`}
                        onClick={() => handleControlToggle('optimalRoute')}
                    >
                        最优路线选择
                    </button>

                    <button
                        className={`${styles.controlButton} ${activeControls.crowdEvacuation ? styles.active : ''}`}
                        onClick={() => handleControlToggle('crowdEvacuation')}
                    >
                        人群智能疏散
                    </button>

                    <button
                        className={`${styles.controlButton} ${activeControls.trafficLight ? styles.active : ''}`}
                        onClick={() => handleControlToggle('trafficLight')}
                    >
                        红绿灯智能调控
                    </button>

                    <button
                        className={`${styles.controlButton} ${activeControls.cavControl ? styles.active : ''}`}
                        onClick={() => handleControlToggle('cavControl')}
                    >
                        智能车辆(CAV)
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ControlButtons;