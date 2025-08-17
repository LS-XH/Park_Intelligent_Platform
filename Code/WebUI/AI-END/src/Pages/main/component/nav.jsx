import React, { useState, useEffect } from 'react';
import styles from '../modulecss/nav.module.css';
import { useWebSocketContext } from "../../../Context/wsContext";



const RouteSearch = ({ onClose }) => {
    const [startPoint, setStartPoint] = useState('我的位置');
    const [endPoint, setEndPoint] = useState('');
    const [activeInput, setActiveInput] = useState(null);
    const { messages, status, sendMessage } = useWebSocketContext();
    const [routePoints, setRoutePoints] = useState(["无", "无", "无", "无"]); // 新增状态存储路线点

    const handleStartChange = (e) => {
        setStartPoint(e.target.value);
    };

    const handleEndChange = (e) => {
        setEndPoint(e.target.value);
    };

    const handleSearch = () => {
        if (status === 'connected') {
            sendMessage(
                JSON.stringify(
                    {
                        status: 10,
                        message: {
                            startName: startPoint,
                            endName: endPoint
                        }
                    }
                ));
            console.log("发送成功");
        }
    };

    const handleInputFocus = (field) => {
        setActiveInput(field);
    };

    const handleInputBlur = () => {
        setActiveInput(null);
    };

    // 处理接收到的消息并更新
    useEffect(() => {
        const currentMessage = messages.filter(msg => msg.status === 10);
        if (currentMessage.length === 0) return;
        const response = currentMessage[0].response;
        console.log("获取到的路段数据:", response);
        setRoutePoints(response.route);
    }, [messages]);


    return (
        <div className={styles.container}>
            {/* 添加动态背景元素以更好地展示毛玻璃效果 */}
            <div className={styles.background} />

            {/* 毛玻璃容器 */}
            <div className={styles.glassContainer}>
                <div className={styles.searchContainer}>
                    <div className={styles.searchHeader}>
                        <div className={styles.searchTitle}>路线规划</div>
                        {/* <button className={styles.closeButton} onClick={onClose}>×</button> */}
                    </div>

                    <div className={styles.searchInputGroup}>
                        <div
                            className={`${styles.inputWrapper} ${activeInput === 'start' ? styles.active : ''}`}
                            onClick={() => document.getElementById('startInput').focus()}
                        >
                            <span className={styles.inputIcon}>A</span>
                            <input
                                id="startInput"
                                type="text"
                                value={startPoint}
                                onChange={handleStartChange}
                                className={styles.inputField}
                                placeholder="请输入起点"
                                onFocus={() => handleInputFocus('start')}
                                onBlur={handleInputBlur}
                            />
                        </div>
                        <div
                            className={`${styles.inputWrapper} ${activeInput === 'end' ? styles.active : ''}`}
                            onClick={() => document.getElementById('endInput').focus()}
                        >
                            <span className={styles.inputIcon}>B</span>
                            <input
                                id="endInput"
                                type="text"
                                value={endPoint}
                                onChange={handleEndChange}
                                className={styles.inputField}
                                placeholder="请输入终点"
                                onFocus={() => handleInputFocus('end')}
                                onBlur={handleInputBlur}
                            />
                        </div>

                        {/* 添加途径站点显示区域 */}
                        <div className={styles.routeStationsContainer}>
                            <div className={styles.stationsLabel}>途径站点:</div>
                            <div className={styles.stationsList}>
                                {routePoints && routePoints.length > 0 ? (
                                    routePoints.map((station, index) => (
                                        <span key={index} className={styles.stationItem}>
                                            {station}
                                        </span>
                                    ))
                                ) : (
                                    <span className={styles.noStations}>暂无途径站点信息</span>
                                )}
                            </div>
                        </div>
                    </div>

                    <button className={styles.searchButton} onClick={handleSearch}>
                        搜索路线
                    </button>
                </div>
            </div>
        </div>
    );
};

export default RouteSearch;