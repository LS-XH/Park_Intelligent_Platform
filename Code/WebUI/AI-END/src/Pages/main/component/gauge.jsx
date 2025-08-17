import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import { getGaugeConfig } from '../../../Hook/gaugeHook';
import styles from '../modulecss/gauge.module.css';
import Light from "./light";
import { useWebSocketContext } from "../../../Context/wsContext";


const Gauge = () => {
    const { messages, status, sendMessage } = useWebSocketContext();
    const chartRefPeople = useRef(null);
    const PeopleChart = useRef(null);
    const chartRefCar = useRef(null);
    const CarChart = useRef(null);
    const chartRefCrowding = useRef(null);
    const CrowdingChart = useRef(null);
    const [trafficData, setTrafficData] = useState({ people: 0, cars: 0, crowding: 0 });

    // 处理接收到的消息并更新图表
    useEffect(() => {
        const currentMessage = messages.filter(msg => msg.status === 5)
        if (currentMessage.length === 0) return;
        setTrafficData(currentMessage[0].response);
    }, [messages]);

    // 发送消息给服务器
    useEffect(() => {
        // setTimeout(() => { 
        // }, 1000);
        if (status === 'connected') {
            sendMessage(
                JSON.stringify(
                    {
                        status: 3,
                        message: {
                            id: 1
                        }
                    }
                ));
            console.log("发送成功");
        }
    }, [status]);



    // 统一的 resize 处理函数
    const handleResize = () => {
        if (PeopleChart.current) {
            PeopleChart.current.resize();
        }
        if (CarChart.current) {
            CarChart.current.resize();
        }
        if (CrowdingChart.current) {
            CrowdingChart.current.resize();
        }
    };

    // 初始化人流量图表
    useEffect(() => {
        if (chartRefPeople.current) {
            PeopleChart.current = echarts.init(chartRefPeople.current);
            const option = getGaugeConfig(trafficData.people, 20000, '人流量', '');
            PeopleChart.current.setOption(option);
        }

        // 添加 resize 监听器
        window.addEventListener('resize', handleResize);

        return () => {
            if (PeopleChart.current) {
                PeopleChart.current.dispose();
            }
        };
    }, []);

    // 初始化车流量图表
    useEffect(() => {
        if (chartRefCar.current) {
            CarChart.current = echarts.init(chartRefCar.current);
            const option = getGaugeConfig(trafficData.cars, 5000, '车流量', '');
            CarChart.current.setOption(option);
        }

        return () => {
            if (CarChart.current) {
                CarChart.current.dispose();
            }
        };
    }, []);

    // 初始化拥挤程度图表
    useEffect(() => {
        if (chartRefCrowding.current) {
            CrowdingChart.current = echarts.init(chartRefCrowding.current);
            const option = getGaugeConfig(trafficData.crowding, 100, '拥挤度', '');
            CrowdingChart.current.setOption(option);
        }

        return () => {
            if (CrowdingChart.current) {
                CrowdingChart.current.dispose();
            }
        };
    }, []);

    // 统一处理所有图表的数据更新
    useEffect(() => {
        // 更新人流量图表数据
        if (PeopleChart.current) {
            PeopleChart.current.setOption({
                series: [{
                    data: [{
                        value: trafficData.people
                    }]
                }]
            });
        }

        // 更新车流量图表数据
        if (CarChart.current) {
            CarChart.current.setOption({
                series: [{
                    data: [{
                        value: trafficData.cars
                    }]
                }]
            });
        }

        // 更新拥挤程度图表数据
        if (CrowdingChart.current) {
            CrowdingChart.current.setOption({
                series: [{
                    data: [{
                        value: trafficData.crowding
                    }]
                }]
            });
        }
    }, [trafficData]);

    // 清理 resize 监听器
    useEffect(() => {
        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    return (
        <div className={styles.container}>
            {/* 添加模糊背景效果元素 */}
            <div className={styles.background} />

            {/* 毛玻璃容器 */}
            <div className={styles.glassContainer}>


                <div className={styles.chartContainer}>
                    <div
                        ref={chartRefPeople}
                        className={styles.chart}
                    />
                    <div
                        ref={chartRefCar}
                        className={styles.chart}
                    />
                    <div
                        ref={chartRefCrowding}
                        className={styles.chart}
                    />
                </div>
                <Light className={styles.chartlight} currentData={trafficData.trafficLight}></Light>
                <div className={styles.title}>
                    <div>
                        <span className={styles.infoItem}>
                            <span className={styles.infoLabel}>当前路段:</span>
                            <span className={styles.infoValue}>
                                {trafficData?.status === 5 ? (trafficData.name ? trafficData.name : '全路段') : '全路段'}
                            </span>
                        </span>
                        <span className={styles.infoItem}>
                            <span className={styles.infoLabel}>当前节点:</span>
                            <span className={styles.infoValue}>
                                {trafficData?.status === 6 ? (trafficData.name ? trafficData.name : '无选中节点') : '无选中节点'}
                            </span>
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Gauge;