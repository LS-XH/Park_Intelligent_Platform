import React, { use, useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import styles from '../modulecss/lineChart.module.css'
import { useWebSocketContext } from "../../../Context/wsContext";


const LineChart = () => {
    const chartRef = useRef(null);
    const myChart = useRef(null);
    const [trafficData, setTrafficData] = useState([97.6, 56.9, 45.2, 53.0, 4.7, 98.7, 59.2, 64.1, 29.6, 40.0, 16.7, 77.6]);
    const { messages, sendMessage, status } = useWebSocketContext();

    useEffect(() => {
        const currentMessage = messages.filter(msg => msg.status === 8)
        if (currentMessage.length === 0) return;
        console.log("从服务器获取到的风险数据", currentMessage[0]?.response);
        setTrafficData(currentMessage[0].response.riskData);
    }, [messages]);


    // useEffect(() => {
    //     // setTimeout(() => {
    //     // }, 1000);
    //     if (status === 'connected') {
    //         sendMessage(
    //             JSON.stringify(
    //                 {
    //                     status: 7,
    //                     message: {
    //                         id: 1
    //                     }
    //                 }
    //             ));
    //         console.log("发送成功");
    //     }
    // }, [status]);
    useEffect(() => {
        // 初始化图表
        myChart.current = echarts.init(chartRef.current);
        // 生成模拟数据
        const xAxisData = [];
        let seriesData;
        for (let i = 0; i < 12; i++) {
            xAxisData.push(`${i + 1}`);
        }

        seriesData = trafficData;

        // 图表配置
        const option = {
            title: {
                text: '风险趋势',
                textStyle: {
                    color: 'rgba(255, 255, 255, 0.9)',
                    fontSize: 16,
                    fontWeight: 300
                },
                left: 'center',
                top: 10
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(30, 30, 46, 0.8)',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1,
                textStyle: {
                    color: 'rgba(255, 255, 255, 0.9)'
                }
            },
            xAxis: {
                type: 'category',
                data: xAxisData,
                axisLine: {
                    lineStyle: {
                        color: 'rgba(255, 255, 255, 0.3)'
                    }
                },
                axisLabel: {
                    color: 'rgba(255, 255, 255, 0.7)',
                    fontSize: 10
                },
                axisTick: {
                    lineStyle: {
                        color: 'rgba(255, 255, 255, 0.3)'
                    }
                }
            },
            yAxis: {
                type: 'value',
                splitLine: {
                    lineStyle: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: 'rgba(255, 255, 255, 0.3)'
                    }
                },
                axisLabel: {
                    color: 'rgba(255, 255, 255, 0.7)',
                    fontSize: 10
                }
            },
            series: [{
                data: seriesData,
                type: 'line',
                smooth: true,
                symbol: 'circle',
                symbolSize: 6,
                lineStyle: {
                    color: '#00d9ff',
                    width: 3
                },
                itemStyle: {
                    color: '#00d9ff',
                    borderWidth: 2,
                    borderColor: '#fff'
                },
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0,
                        y: 0,
                        x2: 0,
                        y2: 1,
                        colorStops: [{
                            offset: 0,
                            color: 'rgba(0, 217, 255, 0.3)'
                        }, {
                            offset: 1,
                            color: 'rgba(0, 217, 255, 0.01)'
                        }]
                    }
                }
            }],
            grid: {
                left: '5%',
                right: '5%',
                top: '20%',
                bottom: '15%',
                containLabel: true
            }
        };

        // 设置配置项
        myChart.current.setOption(option);



        // 响应式处理
        const handleResize = () => {
            if (myChart.current) {
                myChart.current.resize();
            }
        };

        window.addEventListener('resize', handleResize);

        // 清理函数
        return () => {
            window.removeEventListener('resize', handleResize);
            if (myChart.current) {
                myChart.current.dispose();
            }
        };
    }, []);

    useEffect(() => {
        // 增量更新 ECharts 配置（只更新数据部分）
        myChart.current.setOption({
            series: [{
                data: trafficData // 使用最新的 trafficData 更新图表数据
            }]
        }, false); // 第二个参数为 true：表示不合并旧配置，只应用当前设置的属性
    }, [trafficData]); // 依赖 trafficData 变化：当图表数据更新时触发



    return (
        <div className={styles.container}>
            {/* 添加动态背景元素以更好地展示毛玻璃效果 */}
            <div className={styles.background} />

            {/* 毛玻璃容器 */}
            <div className={styles.glassContainer}>
                <div
                    ref={chartRef}
                    className={styles.chart}
                />
                <div className={styles.title}>
                    <div>
                        <span className={styles.infoItem}>
                            <span className={styles.infoLabel}>当前路段:</span>
                            <span className={styles.infoValue}>{trafficData.name ? trafficData.name : '全路段'}</span>
                        </span>
                        <span className={styles.infoItem}>
                            <span className={styles.infoLabel}>当前节点:</span>
                            <span className={styles.infoValue}>{trafficData.road_names ? trafficData.road_names : '无选中节点'}</span>
                        </span>
                    </div>
                </div>
            </div>


        </div>
    );
};

export default LineChart;