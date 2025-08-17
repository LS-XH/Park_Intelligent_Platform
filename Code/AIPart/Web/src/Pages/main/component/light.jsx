import React, { useEffect, useRef, useCallback } from 'react';
import * as echarts from 'echarts';
import styles from '../modulecss/light.module.css';
const Light = ({ currentData }) => {
    const chartRef = useRef(null);
    const chartInstanceRef = useRef(null);

    //初始化图表的函数
    const initChart = useCallback((color, time, totalTime) => {
        if (!chartRef.current) return;

        chartInstanceRef.current = echarts.init(chartRef.current);
        // 初始化为绿灯60秒
        updateChart(color, time, totalTime);
    });

    // 更新图表的函数
    const updateChart = useCallback((color, currentTime, totalTime) => {
        if (!chartInstanceRef.current) return;

        // 计算进度比例
        const progress = currentTime / totalTime;

        // 根据当前灯色确定颜色
        const progressColor = color === 'red' ? '#ff4d4d' : '#4dff4d';
        const backgroundColor = color === 'red' ? '#4d0000' : '#004d00';

        //图表需要的配置
        const option = {
            series: [
                {
                    type: 'pie',
                    radius: ['30%', '45%'], // 调整半径范围，确保完整显示
                    center: ['50%', '50%'], // 将中心点设置在正中心
                    avoidLabelOverlap: false,
                    hoverAnimation: false,
                    label: {
                        show: false,
                    },
                    labelLine: {
                        show: false
                    },
                    data: [
                        {
                            value: progress * 100,
                            itemStyle: {
                                color: progressColor,
                                shadowColor: color === 'red' ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 255, 0, 0.8)',
                                shadowBlur: 20
                            }
                        },
                        {
                            value: (1 - progress) * 100,
                            itemStyle: {
                                color: backgroundColor
                            }
                        }
                    ]
                },
                {
                    type: 'pie',
                    radius: ['0%', '40%'], // 保持内圈大小
                    center: ['50%', '50%'], // 与外圈保持中心对齐
                    avoidLabelOverlap: false,
                    hoverAnimation: false,
                    label: {
                        show: true,
                        position: 'center',
                        formatter: () => {
                            return `{colorLabel|${color === 'red' ? '● 红灯' : '● 绿灯'}}\n`
                        },
                        // 富文本样式
                        rich: {
                            colorLabel: {
                                fontSize: 8,
                                color: color === 'red' ? '#ff9999' : '#99ff99',
                            }
                        }
                    },
                    labelLine: {
                        show: false
                    },
                    data: [
                        { value: 1, itemStyle: { color: 'transparent' } }
                    ]
                }
            ],
            grid: {
                right: '20%',
                top: '10%',
                bottom: '10%',
                containLabel: true,
            }
        };

        chartInstanceRef.current.setOption(option);
    });


    // 初始化图表
    useEffect(() => {

        if (currentData === undefined) {
            initChart('green', 30, 60);
            return;
        }
        initChart(currentData[0], currentData[1], currentData[2]);

        const handleResize = () => {
            if (chartInstanceRef.current) {
                chartInstanceRef.current.resize();
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    //更新图表
    useEffect(() => {

        if (currentData === undefined) return;

        // 计算进度比例
        const progress = currentData[1] / currentData[2];

        // 根据当前灯色确定颜色
        const progressColor = currentData[0] === 'red' ? '#ff4d4d' : '#4dff4d';
        const backgroundColor = currentData[0] === 'red' ? '#4d0000' : '#004d00';

        chartInstanceRef.current.setOption({
            series: [{
                data: [
                    {
                        value: progress * 100,
                        itemStyle: {
                            color: progressColor,
                            shadowColor: currentData[0] === 'red' ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 255, 0, 0.8)',
                            shadowBlur: 20
                        }
                    },
                    {
                        value: (1 - progress) * 100,
                        itemStyle: {
                            color: backgroundColor
                        }
                    }
                ]
            },
            {
                label: {
                    formatter: () => {
                        return `{colorLabel|${currentData[0] === 'red' ? '● 红灯' : '● 绿灯'}}\n`
                    },
                    // 富文本样式
                    rich: {
                        colorLabel: {
                            fontSize: 8,
                            color: currentData[0] === 'red' ? '#ff9999' : '#99ff99',
                        }
                    }
                },
            }
            ]
        });

    }, [currentData]);

    // 获取状态颜色的函数
    const getStatusColor = () => {
        if (!currentData) return '#4dff4d';
        return currentData[0] === 'red' ? '#ff4d4d' : '#4dff4d';
    };

    return (
        <div className={styles.container}>
            <div className={styles.infoPanel}>
                <div className={styles.statusLabel}>
                    <span>状态:</span>
                    <span style={{ color: getStatusColor() }}>
                        {currentData ? (currentData[0] === 'red' ? '红灯' : '绿灯') : '绿灯'}
                    </span>
                </div>
                <div className={styles.timerLabel}>
                    <span> 倒计时: </span>
                    <span style={{ color: getStatusColor() }}>
                        {currentData ? currentData[1] : 30}s
                    </span>
                </div>
            </div>
            <div
                className={styles.chartContainer}
                ref={chartRef}
            />
        </div>
    );
};

export default Light;