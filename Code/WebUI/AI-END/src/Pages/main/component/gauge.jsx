import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

const Board = () => {
    const chartRef = useRef(null);
    const myChart = useRef(null);

    useEffect(() => {
        // 初始化图表
        myChart.current = echarts.init(chartRef.current);

        // 图表配置
        const option = {
            series: [{
                type: 'gauge',
                center: ['50%', '60%'],
                startAngle: 200,
                endAngle: -20,
                min: 0,
                max: 220,
                splitNumber: 11,
                itemStyle: {
                    color: '#00d9ff'
                },
                progress: {
                    show: true,
                    width: 30
                },
                pointer: {
                    show: true,
                    length: '75%',
                    width: 5
                },
                axisLine: {
                    lineStyle: {
                        width: 30,
                        color: [
                            [0.4545, '#00d9ff'],
                            [0.8181, '#302b63'],
                            [1, '#fc00ff']
                        ]
                    }
                },
                axisTick: {
                    distance: -45,
                    splitNumber: 5,
                    lineStyle: {
                        width: 2,
                        color: '#999'
                    }
                },
                splitLine: {
                    distance: -52,
                    length: 14,
                    lineStyle: {
                        width: 3,
                        color: '#999'
                    }
                },
                axisLabel: {
                    distance: -20,
                    color: 'rgba(255, 255, 255, 0.85)',
                    fontSize: 12
                },
                anchor: {
                    show: true,
                    showAbove: true,
                    size: 20,
                    itemStyle: {
                        borderWidth: 8,
                        borderColor: '#00d9ff'
                    }
                },
                title: {
                    show: false
                },
                detail: {
                    valueAnimation: true,
                    width: '60%',
                    lineHeight: 40,
                    borderRadius: 8,
                    offsetCenter: [0, '10%'],
                    fontSize: 32,
                    fontWeight: 'bolder',
                    color: 'rgba(255, 255, 255, 0.9)',
                    formatter: '{value} km/h'
                },
                data: [{
                    value: 80
                }]
            }],
            graphic: {
                elements: [{
                    type: 'text',
                    $action: 'replace',
                    style: {
                        text: 'SPEED',
                        fontSize: 24,
                        fontWeight: 'bold',
                        lineDash: [0, 200],
                        lineDashOffset: 0,
                        fill: 'transparent',
                        stroke: '#00d9ff',
                        lineWidth: 1,
                        opacity: 0.8
                    },
                    keyframeAnimation: {
                        duration: 3000,
                        loop: true,
                        keyframes: [
                            {
                                percent: 0.7,
                                style: {
                                    fill: 'transparent'
                                }
                            },
                            {
                                percent: 0.8,
                                style: {
                                    fill: '#00d9ff'
                                }
                            },
                            {
                                percent: 1,
                                style: {
                                    fill: '#00d9ff'
                                }
                            }
                        ]
                    }
                }]
            }
        };

        // 设置配置项
        myChart.current.setOption(option);

        // 模拟速度变化
        const simulateSpeedChange = () => {
            setInterval(() => {
                if (myChart.current) {
                    const randomValue = Math.round(Math.random() * 200);
                    myChart.current.setOption({
                        series: [{
                            data: [{
                                value: randomValue
                            }]
                        }]
                    });
                }
            }, 3000);
        };

        simulateSpeedChange();

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

    return (
        <div style={{

            width: '80%',
            height: '100%',
            margin: '0 auto',
            minHeight: '400px',
            borderRadius: '16px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            overflow: 'hidden',
        }}>
            {/* 添加模糊背景效果元素 */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                background: `
          radial-gradient(circle at 15% 25%, rgba(255, 0, 199, 0.2) 0%, transparent 20%),
          radial-gradient(circle at 85% 35%, rgba(0, 217, 255, 0.2) 0%, transparent 20%),
          radial-gradient(circle at 45% 85%, rgba(128, 0, 255, 0.2) 0%, transparent 20%)
        `,
                zIndex: 0
            }} />

            {/* 毛玻璃容器 */}
            <div style={{
                width: '100%',
                minHeight: '400px',
                background: 'rgba(255, 255, 255, 0.15)',
                borderRadius: '16px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
                position: 'relative',
                zIndex: 1,
                padding: '20px'
            }}>
                <div
                    ref={chartRef}
                    style={{
                        width: '100%',
                        height: '400px',
                        zIndex: 1
                    }}
                />
                <div style={{
                    color: 'rgba(255, 255, 255, 0.9)',
                    textAlign: 'center',
                    zIndex: 1
                }}>
                    <h2 style={{
                        margin: 0,
                        fontWeight: 300,
                        fontSize: '1.5rem',
                        letterSpacing: '1px'
                    }}>实时速度监控</h2>
                    <p style={{
                        margin: '5px 0 0 0',
                        fontWeight: 200,
                        fontSize: '0.9rem',
                        opacity: 0.8
                    }}>当前速度状态显示</p>
                </div>
            </div>

        </div>
    );
};

export default Board;