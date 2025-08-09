import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

const LineChart = () => {
    const chartRef = useRef(null);
    const myChart = useRef(null);

    useEffect(() => {
        // 初始化图表
        myChart.current = echarts.init(chartRef.current);

        // 生成模拟数据
        const xAxisData = [];
        const seriesData = [];
        for (let i = 0; i < 30; i++) {
            xAxisData.push(`Day ${i + 1}`);
            seriesData.push(Math.floor(Math.random() * 100) + 20);
        }

        // 图表配置
        const option = {
            title: {
                text: '数据趋势',
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
            {/* 添加动态背景元素以更好地展示毛玻璃效果 */}
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
                        height: '100%',
                        minHeight: '350px'
                    }}
                />
            </div>
        </div>
    );
};

export default LineChart;