export const getGaugeConfig = (value, max, label, Unit) => {
    return {
        series: [{
            type: 'gauge',
            center: ['50%', '70%'],
            radius: '50%',
            startAngle: 200,
            endAngle: -20,
            min: 0,
            max: max,
            splitNumber: 10,
            itemStyle: {
                color: '#00d9ff'
            },
            progress: {
                show: true,
                width: 10
            },
            pointer: {
                show: true,
                length: '70%',
                width: "10%"
            },
            axisLine: {
                lineStyle: {
                    width: 10,
                    color: [
                        [0.4545, '#00d9ff'],
                        [0.8181, '#302b63'],
                        [1, '#fc00ff']
                    ]
                }
            },
            axisTick: {
                distance: -35,
                splitNumber: 5,
                lineStyle: {
                    width: 1,
                    color: '#999'
                }
            },
            splitLine: {
                distance: -40,
                length: 10,
                lineStyle: {
                    width: 2,
                    color: '#999'
                }
            },
            axisLabel: {
                distance: -15,
                color: 'rgba(255, 255, 255, 0.85)',
                fontSize: 10
            },
            anchor: {
                show: true,
                showAbove: true,
                size: "5%",
                itemStyle: {
                    borderWidth: 3,
                    borderColor: '#00d9ff'
                }
            },
            title: {
                show: false
            },
            detail: {
                valueAnimation: true,
                width: '50%',
                lineHeight: 30,
                borderRadius: 8,
                offsetCenter: [0, '20%'],
                fontSize: "20%",
                fontWeight: 'bolder',
                color: 'rgba(255, 255, 255, 0.9)',
                formatter: `{value} ${Unit}`
            },
            data: [{
                value: value, // 实时数据
            }]
        }],
        graphic: {
            elements: [{
                type: 'text',
                $action: 'replace',
                style: {
                    text: label,
                    fontSize: 18,
                    fontWeight: 'bold',
                    lineDash: [0, 200],
                    lineDashOffset: 0,
                    fill: 'transparent',
                    stroke: '#00d9ff',
                    lineWidth: 1,
                    opacity: 0.8
                },
                keyframeAnimation: {
                    duration: 2000,
                    loop: true,
                    keyframes: [
                        {
                            percent: 0,
                            style: {
                                fill: 'transparent'
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
        },

    };
};