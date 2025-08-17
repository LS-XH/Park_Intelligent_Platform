import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import styles from '../modulecss/heatmap.module.css';
import { useWebSocketContext } from "../../../Context/wsContext";

const DensityMap = () => {


    const { messages, status, sendMessage } = useWebSocketContext();
    // const chartRef = useRef(null);
    // const chartInstance = useRef(null);
    const [trafficData, setTrafficData] = useState(null);

    // useEffect(() => {
    //     if (chartRef.current) {
    //         chartInstance.current = echarts.init(chartRef.current);
    //         renderChart();
    //     }

    //     const handleResize = () => {
    //         if (chartInstance.current) {
    //             chartInstance.current.resize();
    //         }
    //     };

    //     window.addEventListener('resize', handleResize);

    //     return () => {
    //         window.removeEventListener('resize', handleResize);
    //         if (chartInstance.current) {
    //             chartInstance.current.dispose();
    //         }
    //     };
    // }, []);

    useEffect(() => {
        console.log("尝试获取数据");

        const currentMessage = messages.filter(msg => msg.status === 13)
        console.log("从服务器获取到的数据", currentMessage);

        if (currentMessage.length === 0) return;
        console.log("从服务器获取到的数据", currentMessage[0]?.response);
        //把收到的二进制数据转成url
        if (currentMessage[0].response) {
            //     try {
            //         // 判断响应数据类型并转换为 Blob
            //         const blob = new Blob([currentMessage[0].response], { type: 'image/jpeg' });
            //         // 生成 URL
            //         const imageUrl = URL.createObjectURL(blob);
            //         console.log("Url", imageUrl);

            //         setTrafficData(imageUrl);
            //         // 清理函数
            //         return () => {
            //             URL.revokeObjectURL(imageUrl);
            //         };
            //     } catch (error) {
            //         console.error("处理二进制数据时出错:", error);
            //     }
            try {
                // 检查是否是 Base64 格式
                const base64Data = currentMessage[0].response;

                // 如果是 Base64 字符串，需要添加正确的 MIME 类型前缀
                let imageUrl;
                if (base64Data.startsWith('data:image/')) {
                    // 已经包含 MIME 类型前缀
                    imageUrl = base64Data;
                } else {
                    // 添加 PNG 格式的 MIME 类型前缀
                    imageUrl = `data:image/png;base64,${base64Data}`;
                }

                setTrafficData(imageUrl);
            } catch (error) {
                console.error("处理 Base64 数据时出错:", error);
            }
        }
        // setTrafficData(currentMessage[0].response);
    }, [messages]);

    // useEffect(() => {
    //     if (chartInstance.current && trafficData) {
    //         updateChart();
    //     }
    // }, [trafficData]);

    // const processData = () => {
    //     if (!trafficData || !Array.isArray(trafficData) || trafficData.length !== 200) {
    //         // 返回示例数据如果数据格式不正确
    //         return null;
    //     }

    //     const processedData = [];
    //     for (let i = 0; i < 200; i++) {
    //         if (!Array.isArray(trafficData[i]) || trafficData[i].length !== 200) {
    //             // 如果某一行数据格式不正确，返回示例数据
    //             return null;
    //         }

    //         for (let j = 0; j < 200; j++) {
    //             processedData.push([i, j, trafficData[i][j]]);
    //         }
    //     }

    //     return processedData;
    // };

    // const updateChart = () => {
    //     const data = processData();

    //     chartInstance.current.setOption({
    //         series: [{
    //             data: data
    //         }]
    //     });
    // };

    // const renderChart = () => {
    //     const data = processData();

    //     const option = {
    //         title: {
    //             text: '人群密度图',
    //             textStyle: {
    //                 color: 'rgba(255, 255, 255, 0.9)',
    //                 fontSize: 16,
    //                 fontWeight: 300
    //             },
    //             left: 'center',
    //             top: 10
    //         },
    //         tooltip: {
    //             position: 'top',
    //             backgroundColor: 'rgba(30, 30, 46, 0.8)',
    //             borderColor: 'rgba(255, 255, 255, 0.1)',
    //             borderWidth: 1,
    //             textStyle: {
    //                 color: 'rgba(255, 255, 255, 0.9)'
    //             },
    //             formatter: function (params) {
    //                 return `位置: [${params.data[0]}, ${params.data[1]}]<br/>密度值: ${params.data[2]}`;
    //             }
    //         },
    //         xAxis: {
    //             type: 'category',
    //             data: Array.from({ length: 200 }, (_, i) => i),
    //             splitArea: {
    //                 show: true
    //             },
    //             axisLine: {
    //                 lineStyle: {
    //                     color: 'rgba(255, 255, 255, 0.3)'
    //                 }
    //             },
    //             axisLabel: {
    //                 color: 'rgba(255, 255, 255, 0.7)',
    //                 fontSize: 10
    //             },
    //             axisTick: {
    //                 lineStyle: {
    //                     color: 'rgba(255, 255, 255, 0.3)'
    //                 }
    //             }
    //         },
    //         yAxis: {
    //             type: 'category',
    //             data: Array.from({ length: 200 }, (_, i) => i),
    //             splitArea: {
    //                 show: true
    //             },
    //             axisLine: {
    //                 lineStyle: {
    //                     color: 'rgba(255, 255, 255, 0.3)'
    //                 }
    //             },
    //             axisLabel: {
    //                 color: 'rgba(255, 255, 255, 0.7)',
    //                 fontSize: 10
    //             },
    //             axisTick: {
    //                 lineStyle: {
    //                     color: 'rgba(255, 255, 255, 0.3)'
    //                 }
    //             }
    //         },
    //         visualMap: {
    //             min: 0,
    //             max: 0.5,
    //             calculable: true,
    //             orient: 'horizontal',
    //             left: 'center',
    //             bottom: '15%',
    //             textStyle: {
    //                 color: 'rgba(255, 255, 255, 0.7)',
    //                 fontSize: 10
    //             },
    //             inRange: {
    //                 color: [
    //                     '#313695',
    //                     '#4575b4',
    //                     '#74add1',
    //                     '#abd9e9',
    //                     '#e0f3f8',
    //                     '#ffffbf',
    //                     '#fee090',
    //                     '#fdae61',
    //                     '#f46d43',
    //                     '#d73027',
    //                     '#a50026'
    //                 ]
    //             }
    //         },
    //         series: [{
    //             name: '人群密度',
    //             type: 'heatmap',
    //             data: data,
    //             label: {
    //                 show: false
    //             },
    //             emphasis: {
    //                 itemStyle: {
    //                     shadowBlur: 10,
    //                 }
    //             },
    //             smooth: true
    //         }],
    //         grid: {
    //             left: '5%',
    //             right: '5%',
    //             top: '20%',
    //             bottom: '30%',
    //             containLabel: true
    //         }
    //     };

    //     chartInstance.current.setOption(option);
    // };

    // 生成示例数据的函数
    // const generateSampleData = () => {
    //     const data = [];
    //     for (let i = 0; i < 200; i++) {
    //         for (let j = 0; j < 200; j++) {
    //             data.push([i, j, Math.floor(Math.random() * 100)]);
    //         }
    //     }
    //     return data;
    // };



    // const handleSendMessage = () => {
    //     if (status === 'connected') {
    //         sendMessage(
    //             JSON.stringify(
    //                 {
    //                     status: 13,
    //                     message: {
    //                         id: 1
    //                     }
    //                 }
    //             ));
    //         console.log("发送成功");
    //     }
    // };


    return (
        <div className={styles.container}>
            {/* 添加动态背景元素以更好地展示毛玻璃效果 */}
            <div className={styles.background} />

            {/* 毛玻璃容器 */}
            <div className={styles.glassContainer}>
                <div
                    // ref={chartRef}
                    className={styles.chart}
                >
                    {trafficData ? (
                        <img
                            src={trafficData}
                            alt="热力图"
                            style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                                objectPosition: 'center',
                                display: 'block'
                            }}
                        />
                    ) : (
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            width: '100%',
                            height: '100%',
                            color: 'rgba(255, 255, 255, 0.7)',
                            fontSize: '16px',
                            backgroundColor: 'rgba(0, 0, 0, 0.1)'
                        }}>
                            暂时无图片
                        </div>
                    )}
                </div>
                {/* <button onClick={handleSendMessage}> 发送信息</button> */}
            </div>

        </div>
    );
};

export default DensityMap;