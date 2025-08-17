import React from 'react';
import { Card, Row, Col, Typography, Button, Popconfirm } from 'antd';
import {
    SkinOutlined,
    StarOutlined,
    CloudOutlined,
    BarChartOutlined,
} from '@ant-design/icons';
import { useState, useEffect } from 'react';
import { SliderComponent } from './slider';
import { message } from 'antd';
import '@ant-design/v5-patch-for-react-19';
import styles from '../modulecss/weather.module.css';
import { useWebSocketContext } from '../../../Context/wsContext';

const { Title, Text } = Typography;

const WeatherDisplay = ({ weatherData }) => {
    const { messages, status, sendMessage } = useWebSocketContext();
    const [currentData, setcurrentData] = useState(weatherData);
    useEffect(() => {
        setcurrentData(weatherData);
    }, [weatherData]);

    // 空气质量指数颜色和等级
    const getAirQualityLevel = (aqi) => {
        if (aqi <= 30) return { level: '差', color: '#fa4a14ff' };
        if (aqi <= 80) return { level: '良', color: '#faad14' };
        if (aqi <= 100) return { level: '优', color: '#52c41a' };
    };
    const aqiInfo = getAirQualityLevel(currentData.airQuality);

    //温度指数
    const getTemperatureLevel = (temperature) => {
        if (temperature <= 0) return { level: '寒冷', color: '#14b1faff' };
        if (temperature <= 25) return { level: '凉爽', color: '#faad14' };
        if (temperature <= 40) return { level: '炎热', color: '#fa4a14ff' };
    }
    const temInfo = getTemperatureLevel(currentData.temperature);

    //风力等级指数
    const getWindLevel = (windSpeed) => {
        console.log("fengsu", windSpeed);

        if (windSpeed <= 30) return { level: '微风', color: '#14b1faff' };
        if (windSpeed <= 80) return { level: '中风', color: '#14fa46ff' };
        if (windSpeed <= 100) return { level: '强风', color: '#fa4a14ff' };
    }
    const windInfo = getWindLevel(currentData.wind);

    //降水量等级
    const getRainLevel = (rain) => {
        if (rain <= 0) return { level: '无雨', color: ' #fa4a14ff' };
        if (rain <= 50) return { level: '小雨', color: '#14f2faff' };
        if (rain <= 80) return { level: '中雨', color: '#1478faff' };
        if (rain <= 100) return { level: '大雨', color: '#1423faff' }
    }
    const rainInfo = getRainLevel(currentData.rain);



    //处理滑动条的数据
    const handleChange = (type, value) => {
        switch (type) {
            case 'temperature':
                setcurrentData(
                    { ...currentData, temperature: value }
                )
                break;
            case 'wind':
                setcurrentData(
                    { ...currentData, wind: value }
                )
                break;
            case 'rain':
                setcurrentData(
                    { ...currentData, rain: value }
                )
                break;
            case 'airQuality':
                setcurrentData(
                    { ...currentData, airQuality: value }
                )
                break;
            default:
                break;
        }
    };
    const handleUpdate = () => {
        if (status === 'connected') {
            sendMessage(
                JSON.stringify(
                    {
                        status: 2,
                        message: currentData
                    }
                ));
            message.success("更新成功")
        }

    }

    return (
        <Card
            title={
                <div>
                    天气情况

                    <Button
                        type="primary"
                        onClick={handleUpdate}
                        style={{
                            marginLeft: "10px",
                            backgroundColor: ' rgba(255, 255, 255, 0.1)',
                            color: '#00d9ff'
                        }}
                    >更新</Button>

                </div>
            }

            style={{ backgroundColor: 'transparent', border: 'none', height: '100%', width: '100%' }}
            headStyle={
                {
                    color: 'rgba(255, 255, 255, 0.7)',
                    borderBottom: "1px solid rgba(255, 255, 255, 0.2)",
                    height: '20%',
                    minHeight: '0px',
                    fontSize: '100%',
                    padding: '2px 10px',
                }
            }
            bodyStyle={
                {
                    height: '80%',
                }
            }

        >
            <Row gutter={[36, 16]} style={{ height: '100%' }}>
                {/* 温度 */}
                <Col xs={24} sm={12} md={6} style={
                    {
                        height: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }
                }
                >
                    <div style={{ textAlign: 'center', width: '100%' }}>
                        <SkinOutlined style={{ fontSize: '100%', color: temInfo.color }} />
                        <Title level={3} style={{
                            margin: '8px 0',
                            width: '100%',
                            textAlign: 'center',
                            fontFamily: 'monospace',
                            fontSize: "100%",
                            color: temInfo.color
                        }}>
                            {currentData.temperature}°C
                        </Title>
                        <Text type="secondary" style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: "80%" }}>体感温度</Text>
                        <SliderComponent min={-10} max={40} currentData={currentData.temperature} handleChange={handleChange} type={'temperature'} />
                    </div>
                </Col>

                {/* 风力 */}
                <Col xs={24} sm={12} md={6} style={
                    {
                        height: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                    <div style={{ textAlign: 'center', width: '100%' }}>
                        <BarChartOutlined style={{ fontSize: '100%', color: windInfo.color }} />
                        <Title level={3} style={{
                            margin: '8px 0',
                            width: '100%',
                            textAlign: 'center',
                            fontFamily: 'monospace',
                            fontSize: "100%",
                            color: windInfo.color
                        }}>
                            {currentData.wind}级
                        </Title>
                        <Text type="secondary" style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: "80%" }}>风力等级</Text>
                        <SliderComponent min={0} max={100} currentData={currentData.wind} handleChange={handleChange} type={'wind'} />
                    </div>
                </Col>

                {/* 降水量 */}
                <Col xs={24} sm={12} md={6} style={
                    {
                        height: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                    <div style={{ textAlign: 'center', width: '100%' }}>
                        <CloudOutlined style={{
                            fontSize: '100%',
                            color: rainInfo.color
                        }} />
                        <Title level={3} style={{
                            margin: '8px 0',
                            width: '100%',
                            textAlign: 'center',
                            fontFamily: 'monospace',
                            fontSize: "100%",
                            color: rainInfo.color
                        }}>
                            {currentData.rain}mm
                        </Title>
                        <Text type="secondary" style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: "80%" }}>降水量 </Text>
                        <SliderComponent min={0} max={100} currentData={currentData.rain} handleChange={handleChange} type={'rain'} />
                    </div>
                </Col>

                {/* 空气质量 */}
                <Col xs={24} sm={12} md={6} style={
                    {
                        height: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                    <div style={{ textAlign: 'center', width: '100%' }}>
                        <StarOutlined style={{ fontSize: '100%', color: aqiInfo.color }} />
                        <Title level={3} style={{
                            margin: '8px 0',
                            color: aqiInfo.color,
                            width: '100%',
                            textAlign: 'center',
                            fontFamily: 'monospace',
                            fontSize: "100%"
                        }}>
                            {currentData.airQuality} {aqiInfo.level}
                        </Title>
                        <Text type="secondary" style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: "80%" }}>空气质量</Text>
                        <SliderComponent min={0} max={100} currentData={currentData.airQuality} handleChange={handleChange} type={'airQuality'} />
                    </div>
                </Col>
            </Row>
        </Card >
    );
};




export default WeatherDisplay;