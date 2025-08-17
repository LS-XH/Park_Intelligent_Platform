import React from 'react';
import WeatherDisplay from './weather.jsx';
import styles from '../modulecss/weather.module.css';
import { useEffect, useState } from 'react';
import { useWebSocketContext } from "../../../Context/wsContext";


const WeatherCard = () => {

    const { sendMessage, messages, status } = useWebSocketContext();
    const [weatherData, setWeatherData] = useState({
        temperature: 28,
        wind: 2,
        rain: 0.5,
        airQuality: 78,
    });
    //处理接收到的天气数据
    useEffect(() => {
        const currentMessage = messages.filter(msg => msg.status === 1)
        if (currentMessage.length === 0) return;
        console.log("从服务器获取到的天气数据", currentMessage[0]?.response);
        setWeatherData(currentMessage[0].response);
    }, [messages]);

    useEffect(() => {
        // setTimeout(() => { 
        // }, 1000);
        if (status === 'connected') {
            sendMessage(
                JSON.stringify(
                    {
                        status: 1,
                        message: {
                            id: 1
                        }
                    }
                ));
            console.log("发送成功");
        }
    }, [status]);
    const handleSendMessage = () => {
        // 发送消息给服务器
        console.log("点击按钮");

    }


    return (
        <div
            className={styles.container}
        >
            <div className={styles.background} />
            <div className={styles.glassContainer}>
                <WeatherDisplay weatherData={weatherData} status={status} sendMessage={sendMessage} />
            </div>
        </div>
    );
};

export default WeatherCard;