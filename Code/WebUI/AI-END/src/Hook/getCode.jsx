import { useState, useEffect } from "react";
import { message } from "antd";
import '@ant-design/v5-patch-for-react-19';
import { getCodeAPI } from "../api/service/registerService"; // Adjust the import path as necessary

//=====复用获取验证码逻辑的hook=====
export const useCode = () => {
    const [countdown, setCountdown] = useState(0);

    //点击发送验证码获取验证码
    const sendCode = async (email) => {
        if (!email) {
            message.error("请输入邮箱");
            return;
        }

        const response = await getCodeAPI(email);
        if (response.status === 200) {
            setCountdown(60);
            message.success("验证码已发送");
        } else {
            message.error("验证码发送失败");
        }
    };

    //点击发送验证码的倒计时
    useEffect(() => {
        if (countdown > 0) {
            const timer = setTimeout(() => setCountdown(c => c - 1), 1000);
            return () => clearTimeout(timer);
        }
    }, [countdown]);

    //返回倒计时和发送验证码的函数
    return {
        countdown,
        sendCode,
    };
};