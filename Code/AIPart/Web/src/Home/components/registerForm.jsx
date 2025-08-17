import { Form, Input, Button, Space, message } from "antd";
import { useState, useEffect } from "react";
import { registerAPI, getCodeAPI } from "../../api/service/registerService";
// import '@ant-design/v5-patch-for-react-19';
import { encryptWithAESAndRSA } from "../../../util/entrypt";
import { jsxs } from "react/jsx-runtime";

const RegisterForm = ({ onTabChange }) => {
    const [form] = Form.useForm();
    const [countdown, setCountdown] = useState(0);

    //加密公钥
    const publicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsQBfHrU7NYVB8l0kmD79ayRbS2Nmu0gOKIg177flG/MiZd5TIYuH+eOINrFFgu6K1jmTqeUDw5Lm2SPofC1fV++V6yhJu8Vveaa0WhFElSrp5F4vsZ34HB7kpZmH6Vp/u9tdohDrXe+cVdO74ILxsw9CLpEpFrFHmgThVSKtNfwCExZeOT5lN6UKgsxp+HIFbhKWF9NMpmeYw5ie10YevN9Fq9x11aeg+ZgKct1GzF9RfOcX0h6Mz4xu45q5bWRQS+djvprBS5tvYOCVZj9KEanltbFFq71PmiQLdkH7imCFtwHPZzK5TAYeknH+raSjlGDMsijs+I8tR8XpuQcXtwIDAQAB
-----END PUBLIC KEY-----`;
    //注册API
    const handleRegister = async (values) => {
        const { encryptedData, encryptedKey } = encryptWithAESAndRSA(
            JSON.stringify({
                users: {
                    email: values.email,
                    password: values.password
                },
                code: values.code
            }),
            publicKey
        );
        const response = await registerAPI(encryptedData, encryptedKey);

        if (response.status === 200) {
            message.success("注册成功");
            onTabChange?.('1'); // 切换到登录 tab
        } else {
            message.error("验证码错误");
        }
    };

    //获取验证码
    const handleGetCode = async () => {

        const email = form.getFieldValue("email");

        if (!email) {
            message.error("请输入邮箱");
            return;
        }
        console.log("第一步", email);

        // 使用 encryptWithAESAndRSA 函数加密邮箱
        const { encryptedData, encryptedKey } = encryptWithAESAndRSA(JSON.stringify({ email: email }), publicKey);
        const response = await getCodeAPI(encryptedData, encryptedKey);

        if (response.status === 200) {
            setCountdown(60);
            message.success("验证码已发送");
        } else {
            message.error("验证码发送失败");
        }
    };


    //发送验证码的倒计时
    useEffect(() => {
        if (countdown > 0) {
            const timer = setTimeout(() => {
                setCountdown(prev => prev - 1);
            }, 1000);

            return () => clearTimeout(timer);
        }
    }, [countdown]);

    return (
        <div>
            <Form
                form={form}
                name="register"
                onFinish={handleRegister}
                layout="vertical"
                autoComplete="off"
            >
                <Form.Item
                    name="email"
                    label="邮箱"
                    rules={[{ required: true, type: "email", message: "请输入有效邮箱" }]}
                >
                    <Input placeholder="请输入邮箱" />
                </Form.Item>
                <Form.Item
                    name="password"
                    label="密码"
                    rules={[
                        { required: true, message: "请输入密码" },
                        { min: 6, message: "密码至少6个字符!" },
                    ]}
                >
                    <Input.Password placeholder="请输入密码" />
                </Form.Item>
                <Form.Item
                    name="confirmPassword"
                    label="确认密码"
                    rules={[
                        { required: true, message: "请输入确认密码" },
                        { min: 6, message: "密码至少6个字符!" },
                    ]}
                >
                    <Input.Password placeholder="请输入确认密码" />
                </Form.Item>
                <Form.Item
                    label="验证码"
                    required
                    rules={[{ required: true, message: "请输入验证码" }]}
                >
                    <Space.Compact>
                        <Form.Item
                            name="code"
                            noStyle
                            rules={[{ required: true, message: "请输入验证码" }]}
                        >
                            <Input.OTP
                                formatter={(str) => str.toUpperCase()}
                                style={{ flex: 1 }}
                            />
                        </Form.Item>
                        <Button
                            type="link"
                            onClick={handleGetCode}
                            disabled={countdown > 0}
                            style={{ color: "#121212" }}
                        >
                            {countdown > 0 ? `(${countdown})秒后重新发送` : '发送验证码'}
                        </Button>
                    </Space.Compact>
                </Form.Item>

                <Form.Item>
                    <Space style={{ width: "100%" }}>
                        <Button type="primary" htmlType="submit" block
                            style={{ backgroundColor: '#121212' }}>
                            注册
                        </Button>
                        <Button htmlType="button">
                            {/* onClick={onReset} */}
                            重置
                        </Button>
                    </Space>
                </Form.Item>
            </Form>
        </div>
    );
};

export default RegisterForm;