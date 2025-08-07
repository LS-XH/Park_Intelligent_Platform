import { Form, Input, Button, Space, message } from "antd";
import { useState, useEffect } from "react";
// import '@ant-design/v5-patch-for-react-19';

const RegisterForm = () => {
    const [form] = Form.useForm();
    const [countdown, setCountdown] = useState(0);
    //注册API
    const handleRegister = async (values) => {
        const response = await registerAPI(values.email, values.password, values.verificationCode);
        if (response.status === 200) {
            message.success("注册成功");
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
        const response = await getCodeAPI(email);

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
                        >
                            {countdown > 0 ? `(${countdown})秒后重新发送` : '发送验证码'}
                        </Button>
                    </Space.Compact>
                </Form.Item>

                <Form.Item>
                    <Space style={{ width: "100%" }}>
                        <Button type="primary" htmlType="submit" block
                            style={{ backgroundColor: '#1890ff' }}>
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